"""Dense Qwen3 with nanochat's training/KV-cache interface and FA4 backend.

Architecture follows Qwen/Qwen3-1.7B and Transformers v4.57.1 Qwen3:
learned RMSNorm, Q/K head norms before RoPE, GQA, SwiGLU, ordinary residuals,
full causal attention, and (by default) tied embeddings. The tokenizer, depth/width
ladder and Muon+AdamW training recipe are this project's choices, not Qwen weights.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.flash_attention import flash_attn
from nanochat.gpt import Linear, GPT
from nanochat.optim import MuonAdamW


@dataclass
class Qwen3Config:
    architecture: str = 'qwen3'
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 6
    n_kv_head: int = 3
    head_dim: int = 128
    intermediate_size: int = 2304
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    window_pattern: str = 'L'

    def __post_init__(self):
        if self.architecture != 'qwen3' or self.window_pattern != 'L':
            raise ValueError('Qwen3 uses full causal attention: --window-pattern=L')
        if min(self.sequence_len, self.vocab_size, self.n_layer, self.n_embd,
               self.n_head, self.n_kv_head, self.head_dim, self.intermediate_size) <= 0:
            raise ValueError('Qwen3 dimensions must be positive')
        if self.n_head % self.n_kv_head or self.head_dim % 2:
            raise ValueError('query heads must divide into KV groups; head_dim must be even')
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError('RMSNorm epsilon and RoPE theta must be positive')


def config_for_depth(depth, *, vocab_size=32768, sequence_len=2048, aspect_ratio=64,
                     head_dim=128, kv_group_size=2, mlp_ratio=3.0, rope_theta=1e6,
                     tie_word_embeddings=True, window_pattern='L'):
    """One shared shape rule for the trainer and the ladder planner.

    Residual width = ceil(64*d / head_dim)*head_dim by default. Query heads are
    rounded up to complete GQA groups; attention projection width may differ from
    residual width, which Qwen3 supports (notably its official 0.6B configuration).
    """
    if min(depth, aspect_ratio, head_dim, kv_group_size) <= 0 or not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
        raise ValueError('depth, aspect ratio, head dimension, KV group size and MLP ratio must be positive')
    width = math.ceil(depth * aspect_ratio / head_dim) * head_dim
    heads = math.ceil(width / (head_dim * kv_group_size)) * kv_group_size
    intermediate = math.ceil(mlp_ratio * width / 128) * 128
    return Qwen3Config(sequence_len=sequence_len, vocab_size=vocab_size, n_layer=depth,
                       n_embd=width, n_head=heads, n_kv_head=heads // kv_group_size,
                       head_dim=head_dim, intermediate_size=intermediate, rope_theta=rope_theta,
                       tie_word_embeddings=tie_word_embeddings, window_pattern=window_pattern)


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def rotary(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.config, self.layer_idx = c, layer_idx
        self.q_proj = Linear(c.n_embd, c.n_head * c.head_dim, bias=False)
        self.k_proj = Linear(c.n_embd, c.n_kv_head * c.head_dim, bias=False)
        self.v_proj = Linear(c.n_embd, c.n_kv_head * c.head_dim, bias=False)
        self.o_proj = Linear(c.n_head * c.head_dim, c.n_embd, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)

    def forward(self, x, cos_sin, kv_cache):
        b, t, _ = x.shape
        c = self.config
        q = self.q_norm(self.q_proj(x).view(b, t, c.n_head, c.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, t, c.n_kv_head, c.head_dim))
        v = self.v_proj(x).view(b, t, c.n_kv_head, c.head_dim)
        cos, sin = (v.to(q.dtype) for v in cos_sin)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
        else:
            kc, vc = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=(-1, -1))
            if self.layer_idx == c.n_layer - 1:
                kv_cache.advance(t)
        return self.o_proj(y.reshape(b, t, c.n_head * c.head_dim))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = Linear(c.n_embd, c.intermediate_size, bias=False)
        self.up_proj = Linear(c.n_embd, c.intermediate_size, bias=False)
        self.down_proj = Linear(c.intermediate_size, c.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.self_attn = Attention(c, layer_idx)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.n_embd, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.n_embd, c.rms_norm_eps)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.self_attn(self.input_layernorm(x), cos_sin, kv_cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        vocab = math.ceil(config.vocab_size / pad_vocab_size_to) * pad_vocab_size_to
        # State-dict names match HF Qwen3, allowing independent reference comparisons.
        self.model = nn.ModuleDict({
            'embed_tokens': nn.Embedding(vocab, config.n_embd),
            'layers': nn.ModuleList([DecoderLayer(config, i) for i in range(config.n_layer)]),
            'norm': RMSNorm(config.n_embd, config.rms_norm_eps),
        })
        self.lm_head = Linear(config.n_embd, vocab, bias=False)
        self.tie_weights()
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._rotary_cache()
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        # to_empty/assign=True can break parameter identity, so retie after both.
        self.tie_weights()
        self.cos, self.sin = self._rotary_cache()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.tie_weights()
        return result

    def _rotary_cache(self):
        c, device = self.config, self.get_device()
        freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, device=device, dtype=torch.float32) / c.head_dim))
        angles = torch.outer(torch.arange(self.rotary_seq_len, device=device, dtype=torch.float32), freq)
        return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]

    def get_device(self):
        return self.model.embed_tokens.weight.device

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        offset = 0 if kv_cache is None else kv_cache.get_pos()
        length = idx.shape[1]
        assert offset + length <= self.rotary_seq_len, 'sequence exceeds RoPE cache'
        cs = self.cos[:, offset:offset + length], self.sin[:, offset:offset + length]
        x = self.model.embed_tokens(idx).to(COMPUTE_DTYPE)
        for layer in self.model.layers:
            x = layer(x, cs, kv_cache)
        logits = self.lm_head(self.model.norm(x))[..., :self.config.vocab_size].float()
        if targets is None:
            return logits
        # nanochat's dataloader already shifts targets; do not shift a second time.
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-1, reduction=loss_reduction)

    def num_scaling_params(self):
        matrix = sum(m.weight.numel() for layer in self.model.layers for m in layer.modules() if isinstance(m, nn.Linear))
        norms = sum(m.weight.numel() for m in self.modules() if isinstance(m, RMSNorm))
        # Count tied weights once, under the head (they participate in its matmul).
        head = self.lm_head.weight.numel()
        embed = 0 if self.config.tie_word_embeddings else self.model.embed_tokens.weight.numel()
        counts = dict(wte=embed, value_embeds=0, lm_head=head, transformer_matrices=matrix,
                      scalars=norms, total=embed + head + matrix + norms)
        assert counts['total'] == sum(p.numel() for p in self.parameters())
        return counts

    def num_matmul_params(self):
        c = self.num_scaling_params()
        return c['transformer_matrices'] + c['lm_head']

    def estimate_flops(self):
        c = self.config
        return 6 * self.num_matmul_params() + 12 * c.n_layer * c.n_head * c.head_dim * c.sequence_len

    def estimate_decode_flops(self, context_len):
        c = self.config
        return 2 * self.num_matmul_params() + 4 * c.n_layer * c.n_head * c.head_dim * context_len

    def estimate_prefill_flops(self, num_tokens):
        c = self.config
        return 2 * self.num_matmul_params() * num_tokens + 4 * c.n_layer * c.n_head * c.head_dim * num_tokens * (num_tokens + 1) // 2

    def kv_bytes_per_token(self):
        c = self.config
        return 2 * c.n_layer * c.n_kv_head * c.head_dim * COMPUTE_DTYPE.itemsize

    def kv_read_bytes(self, context_len):
        return self.kv_bytes_per_token() * context_len

    def setup_optimizer(self, unembedding_lr=0.002, embedding_lr=0.3, matrix_lr=0.005,
                        weight_decay=0.28, scalar_lr=0.1):
        scale = (self.config.n_embd / 768) ** -0.5
        matrices = [m.weight for layer in self.model.layers for m in layer.modules() if isinstance(m, nn.Linear)]
        norms = [m.weight for m in self.modules() if isinstance(m, RMSNorm)]
        groups = [dict(kind='adamw', params=[self.lm_head.weight], lr=unembedding_lr * scale,
                       betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
                  dict(kind='adamw', params=norms, lr=scalar_lr * 0.01,
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)]
        if not self.config.tie_word_embeddings:
            groups.append(dict(kind='adamw', params=[self.model.embed_tokens.weight], lr=embedding_lr * scale,
                               betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001))
        else:
            print0('Tied Qwen3 embedding/head uses --unembedding-lr; --embedding-lr applies only when untied')
        for shape in sorted({p.shape for p in matrices}):
            groups.append(dict(kind='muon', params=[p for p in matrices if p.shape == shape],
                               lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
        assigned = [p for g in groups for p in g['params']]
        assert len(assigned) == len({id(p) for p in assigned}) == len(list(self.parameters()))
        optimizer = MuonAdamW(groups)
        for g in optimizer.param_groups:
            g['initial_lr'] = g['lr']
        return optimizer

    generate = GPT.generate

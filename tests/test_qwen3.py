"""Architecture parity against an independent Qwen3 implementation, plus integration contracts."""
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

import nanochat.qwen3 as native
import nanochat.flash_attention as attention
from nanochat.engine import KVCache


@pytest.fixture(autouse=True)
def cpu_fp32(monkeypatch):
    monkeypatch.setattr(native, 'COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr(attention, 'USE_FA4', False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        yield


def small_config(tied=True):
    # Projection width deliberately differs from residual width, as in d10/d14.
    return native.Qwen3Config(vocab_size=64, sequence_len=32, n_layer=2,
        n_embd=32, n_head=4, n_kv_head=2, head_dim=16, intermediate_size=96,
        tie_word_embeddings=tied)


@pytest.mark.parametrize('tied', [True, False])
def test_huggingface_logits_and_gradients(tied):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    c = small_config(tied)
    model = native.Qwen3(c)
    model.init_weights()
    reference = Qwen3ForCausalLM(Qwen3Config(vocab_size=c.vocab_size,
        hidden_size=c.n_embd, intermediate_size=c.intermediate_size,
        num_hidden_layers=c.n_layer, num_attention_heads=c.n_head,
        num_key_value_heads=c.n_kv_head, head_dim=c.head_dim,
        rms_norm_eps=c.rms_norm_eps, rope_theta=c.rope_theta,
        tie_word_embeddings=tied, attention_dropout=0.0,
        _attn_implementation='eager'))
    reference.load_state_dict(model.state_dict(), strict=True)
    x, y = torch.randint(64, (2, 13)), torch.randint(64, (2, 13))
    actual = model(x)
    expected = reference(x, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    loss = model(x, y)
    reference_loss = F.cross_entropy(expected.reshape(-1, 64), y.reshape(-1))
    torch.testing.assert_close(loss, reference_loss)
    loss.backward()
    reference_loss.backward()
    expected_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, expected_params[name].grad, atol=3e-6, rtol=3e-4,
                                   msg=lambda msg: f'{name}: {msg}')


def test_causal_kv_cache_prefill_decode_and_branch():
    c = small_config()
    model = native.Qwen3(c).eval()
    model.init_weights()
    def cache(batch):
        return KVCache(batch, c.n_kv_head, 32, c.head_dim, c.n_layer, 'cpu', torch.float32)
    x = torch.randint(64, (1, 11))
    with torch.no_grad():
        full = model(x)
        kv = cache(1)
        pieces = [model(x[:, :5], kv_cache=kv)]
        for i in range(5, 11):
            pieces.append(model(x[:, i:i+1], kv_cache=kv))
        torch.testing.assert_close(torch.cat(pieces, 1), full, atol=2e-6, rtol=2e-5)
        assert kv.get_pos() == 11
        prefix = cache(1)
        model(x[:, :5], kv_cache=prefix)
        branch = cache(2)
        branch.prefill(prefix)
        torch.testing.assert_close(model(x[:, 5:].expand(2, -1), kv_cache=branch),
                                   full[:, 5:].expand(2, -1, -1), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('tied', [True, False])
def test_checkpoint_roundtrip_preserves_logits_and_weight_tying(tmp_path, monkeypatch, tied):
    from nanochat import checkpoint_manager as checkpoint
    c = small_config(tied)
    with torch.device('meta'):
        model = native.Qwen3(c)
    model.to_empty(device='cpu')
    model.init_weights()
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tied
    checkpoint.save_checkpoint(tmp_path, 3, model.state_dict(), None, {'model_config': asdict(c)})
    class Tokenizer:
        def get_vocab_size(self):
            return 64
    monkeypatch.setattr(checkpoint, 'get_tokenizer', Tokenizer)
    restored, _, _ = checkpoint.build_model(tmp_path, 3, torch.device('cpu'), 'eval')
    assert isinstance(restored, native.Qwen3)
    assert (restored.lm_head.weight is restored.model.embed_tokens.weight) == tied
    x = torch.randint(64, (2, 9))
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_depth_width_scaling_and_size_limit():
    previous = 0
    for depth in (8, 10, 12, 14, 16, 18, 20, 24, 26, 28, 32):
        c = native.config_for_depth(depth)
        assert c.n_layer == depth and c.n_embd == 64 * depth
        assert c.n_head == 2 * c.n_kv_head
        assert c.intermediate_size == 3 * c.n_embd
        with torch.device('meta'):
            model = native.Qwen3(c)
        total = model.num_scaling_params()['total']
        assert previous < total <= 1.7e9
        previous = total

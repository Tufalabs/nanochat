"""
Unified Flash Attention interface with automatic FA4/SDPA switching.

Exports `flash_attn` module that matches the FA3-style API used by gpt.py, but uses
FlashAttention-4 (CuTe DSL, JIT-compiled, pip package `flash-attn-4`) on Hopper/Blackwell
and falls back to PyTorch SDPA everywhere else (other GPUs, MPS, CPU, non-bf16 dtypes).

Usage:
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache) -> always SDPA (FA4 has no kvcache API we use)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import os
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA4 on CUDA GPUs
# =============================================================================
def _load_flash_attention_4():
    """Try to load FlashAttention-4 (CuTe DSL). Returns its interface module or None."""
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    # FA4 training kernels (fwd+bwd) target Hopper (sm90) and datacenter Blackwell (sm100/sm103/sm110)
    if major not in (9, 10, 11):
        return None
    try:
        from flash_attn.cute import interface as fa4_interface
        return fa4_interface
    except Exception:
        return None


_fa4 = _load_flash_attention_4()
HAS_FA4 = _fa4 is not None

# Override for testing: set to 'fa4', 'sdpa', or None (auto). Also settable via env: NANOCHAT_ATTN=fa4|sdpa
_override_impl = os.environ.get("NANOCHAT_ATTN") or None


def _resolve_use_fa4():
    """Decide once whether to use FA4, based on availability, override, and dtype."""
    if _override_impl == 'fa4':
        assert HAS_FA4, "Cannot override to FA4: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA4:
        # FA4 kernels only support bf16/fp16 (and fp8 fwd); nanochat only routes bf16 to FA4
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA4 = _resolve_use_fa4()


# =============================================================================
# FA4 as torch custom ops, so torch.compile treats attention as one opaque op
# (FA4's own autograd.Function is not traceable by dynamo => graph breaks every layer)
# window_left < 0 means unlimited (plain causal); otherwise causal sliding window.
# =============================================================================
def _fa4_window(window_left):
    return None if window_left < 0 else window_left

@torch.library.custom_op("nanochat::fa4_fwd", mutates_args=(), device_types="cuda")
def _fa4_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _fa4._flash_attn_fwd(
        q, k, v, causal=True,
        window_size_left=_fa4_window(window_left), window_size_right=0,
        return_lse=True,
    )
    return out, lse

@_fa4_fwd.register_fake
def _(q, k, v, window_left):
    B, T, H, _ = q.shape
    out = q.new_empty((B, T, H, v.size(-1)))
    lse = q.new_empty((B, H, T), dtype=torch.float32)
    return out, lse

@torch.library.custom_op("nanochat::fa4_bwd", mutates_args=(), device_types="cuda")
def _fa4_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             out: torch.Tensor, lse: torch.Tensor, window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _fa4._flash_attn_bwd(
        q, k, v, out, dout, lse, causal=True,
        window_size_left=_fa4_window(window_left), window_size_right=0,
    )
    return dq, dk, dv

@_fa4_bwd.register_fake
def _(dout, q, k, v, out, lse, window_left):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

def _fa4_setup_context(ctx, inputs, output):
    q, k, v, window_left = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.window_left = window_left

def _fa4_backward(ctx, dout, dlse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _fa4_bwd(dout.contiguous(), q, k, v, out, lse, ctx.window_left)
    return dq, dk, dv, None

_fa4_fwd.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA4 and causal:
        window = window_size[0]
        # a left window that covers the whole sequence is plain causal attention (faster FA4 path)
        window_left = -1 if (window < 0 or window >= q.size(1) - 1) else window
        out, _ = _fa4_fwd(q, k, v, window_left)
        return out

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Attention with KV cache for inference (SDPA, updates k_cache/v_cache in-place like FA3).

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)

"""Attention outputs/gradients against a dense oracle, and cached decoding contracts."""
import math

import pytest
import torch

import nanochat.flash_attention as attention
from nanochat.engine import KVCache


def dense_causal_attention(q, k, v, window=-1):
    """Reference using neither SDPA nor the production mask/cache helpers.

    Repeat KV heads to express GQA as ordinary attention, then evaluate one
    query position at a time against its allowed prefix.
    """
    groups = q.size(2) // k.size(2)
    k, v = (x.repeat_interleave(groups, dim=2) for x in (k, v))
    outputs = []
    for pos in range(q.size(1)):
        start = 0 if window < 0 else max(0, pos - window)
        keys, values = k[:, start:pos + 1], v[:, start:pos + 1]
        scores = torch.einsum('bhd,bthd->bht', q[:, pos], keys) / math.sqrt(q.size(-1))
        outputs.append(torch.einsum('bht,bthd->bhd', scores.softmax(-1), values))
    return torch.stack(outputs, dim=1)


@pytest.fixture(autouse=True)
def deterministic_inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        yield


@pytest.mark.parametrize('kv_heads', [2, 4])
@pytest.mark.parametrize('window', [-1, 3, 32])
def test_sdpa_outputs_and_gradients_against_dense_attention(monkeypatch, kv_heads, window):
    monkeypatch.setattr(attention, 'USE_FA4', False)
    q = torch.randn(2, 12, 4, 8, requires_grad=True)
    k, v = (torch.randn(2, 12, kv_heads, 8, requires_grad=True) for _ in range(2))
    actual = attention.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))
    expected = dense_causal_attention(q, k, v, window)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('window', [-1, 3])
def test_cache_prefill_decode_and_chunk_match_dense_attention(monkeypatch, window):
    # Cached inference must work on CPU even when training's FA4 backend is selected.
    monkeypatch.setattr(attention, 'USE_FA4', True)
    q = torch.randn(2, 10, 4, 8)
    k, v = (torch.randn(2, 10, 2, 8) for _ in range(2))
    expected = dense_causal_attention(q, k, v, window)
    cache = KVCache(2, 2, 16, 8, 1, 'cpu', torch.float32)
    kc, vc = cache.get_layer_cache(0)
    kc.fill_(float('nan'))
    vc.fill_(float('nan'))  # Unwritten cache slots must never contribute.
    pos = 0
    for length in (5, 1, 4):
        end = pos + length
        actual = attention.flash_attn_with_kvcache(
            q[:, pos:end], kc, vc, k=k[:, pos:end], v=v[:, pos:end],
            cache_seqlens=cache.cache_seqlens, causal=True, window_size=(window, 0))
        torch.testing.assert_close(actual, expected[:, pos:end])
        torch.testing.assert_close(kc[:, :end], k[:, :end])
        torch.testing.assert_close(vc[:, :end], v[:, :end])
        assert kc[:, end:].isnan().all() and vc[:, end:].isnan().all()
        cache.advance(length)
        pos = end


@pytest.mark.skipif(not attention.HAS_FA4, reason='FlashAttention-4 and a supported CUDA GPU required')
@pytest.mark.parametrize('kv_heads,window', [(4, -1), (2, -1), (4, 7)])
def test_fa4_outputs_and_gradients_against_dense_attention(monkeypatch, kv_heads, window):
    monkeypatch.setattr(attention, 'USE_FA4', True)
    # head_dim=128 is the training configuration; cover MHA, GQA and legacy windows.
    q = torch.randn(2, 32, 4, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k, v = (torch.randn(2, 32, kv_heads, 128, device='cuda', dtype=torch.bfloat16,
                        requires_grad=True) for _ in range(2))
    actual = attention.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))
    reference_inputs = [x.detach().float().requires_grad_(True) for x in (q, k, v)]
    expected = dense_causal_attention(*reference_inputs, window)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream.float())
    torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.025)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        relative_error = (actual_grad.float() - expected_grad).norm() / expected_grad.norm()
        assert relative_error < 0.02, f'relative L2 gradient error: {relative_error}'

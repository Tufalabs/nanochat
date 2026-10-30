"""Compiled Qwen3 training agrees with eager SDPA and updates all model parameters."""
import pytest
import torch
import nanochat.qwen3 as native
import nanochat.flash_attention as attention


@pytest.mark.skipif(not attention.HAS_FA4, reason='FlashAttention-4 and a supported CUDA GPU required')
@pytest.mark.parametrize('tied', [True, False])
def test_fa4_compiled_training_step(monkeypatch, tied):
    torch.manual_seed(23)
    monkeypatch.setattr(native, 'COMPUTE_DTYPE', torch.bfloat16)
    c = native.Qwen3Config(vocab_size=128, sequence_len=128, n_layer=2,
        n_embd=256, n_head=4, n_kv_head=2, head_dim=128, intermediate_size=768,
        tie_word_embeddings=tied)
    model = native.Qwen3(c).cuda()
    model.init_weights()
    x, y = torch.randint(128, (2, 128), device='cuda'), torch.randint(128, (2, 128), device='cuda')
    monkeypatch.setattr(attention, 'USE_FA4', False)
    reference = model(x, y)
    reference.backward()
    reference_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(attention, 'USE_FA4', True)
    compiled = torch.compile(model, fullgraph=True)
    actual = compiled(x, y)
    actual.backward()
    torch.testing.assert_close(actual, reference, atol=0.002, rtol=0.001)
    for name, p in model.named_parameters():
        expected = reference_grads[name]
        relative_error = (p.grad - expected).norm() / expected.norm().clamp_min(1e-10)
        assert relative_error < 0.035, f'{name}: relative L2 grad error {relative_error}'
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.setup_optimizer().step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name
        assert not torch.equal(p, before[name]), f'{name} was not updated by the optimizer'

"""CPU-only checks for run-owned artifacts, checkpoint discovery and disabled tracking."""
import os
from pathlib import Path
import tempfile

import pytest

from nanochat.run_paths import configure_run_dir


@pytest.fixture
def isolated_environment(monkeypatch):
    # configure_run_dir deliberately updates process-global environment/temp paths.
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
    monkeypatch.delenv('NANOCHAT_RUN_DIR', raising=False)
    monkeypatch.delenv('NANOCHAT_DIR', raising=False)
    monkeypatch.delenv('LADDERS_DIR', raising=False)


def test_explicit_directory_contains_runtime_files(tmp_path, monkeypatch, isolated_environment):
    monkeypatch.setenv('WANDB_DIR', str(tmp_path / 'old-global-wandb'))
    run = configure_run_dir(tmp_path / 'ladder' / 'd12')
    for key in ('NANOCHAT_RUN_DIR', 'WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR',
                'WANDB_DATA_DIR', 'TMPDIR', 'NCCL_DEBUG_FILE'):
        assert Path(os.environ[key]).is_relative_to(run)
    with tempfile.NamedTemporaryFile() as f:
        assert Path(f.name).is_relative_to(run)
    assert not (tmp_path / 'old-global-wandb').exists()
    assert configure_run_dir(tag='ignored') == run


def test_unmanaged_run_gets_descriptive_folder(tmp_path, monkeypatch, isolated_environment):
    monkeypatch.setenv('PRETRAIN_ROOT', str(tmp_path))
    assert configure_run_dir(tag='qwen3-d12') == tmp_path / 'nanochat/ladders/standalone/qwen3-d12'


def test_custom_ladder_root_is_respected(tmp_path, monkeypatch, isolated_environment):
    monkeypatch.setenv('LADDERS_DIR', str(tmp_path / 'custom'))
    assert configure_run_dir(tag='d12') == tmp_path / 'custom/standalone/d12'


def test_checkpoint_discovery_and_optimizer_stay_in_run(tmp_path, monkeypatch, isolated_environment):
    import torch
    from nanochat import checkpoint_manager as cm
    run = configure_run_dir(tmp_path / 'ladder/d12')
    checkpoint = run / 'checkpoints'
    # Unsorted filenames distinguish numerical latest-step selection from directory order.
    for step in (10, 2, 9):
        (checkpoint / f'model_{step:06d}.pt').touch()
    torch.save({'counter': 2}, checkpoint / 'optim_000002_rank1.pt')
    torch.save({'counter': 10}, checkpoint / 'optim_000010_rank1.pt')
    calls = []
    def build(path, step, device, phase):
        calls.append((Path(path), step, device, phase))
        return 'model', 'tokenizer', {'step': step}
    monkeypatch.setattr(cm, 'build_model', build)
    assert cm.load_model('base', 'cpu', phase='eval') == ('model', 'tokenizer', {'step': 10})
    assert calls == [(checkpoint, 10, 'cpu', 'eval')]
    assert cm.load_optimizer_state('base', 'cpu', rank=1) == {'counter': 10}
    assert cm.load_optimizer_state('base', 'cpu', rank=1, step=2) == {'counter': 2}


def test_disabled_tracking_never_initializes_wandb(monkeypatch):
    import wandb
    from nanochat.tracking import init_run

    def forbidden(**kwargs):
        pytest.fail('disabled/worker logging must not initialize W&B')

    monkeypatch.setattr(wandb, 'init', forbidden)
    for name, master in [('dummy', True), ('test', False)]:
        run = init_run(name, {}, master_process=master)
        run.config.update({'effective_num_iterations': 40})
        run.summary.update({'completed_steps': 40})
        run.log({'step': 1, 'train/loss': 1.0})
        run.finish()

"""Run-owned outputs, separate from shared datasets and reusable compiler caches."""
import os
from pathlib import Path
import re
import tempfile


def validate_tag(tag):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', tag):
        raise ValueError('run/ladder tags must start with a letter/digit and contain only letters, digits, _, . or -')
    return tag


def configure_run_dir(path=None, *, tag=None):
    """Set all runtime artifact locations before initializing W&B or training."""
    path = path or os.environ.get('NANOCHAT_RUN_DIR')
    if not path:
        root = Path(os.environ.get('PRETRAIN_ROOT', '/nvidia/store/matteos/pre-train'))
        checkout = Path(os.environ.get('NANOCHAT_DIR', root / 'nanochat'))
        ladders = Path(os.environ.get('LADDERS_DIR', checkout / 'ladders'))
        path = ladders / 'standalone' / validate_tag(tag or 'run')
    run = Path(path).expanduser().resolve()
    os.environ['NANOCHAT_RUN_DIR'] = str(run)
    paths = {'WANDB_DIR': run / 'wandb', 'WANDB_CACHE_DIR': run / 'wandb' / 'cache',
             'WANDB_CONFIG_DIR': run / 'wandb' / 'config', 'WANDB_DATA_DIR': run / 'wandb' / 'data',
             'TMPDIR': run / 'tmp'}
    for key, directory in paths.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    (run / 'checkpoints').mkdir(exist_ok=True)
    (run / 'nccl').mkdir(exist_ok=True)
    os.environ['NCCL_DEBUG_FILE'] = str(run / 'nccl' / '%h.%p.log')
    tempfile.tempdir = str(paths['TMPDIR'])
    return run

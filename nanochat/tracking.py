"""Opt-in, rank-zero W&B logging. All local files stay inside the run folder."""
import os
from pathlib import Path

from nanochat.common import DummyWandb, print0


def init_run(name, config, *, master_process=True):
    if name == 'dummy' or not master_process:
        return DummyWandb()
    from nanochat.run_paths import configure_run_dir
    configure_run_dir(tag=name)
    import wandb
    directory = Path(os.environ['WANDB_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'nanochat'),
                     entity=os.environ.get('WANDB_ENTITY'), name=name,
                     group=os.environ.get('WANDB_RUN_GROUP'), dir=str(directory),
                     config=config, save_code=False, settings=wandb.Settings(disable_git=True))
    run.define_metric('step')
    run.define_metric('*', step_metric='step')
    run.define_metric('val/bpb', step_metric='step', summary='min')
    run.define_metric('core_metric', step_metric='step', summary='max')
    print0(f'W&B run: {run.url or run.dir}')
    return run

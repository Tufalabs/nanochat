# Pre-training on thur

Dense Qwen3 training on GB300, with **two nodes / eight GPUs per run** by default.
Training is launched by an explicit `submit` action; the preset runner defaults
to a dry plan.

## Where to edit

| Path | Purpose |
|---|---|
| [configs/qwen3_scaling.sh](configs/qwen3_scaling.sh) | Editable ladder: model sizes, training budget, batch, evals, nodes and W&B |
| [run_scaling_ladder.sh](run_scaling_ladder.sh) | Plan or submit an experiment preset |
| [ladder.py](ladder.py) | Direct grid CLI; status, report, list and resume |
| [submit_run.py](submit_run.py) | Submit one named run inside a ladder folder |
| [env.sh](env.sh), [setup.sh](setup.sh) | Shared paths, environment and dependency installation |
| [jobs/](jobs/) | Slurm templates: training and one-off data preparation |
| [benchmarks/](benchmarks/) | Node-throughput probes and their report tool |
| [docs/](docs/) | Detailed usage, model/eval configuration and tracking |

The preset currently selects depths **8, 12, 16, 20, 24**, tokens:scaling-params ratio
**20**, context **8192**, fixed global batch **524,288 tokens/update**, and W&B enabled.
`NODES=2` and `MAX_PARALLEL=5` permit up to **five concurrent runs** (ten nodes).
Slurm's current `dev` QoS limits each user to eight nodes / 32 GPUs, so some runs
will queue; other allocations using that QoS also count against the limit.
The device-batch upper bound remains 16;
with the fixed global batch, the trainer uses **8 sequences/GPU** and one microbatch
per update. The preset logs synchronized optimizer timing every 10 updates
(`--optimizer-timing-every=10`). Node count can be set to 1, 2 or 4.

## Everyday commands

From the nanochat checkout:

```bash
source slurm/env.sh
bash slurm/run_scaling_ladder.sh plan       # default preset; submits nothing
# Only when ready to launch:
# bash slurm/run_scaling_ladder.sh submit

python slurm/ladder.py list
python slurm/ladder.py status LADDER
python slurm/ladder.py report LADDER
```

For a new experiment, copy `slurm/configs/qwen3_scaling.sh`, change `NAME` and the
settings, then preview it:

```bash
cp slurm/configs/qwen3_scaling.sh slurm/configs/my_experiment.sh
bash slurm/run_scaling_ladder.sh plan slurm/configs/my_experiment.sh
```

Use the same preset path with the explicit `submit` action to launch it. An existing
ladder name is never overwritten. `LADDER_NAME=my-name` overrides the preset name.
The lower-level `ladder.py plan NAME` defaults to the 24-run iso-FLOP grid, independently
of the shell preset, and also uses two nodes and a five-run concurrency cap by default.

For individual runs, use `python slurm/submit_run.py LADDER TAG [options] -- ARGS`.
It prepares output paths before submission, including Slurm startup/error logs.
Avoid submitting `jobs/train.sbatch` directly unless you also configure its output path.

## Outputs and resume

The default output root is **`nanochat/ladders/`**, inside the checkout and ignored by Git:

```text
ladders/<ladder>/
  ladder.json, plan.tsv, results.csv
  <descriptive-run-tag>/
    run.json, execution-<job-id>.json
    train.log, slurm-<job-id>.out, done
    checkpoints/, wandb/, nccl/, torchrun/, tmp/, eval/
```

Run-specific artifacts stay in the run folder. Ladder-level files contain the shared
plan, submission history and aggregate results. Each run is a separate Slurm job;
`afterany` dependency chains enforce the concurrency limit. All jobs use the **live
checkout**: source snapshots are not stored. Editing code can affect queued or resumed
runs. Source metadata records the path, commit and dirty flag, but cannot reconstruct
uncommitted changes.

`python slurm/ladder.py submit LADDER --resume` resubmits unfinished runs, skips queued
or completed runs, and preserves the saved plan and node count. Unfinished runs restart
from step zero. See [running and resuming](docs/running.md) for checkpoint continuation.

`LADDERS_DIR` overrides the output root. Shared data/tokenizer/eval assets live in
`../base/`, reusable compiler caches in `../cache/`, and setup logs in `../logs/`.

## Cluster setup

Always source `slurm/env.sh` before using the Python tools. Setup is idempotent:
`bash slurm/setup.sh` installs the shared venv; `sbatch slurm/jobs/data.sbatch` prepares
data, tokenizer and CORE assets. Data preparation stays
single-node because it is not distributed.

Use account `llm-team` and `--gres=gpu:b300:4` per node. `dev` allows up to eight nodes
per user and 12-hour jobs; account for other allocations when overriding concurrency.
The `short` (one hour) and `interactive` partitions require explicit `--nodes=1`.
`long` allows three-day jobs and up to two jobs per user. A three-node training run
is incompatible with the current optimizer sharding and vocabulary dimensions.

## Further reading

- [Running and resuming experiments](docs/running.md): grid options, multi-node launch, batches and outputs.
- [Model, data, validation and W&B](docs/model-and-evaluation.md): architecture, scaling and evaluation flags.

Benchmark tools also default to two nodes and only print a plan unless `--submit` is
specified. To preview the full comparison: `python slurm/benchmarks/nodes.py comparison
--nodes 1 2 4`. Afterwards, `python slurm/benchmarks/report.py --ladder comparison`
writes aggregates inside that ladder folder.

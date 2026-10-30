# Running and resuming experiments

[Back to the cluster guide](../README.md). Commands assume the nanochat checkout.

## Single runs and ladders

For an editable experiment definition, use [configs/qwen3_scaling.sh](../configs/qwen3_scaling.sh).
The preset defines the name, depth/compute grid, Slurm settings, W&B
switch, and shared model/training/evaluation arguments. The current user-edited preset
has five depths (8/12/16/20/24), tokens:scaling-params ratio 20, 8192-token context,
524,288 tokens/update, device batch upper bound 16, and W&B enabled. It is separate
from `ladder.py`'s default 24-run grid below. `NODES=2` selects nodes per run;
`MAX_PARALLEL=5` permits up to five concurrent runs, subject to Slurm quotas.
Five two-node runs need ten nodes; the current `dev` QoS limits each user to eight
nodes / 32 GPUs, so excess jobs queue. Other jobs on that QoS also count toward the limit.
`BATCH_POLICY=fixed` preserves `GLOBAL_BATCH_TOKENS=524288`; choosing `per_node`
multiplies that reference batch by `NODES` to keep more work on each GPU.

```bash
# From the nanochat checkout; the script sources env.sh itself.
bash slurm/run_scaling_ladder.sh plan     # preview only; this is also the default action
# When ready to launch the configured experiment:
# bash slurm/run_scaling_ladder.sh submit
```

Run the wrapper from a shell, not through `sbatch`. Its `submit` action calls the
ladder runner to create one Slurm job per run. Change `NAME` for a new experiment,
or supply `LADDER_NAME=my-experiment` in the environment. For an existing ladder,
use `python slurm/ladder.py status NAME`, `report NAME`, or `submit NAME --resume`
directly; resume retains the saved plan rather than applying edits to the wrapper.

```bash
# Examples below launch training and require approval for real runs:
# python slurm/submit_run.py trials qwen3-d12 -- --depth=12 --run=dummy

python slurm/ladder.py plan qwen3-baseline-isoflop
python slurm/ladder.py status NAME
python slurm/ladder.py report NAME
python slurm/ladder.py list
# After approval: python slurm/ladder.py submit NAME [same options]
```

Default grid: budgets `1e18 2.15e18 4.64e18 1e19` × depths `10 12 14 16 18 20`
(24 runs). `--mode ratio --ratio 12 --depths ...` instead uses tokens/scaling-params.
`--nodes N` selects nodes per run (default two); each node supplies four GPUs. `--partition dev`
is the default. `--max-parallel` limits concurrent **runs**, so peak nodes used is
`nodes * max_parallel`; its default is five runs. Slurm quotas may allow fewer runs
to start simultaneously. `--time MINUTES` overrides runtime estimates.
The planner reports total/scaling params, FLOPs, tokens, global batch, effective
per-GPU microbatch, accumulation steps and estimated cost.
**Runtime estimates are provisional**. The planner uses configurable MFU assumptions
and ideal compute scaling across nodes, with fixed compile/evaluation overhead.
Calibrate these assumptions for your model, hardware and evaluation protocol.

At 2048-token context, device batch upper bounds are 64 through d16, 32 through d32,
and 16 above. The ladder scales those bounds down by powers of two for longer contexts
to preserve tokens/GPU/microbatch, with a minimum batch of one. For example, d32 uses
batch 16 at 4096 tokens, 8 at 8192, 4 at 16384, and 2 at 32768. The trainer can further
halve the batch to divide the global token batch. `--device-batch-size` explicitly
overrides this rule; standalone `base_train` runs still need that flag set appropriately.
The rule preserves the reference token budget until batch one; it does not guarantee
arbitrarily long sequences fit. Check memory use for the intended configuration.

### Configuring multiple nodes

Edit `NODES` in `configs/qwen3_scaling.sh`, or pass `--nodes` directly to the planner:

```bash
python slurm/ladder.py plan my-four-node-ladder --nodes=4 --max-parallel=1 --partition=dev \
    -- --max-seq-len=8192 --total-batch-size=524288 --device-batch-size=16
# After approval, use the same options with submit.
# Standalone equivalent (launches training):
# python slurm/submit_run.py trials my-run --nodes=4 --partition=dev -- \
#     --max-seq-len=8192 --total-batch-size=524288 --device-batch-size=16
```

The launcher uses one Slurm task / torchrun agent per node and four workers per
agent, with the first allocated node hosting rendezvous. It uses static ranks and
fails the whole Slurm step if a node's agent fails. See the
[PyTorch 2.9 torchrun documentation](https://docs.pytorch.org/docs/2.9/elastic/run.html).
Inductor/Triton caches are separated by host under `cache/{inductor,triton}/nodes/`
for multi-node runs to avoid cross-host compiler-cache contention; FA4's persistent
cache remains shared. NCCL chooses the transport, with no forced network-interface
settings.

The model stays replicated on every GPU. This extends data parallel training and
optimizer-state sharding across nodes; it does not split model layers or context.
Use node counts **1, 2, or 4**. The existing optimizer requires the
vocabulary and large normalization dimensions to divide across ranks; with the
32,768-token vocabulary, three nodes / twelve ranks are not supported. The planner
rejects incompatible shapes or token batches before submission.

`total_batch_tokens = nodes * 4 * sequences_per_GPU * context * accumulation_steps`.
Node count alone never changes the global batch or base learning rates. At 8K with
524,288 tokens/update and device batch upper bound 16, 1/2/4 nodes use effective
microbatches 16/8/4 with one accumulation step. To retain microbatch 16, explicitly
set `--total-batch-size` to 524,288 / 1,048,576 / 2,097,152, respectively.
That changes the number of optimizer updates for a fixed token budget, and the
trainer's existing square-root batch rule also changes effective learning rates and
Muon weight decay. Treat it as a training-hyperparameter change, not just scheduling.

`submit` prepares the descriptive run folders before calling Slurm, so even scheduler
startup failures have a destination inside the correct run. Each run gets its own job;
`--max-parallel` is enforced with `afterany` dependency chains. A failed job releases
the next job in its chain. Submission history is saved after each accepted job, so
partial submission failures can be recovered. Use `submit_run.py LADDER TAG -- ARGS`
for an individual run with the same layout; its submission metadata stays in that run.

`submit NAME --resume` reuses the saved plan and live checkout, skips completed and
queued runs, and **restarts unfinished runs from step zero**. It is not checkpoint
continuation. New jobs wait for already-active jobs before opening new dependency
chains. Queue lookup failure blocks duplicate submission. Node count is preserved;
use a new ladder name to change it. Explicit `--resume-from-step=N` with the trainer
requires the original run directory, model shape, world size and training settings.
Peak memory in logs and W&B summaries is the maximum allocated memory across all ranks.

`report` includes completed runs only and writes notebook-compatible `results.csv`
plus MFU, throughput, memory and wall time. Throughput/MFU are medians over the second
half of the run. Training time excludes the first 11 steps and evals; wall time includes
them. MFU divides estimated model forward/backward FLOPs by full update time and
the active GPU count times the 2.5 PFLOP/s dense-BF16 GB300 peak. The time includes
the optimizer, communication and data loading, but excludes evaluation, logging
and checkpointing. Compilation can inflate early update times; exclude warmup
when comparing steady-state MFU. Optimizer FLOPs are absent from the numerator.
Fits exclude concave or outside-range minima. A useful scaling-law result requires
adequate compute and model-size coverage.


To use another preset, copy `slurm/configs/qwen3_scaling.sh`, change its `NAME` and
settings, and pass its path explicitly:

```bash
bash slurm/run_scaling_ladder.sh plan slurm/configs/my_experiment.sh
# Submit only when requested:
# bash slurm/run_scaling_ladder.sh submit slurm/configs/my_experiment.sh
```

Use `--nodes=1` explicitly for the single-node `short` and `interactive` partitions.
Data preparation (`jobs/data.sbatch`) also uses one node because it is not distributed.

For a single-GPU probe, use `submit_run.py ... --nodes=1 --workers-per-node=1 -- ARGS`.
Slurm still reserves the whole four-GPU node; torchrun only starts one worker.
The trainer reports the actual world size, so MFU uses one GPU in its denominator.
Ladder runs continue to use four workers per node by default.

# Model, data, validation and W&B

[Back to the cluster guide](../README.md). Commands assume the nanochat checkout.

## Model architecture and scaling

New `scripts.base_train` runs use **dense Qwen3** implemented in `nanochat/qwen3.py`:
learned RMSNorm, per-head Q/K RMSNorm before RoPE, grouped-query attention, biasless
projections, SwiGLU MLP, ordinary residuals, full causal attention, final RMSNorm,
and tied input/output embeddings. No nanochat value embeddings, embedding smear,
residual mixing, backout, or logit softcap. FA4 supplies training attention;
SDPA supplies cached inference. `NANOCHAT_ATTN=sdpa` forces SDPA for comparison.

Architecture references: [official Qwen3-1.7B config](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)
and [Transformers Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py).
This trains from scratch with a **32,768-token BPE and Muon+AdamW recipe**.
The trainer defaults to 2048-token context; the ladder preset selects 8192.
It does not load Qwen weights/tokenizer or reproduce their
training data, optimizer, context length, or post-training.

Trainer and planner use one shared shape factory, `config_for_depth`, to scale
both width and depth:

- Layers = `d` (`--depth`). Residual width = `ceil(d * aspect_ratio / head_dim) * head_dim`;
  defaults `aspect_ratio=64`, `head_dim=128`.
- Query heads = `ceil(width / (head_dim * kv_group_size)) * kv_group_size`;
  KV heads = query heads / `kv_group_size` (default 2).
- SwiGLU intermediate width = `ceil(mlp_ratio * width / 128) * 128` (default ratio 3).
- RoPE base = 1,000,000; RMSNorm epsilon = 1e-6; initialization std = 0.02.
- Embeddings are tied by default; `--untie-embeddings` changes this.

For depths with an odd `width / 128`, query heads round up to a complete GQA pair;
attention projection width can therefore exceed residual width. Qwen3 supports this
independence (also used in the [official 0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)).
For example d10 has width 640, 6 query heads and 3 KV heads of size 128.
The shape rule is explicit and consistent across every budget.

| Depth | Width | Query / KV heads | Total parameters |
|---:|---:|---:|---:|
| 8 | 512 | 4 / 2 | 41.95M |
| 10 | 640 | 6 / 3 | 72.60M |
| 12 | 768 | 6 / 3 | 110.12M |
| 16 | 1024 | 8 / 4 | 234.92M |
| 20 | 1280 | 10 / 5 | 435.22M |
| 24 | 1536 | 12 / 6 | 729.89M |
| 26 | 1664 | 14 / 7 | 935.13M |
| 28 | 1792 | 14 / 7 | 1,137.81M |
| 32 | 2048 | 16 / 8 | 1,677.86M |

Thus d32 is under the team's 1.7B **total** parameter target. CSV "scaling params"
are block matrices plus LM head; tied weights count once. Learned norms account for
the small difference from total params.

Muon updates block matrices; AdamW updates the tied embedding/head and learned norms.
The tied matrix uses `--unembedding-lr`; `--embedding-lr` only applies when untied.
Norm LR = `0.01 * --scalar-lr`. The Qwen3 defaults are `--matrix-lr=0.005`,
`--unembedding-lr=0.002`, `--scalar-lr=0.1` (norm LR 0.001), with 40 warmup steps.
Batch/width LR scaling and weight-decay scaling also apply. These rates are starting
values to tune for the experiment. BF16 is the default; FP8 is optional.

New checkpoint metadata records `architecture=qwen3`. The loader supports both Qwen3
and old nanochat checkpoints. Training rejects incompatible checkpoint configs.
Use a **new ladder name** for each experiment.

## Data and held-out validation

`slurm/jobs/data.sbatch` prepares 1000 ClimbMix training shards plus validation shard
`06542` by default, trains the tokenizer on ~2B characters, and downloads the CORE
bundle. `NUM_SHARDS=2000 sbatch slurm/jobs/data.sbatch` adds shards (max 6542);
existing tokenizer/downloads are reused. No offline tokenization is required.
The loader tokenizes parquet text on the fly with BOS-aligned best-fit packing.

`nanochat/dataset.py` uses `base/base_data_climbmix/`: **the last sorted parquet file
is validation; all earlier files are training**. Here the held-out file is `06542`.
Each validation call starts a fresh deterministic loader from that same split;
it is not a randomly sampled training minibatch. Preserve the held-out file when
adding training shards. There is currently no CLI for custom per-domain validation
sets or data mixtures. Adding those requires an explicit loader/split change; keep
the same held-out corpus when comparing mixtures.

The planner assumes approximately 44M packed training tokens/shard. This depends on
tokenizer, sequence length and packing; changes to those require recalibration.

## Default evaluations and configuration

| Evaluation | Standalone `base_train` default | Ladder default |
|---|---|---|
| Held-out bits per byte (bpb), lower is better | Step 0, every 250 steps, final; 41,943,040 tokens/pass | Same schedule; 52,428,800 tokens/pass |
| CORE, higher is better | Every 2000 steps and final; max 500 examples/task | Final; all examples/task (`every=999999`, `max=-1`) |
| Generated samples | Every 2000 steps after step 0 and final | Disabled |
| Checkpoint | Final only | Final only |

Validation bpb = total negative log likelihood / (`ln(2)` * target bytes); special
and ignored tokens are excluded. This is the metric used for iso-FLOP fits.
Validation runs on all participating GPUs. The token budget rounds **down** to a multiple of
`effective_device_batch_size * sequence_len * world_size`; the trainer prints the
actual budget and rejects a budget smaller than one distributed microbatch.
Validation is performed in BF16 even when optional FP8 training is enabled.

**CORE** is the equal-weight mean of 22 task accuracies, each normalized by its random
baseline: `(accuracy - baseline) / (1 - baseline)`. Negative values are possible.
It covers HellaSwag (0- and 10-shot), Jeopardy, Wikidata QA, ARC easy/challenge, COPA,
CommonsenseQA, PIQA, OpenBookQA, LAMBADA, Winograd/WinoGrande, Dyck languages, LSAT AR,
CS algorithms, operators, repeat-copy logic, SQuAD, CoQA, BoolQ and language identification.
The fixed task/few-shot definitions live in `base/eval_bundle/core.yaml`. Capped runs
use deterministic shuffling (seed 1337). The cap now limits scored examples while
retaining the full demonstration pool, so even caps below the 10-shot count work.
Each scored example gets the same prompt as in a full evaluation. This fixes upstream's
small-cap crash and changes capped few-shot scores relative to the previous truncated
demonstration pool; compare runs using the same code/protocol. A capped CORE score is
not directly equivalent to the full-bundle score. This suite does not include dedicated
GSM8K/MATH or code evals.

Training and bpb use 2048-token sequences by default. CORE uses variable-length few-shot
prompts, which can exceed that training length: the implementation's rotary cache permits
up to 10 times the configured sequence length. This is an implementation allowance,
not evidence of trained long-context capability. It preserves nanochat's eval behavior.

Relevant **training** flags:

```text
--eval-every=250                # bpb frequency; -1 disables including final
--eval-tokens=52428800          # total tokens across all GPUs per pass
--core-metric-every=999999      # effectively final-only at usual run lengths
--core-metric-max-per-task=-1   # full bundle; positive N caps examples/task
--sample-every=-1               # disable text samples, including final
--save-every=-1                 # final checkpoint only; positive N also saves periodically
```

A positive eval/core/sample interval always triggers a final pass, even if the final
step is not divisible by that interval. CORE/sampling skip step 0; bpb includes step 0.
For cheaper exploratory validation, preview a ladder with:

```bash
python slurm/ladder.py plan qwen3-exploration -- \
  --eval-every=500 --eval-tokens=10485760 --core-metric-max-per-task=500
```

Extra arguments after `--` apply to each run and override ladder eval defaults. Keep
protocols identical within a comparison and use full CORE for final comparable results.
To evaluate an existing checkpoint separately on an allocated node:

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
  --run-dir=/nvidia/store/matteos/pre-train/nanochat/ladders/LADDER/TAG \
  --eval=core,bpb --max-per-task=-1 --split-tokens=20971520
```

`base_eval --eval` accepts `core,bpb,sample`; it defaults to all three. Its bpb mode
measures **both train and val** splits (20,971,520 tokens per split by default), and
its CORE cap defaults to full. Use `--step=N` for a specific checkpoint; otherwise it
loads the latest step. These standalone eval flags are distinct from training flags.
It loads `checkpoints/` and writes CSVs in `eval/` inside the selected run folder;
it does not initialize a W&B run. Redirect its console output into that same folder
if you want a persistent evaluation log.

## Weights & Biases

Logging is **opt-in**: standalone `--run=NAME`, or ladder `--wandb`. `--run=dummy`
(the default) makes no W&B connection. Only rank zero creates/logs a run.

Supply `WANDB_API_KEY` and `WANDB_ENTITY` in the submitting environment via your
normal secret mechanism; do not put the key in source or plans. `sbatch` exports
the submission environment by default.

```bash
source slurm/env.sh
export WANDB_PROJECT=nanochat       # optional; this is the default project
# WANDB_ENTITY should be your team/user slug.
# WANDB_API_KEY must be available in the submitting shell.

# Dry-run planning only:
python slurm/ladder.py plan qwen3-baseline --wandb
# After real-ladder approval, use the same options with `submit`.

# For a separately authorized standalone run, add:
# --run=my-qwen3-run --log-every=10
```

Ladder runs are grouped by ladder name (`WANDB_RUN_GROUP` overrides it). Configuration
includes CLI arguments, actual model shape and parameter counts, world size, effective
batch sizes, effective iteration count, token and FLOP budgets. Logged metrics include
training loss/LR multiplier/tokens per second/MFU, tokens/FLOPs, bpb, CORE and per-task
centered results. W&B's x-axis is completed optimizer updates (`step`). Training logs
every 10 updates by default (`--log-every`); evals log whenever they run. Final summaries
include steps, tokens, training time, peak allocated memory, best bpb and best CORE.

SDK artifacts/cache/config remain under the run folder’s `wandb/` directory; code upload
and Git metadata scanning are disabled. `WANDB_MODE=offline` records locally and
requires later `wandb sync`; unset it or set `online` for live logging. `WANDB_MODE=disabled`
prevents live logging even if a run is named. See [W&B environment variables](https://docs.wandb.ai/models/track/environment-variables).


## Optimizer timing

`--optimizer-timing-every=N` adds `train/optimizer_dt` (seconds) and
`train/optimizer_time_pct` (`100 * optimizer_dt / train/dt`) every N completed
updates. The trainer defaults to 0 (disabled); the standard ladder preset enables
it every 10 updates.

On sampled updates all ranks synchronize CUDA before and after the optimizer.
The measured wall-clock interval includes Muon/AdamW updates and the gradient
reductions/parameter gathers inside the optimizer. FP16 runs also include gradient
unscaling and scaler operations. Rank zero supplies the reported time, matching
`train/dt`; it is not an average across ranks. Timing adds synchronization and may
reduce overlap, so compare timed and untimed updates before interpreting the
percentage as ordinary-run overhead. Compilation/warmup should be excluded.

The FLOP numerator used for MFU still counts model forward/backward only. This new
percentage is the fraction of update time spent inside the timed optimizer section,
not a count of optimizer FLOPs or a fraction of GPU peak throughput. Timed updates
are logged even when they do not coincide with `--log-every`.

"""
Scaling-ladder runner for nanochat on thur (configurable nodes, 4x GB300 per node).

A ladder is a grid of base_train runs over model size (depth) x compute, with one Slurm
job per run (one or more nodes per job, bounded parallelism). Two modes:
  isoflop : for each FLOPs budget C, train every depth with exactly C FLOPs  (Chinchilla approach 2:
            fit val_bpb vs N per budget -> compute-optimal N*(C), D*(C))
  ratio   : train every depth at a fixed tokens:params ratio (a compute-optimal "miniseries")

Everything for a ladder lives in $LADDERS_DIR/<name>/:
  plan.tsv     model/compute grid and exact training arguments
  ladder.json  configuration and submission history
  <tag>/       one descriptive folder per run: run.json, train.log, slurm-*.out,
               done, checkpoints/, wandb/, nccl/, torchrun/, tmp/, eval/
  results.csv  aggregate ladder report

All jobs run the live NANOCHAT_DIR checkout; no source snapshots are copied.

Usage (on the login node, after `source slurm/env.sh`, from the nanochat dir):
  python slurm/ladder.py plan   NAME [grid options] [-- extra base_train args]   # dry run: print the grid
  python slurm/ladder.py submit NAME [grid options] [-- extra base_train args]   # submit jobs
  python slurm/ladder.py submit NAME --resume                                    # resubmit unfinished runs
  python slurm/ladder.py status NAME                                             # per-run state/progress
  python slurm/ladder.py report NAME                                             # results.csv + fits
  python slurm/ladder.py list
"""
import argparse
import csv
import datetime
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys

PRETRAIN_ROOT = os.environ.get("PRETRAIN_ROOT", "/nvidia/store/matteos/pre-train")
NANOCHAT_DIR = os.environ.get("NANOCHAT_DIR", os.path.join(PRETRAIN_ROOT, "nanochat"))
LADDERS_DIR = os.environ.get("LADDERS_DIR", os.path.join(NANOCHAT_DIR, "ladders"))
BASE_DIR = os.environ.get("NANOCHAT_BASE_DIR", os.path.join(PRETRAIN_ROOT, "base"))
GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", 4))
DEFAULT_NODES = 2
DEFAULT_MAX_PARALLEL = 5  # Slurm account/QoS limits may reduce actual concurrency.

# Provisional planning assumptions; recalibrate for the model and evaluation protocol.
PEAK_FLOPS_PER_GPU = 2.5e15     # GB300 dense bf16
ASSUMED_MFU = 0.35
SMALL_MODEL_MFU = {8: 0.12, 10: 0.25, 12: 0.32}
OVERHEAD_MIN = 10               # compile + full CORE + save at d<=12; scale with depth above d12
VALIDATION_MIN = 0.25           # per 52M-token validation pass; include periodic evals in long ladders
PARTITION_MAX_MIN = {"short": 60, "interactive": 8 * 60, "dev": 12 * 60, "long": 72 * 60}
PARTITION_MAX_NODES = {"short": 1, "interactive": 1, "dev": 8, "long": 16}
# Approximate packed tokens per shard; depends on tokenizer, context length and packing.
TRAIN_TOKENS_PER_SHARD = 44e6

DEFAULT_FLOPS = [1e18, 2.15e18, 4.64e18, 1e19]   # same grid as upstream runs/scaling_laws.sh
DEFAULT_DEPTHS = [10, 12, 14, 16, 18, 20]


def default_device_batch_size(depth, sequence_len=2048):
    # Preserve the reference per-GPU token budget at 2K context.
    # This is an upper bound: base_train can further halve it for global-batch divisibility.
    if sequence_len <= 0:
        raise ValueError("sequence length must be positive")
    base = 64 if depth <= 16 else 32 if depth <= 32 else 16
    batch = base
    while batch > 1 and batch * sequence_len > base * 2048:
        batch //= 2
    return batch


def fmt_flops(c):
    return f"{c:g}".replace("+", "")   # 1e18, 2.15e18, 1e19


# -----------------------------------------------------------------------------
# Size/compute estimates: mirror what scripts/base_train.py will compute

def estimate_runs(depths, mode, flops_list, ratio, extra_args, nodes=DEFAULT_NODES):
    if nodes < 1:
        raise ValueError("nodes must be positive")
    world_size = nodes * GPUS_PER_NODE
    managed = {"--depth", "--target-flops", "--target-param-data-ratio", "--num-iterations",
               "--model-tag", "--run", "--run-dir", "--resume-from-step"}
    if any(arg.split("=", 1)[0] in managed for arg in extra_args):
        raise ValueError("depth, horizon, model-tag, run and resume are ladder-managed; use grid options instead")
    if not depths or any(d <= 0 for d in depths) or len(set(depths)) != len(depths):
        raise ValueError("depths must be distinct positive integers")
    if ratio <= 0 or not math.isfinite(ratio):
        raise ValueError("ratio must be positive and finite")
    if not flops_list or any(c <= 0 or not math.isfinite(c) for c in flops_list) or len(set(flops_list)) != len(flops_list):
        raise ValueError("FLOP budgets must be distinct positive finite numbers")
    sys.path.insert(0, NANOCHAT_DIR)
    os.environ.setdefault("NANOCHAT_BASE_DIR", BASE_DIR)
    import torch
    from nanochat.qwen3 import Qwen3, config_for_depth

    # pick up model-shape overrides from the extra args, if any
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--window-pattern", choices=["L"], default="L")
    p.add_argument("--kv-group-size", type=int, default=2)
    p.add_argument("--mlp-ratio", type=float, default=3.0)
    p.add_argument("--rope-theta", type=float, default=1e6)
    p.add_argument("--untie-embeddings", action="store_true")
    p.add_argument("--total-batch-size", type=int, default=-1)
    p.add_argument("--device-batch-size", type=int)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-tokens", type=int, default=100 * 524288)
    shape, _ = p.parse_known_args(extra_args)

    vocab_size = 32768
    try:
        from nanochat.tokenizer import get_tokenizer
        vocab_size = get_tokenizer().get_vocab_size()
    except Exception:
        pass

    def meta(depth):
        cfg = config_for_depth(depth, sequence_len=shape.max_seq_len, vocab_size=vocab_size,
                        aspect_ratio=shape.aspect_ratio, head_dim=shape.head_dim,
                        kv_group_size=shape.kv_group_size, mlp_ratio=shape.mlp_ratio,
                        rope_theta=shape.rope_theta, tie_word_embeddings=not shape.untie_embeddings,
                        window_pattern=shape.window_pattern)
        with torch.device("meta"):
            m = Qwen3(cfg)
        # The current AdamW implementation shards large tensors along dimension zero.
        # Fail before allocating nodes rather than on the first optimizer update.
        if cfg.vocab_size % world_size or (cfg.n_embd >= 1024 and cfg.n_embd % world_size):
            raise ValueError(f"d{depth}: optimizer sharding requires vocabulary size and large RMSNorm "
                             f"widths divisible by {world_size} ranks; use a compatible node count (e.g. 1, 2, 4)")
        counts = m.num_scaling_params()
        return dict(model_dim=cfg.n_embd, n_scaling=counts["transformer_matrices"] + counts["lm_head"],
                    n_total=counts["total"], flops_per_token=m.estimate_flops())

    ref = meta(12)
    runs = []
    budgets = flops_list if mode == "isoflop" else [None]
    for c in budgets:
        for d in depths:
            m = meta(d)
            # optimal batch size (Power Lines, B ~ D^0.383, ref 2^19 at d12); depends only on N/N_d12 here
            if shape.total_batch_size > 0:
                batch = shape.total_batch_size
            else:
                batch = 2 ** round(math.log2(2**19 * (m["n_scaling"] / ref["n_scaling"]) ** 0.383))
            device_batch = shape.device_batch_size if shape.device_batch_size is not None else default_device_batch_size(d, shape.max_seq_len)
            if device_batch < 1 or batch % (shape.max_seq_len * world_size):
                raise ValueError(f"d{d}: global batch {batch} must divide into {world_size} ranks at "
                                 f"sequence length {shape.max_seq_len}, with a positive device batch")
            while device_batch > 1 and batch % (device_batch * shape.max_seq_len * world_size):
                device_batch //= 2
            micro_tokens = device_batch * shape.max_seq_len * world_size
            if shape.eval_every > 0 and shape.eval_tokens < micro_tokens:
                raise ValueError(f"d{d}: eval-tokens must be at least one distributed microbatch ({micro_tokens})")
            if mode == "isoflop":
                iters = round(c / (m["flops_per_token"] * batch))
                tag = f"c{fmt_flops(c)}_d{d}"
                horizon = [f"--target-flops={c:g}", "--target-param-data-ratio=-1"]
            else:
                iters = int(ratio * m["n_scaling"]) // batch
                tag = f"r{ratio:g}_d{d}"
                horizon = [f"--target-param-data-ratio={ratio:g}"]
            if iters < 1:
                raise ValueError(f"{tag}: budget is too small for one training step")
            tokens = iters * batch
            flops = tokens * m["flops_per_token"]
            mfu = SMALL_MODEL_MFU.get(d, ASSUMED_MFU)
            evals = 2 + (iters - 1) // shape.eval_every if shape.eval_every > 0 else 0
            eval_min = evals * VALIDATION_MIN * shape.eval_tokens / (100 * 524288)
            est_min = (flops / (world_size * PEAK_FLOPS_PER_GPU * mfu) / 60
                       + OVERHEAD_MIN * max(1, d / 12) + eval_min)
            runs.append(dict(tag=tag, depth=d, flops_budget=c if c else flops, ratio=tokens / m["n_scaling"],
                             model_dim=m["model_dim"], n_scaling=m["n_scaling"], n_total=m["n_total"],
                             batch=batch, iters=iters, tokens=tokens, flops=flops, est_min=est_min,
                             horizon=horizon, nodes=nodes, world_size=world_size,
                             device_batch=device_batch, grad_accum_steps=batch // micro_tokens))
    return runs


def build_args(name, run, extra_args, wandb):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--max-seq-len", type=int, default=2048)
    shape, _ = p.parse_known_args(extra_args)
    args = [f"--depth={run['depth']}", *run["horizon"],
            f"--model-tag={name}_{run['tag']}",
            f"--run={name}_{run['tag']}" if wandb else "--run=dummy",
            f"--device-batch-size={default_device_batch_size(run['depth'], shape.max_seq_len)}",
            # same eval protocol as upstream runs/scaling_laws.sh: ~52M val tokens, full CORE at the end only
            f"--eval-tokens={100 * 524288}", "--core-metric-every=999999", "--core-metric-max-per-task=-1",
            "--sample-every=-1", "--save-every=-1"]
    # extra args come last so they override the defaults above (argparse: last one wins)
    return args + list(extra_args)


def count_train_shards():
    d = os.path.join(BASE_DIR, "base_data_climbmix")
    if not os.path.isdir(d):
        return 0
    return max(0, len([f for f in os.listdir(d) if f.endswith(".parquet")]) - 1)   # last one is val


def print_plan(runs, max_parallel=None, nodes=DEFAULT_NODES):
    if max_parallel is None:
        max_parallel = DEFAULT_MAX_PARALLEL
    print("Architecture: dense Qwen3, width scales with depth. Runtime estimates remain provisional: legacy nanochat calibration.")
    print(f"Each run: {nodes} node(s), {nodes * GPUS_PER_NODE} GPUs. Multi-node time assumes ideal scaling; communication overhead is not calibrated.")
    print(f"{'tag':<16}{'depth':>6}{'dim':>6}{'N_scaling':>12}{'N_total':>11}{'tokens':>10}{'tok/N':>7}{'FLOPs':>11}"
          f"{'batch':>11}{'micro':>7}{'accum':>7}{'iters':>8}{'est':>8}")
    for r in runs:
        print(f"{r['tag']:<16}{r['depth']:>6}{r['model_dim']:>6}{r['n_scaling']/1e6:>11.1f}M{r['n_total']/1e6:>10.1f}M{r['tokens']/1e9:>9.2f}B"
              f"{r['ratio']:>7.1f}{r['flops']:>11.2e}{r['batch']:>11,}{r['device_batch']:>7}{r['grad_accum_steps']:>7}{r['iters']:>8,}{r['est_min']:>7.0f}m")
    total_gpu_h = sum(r["est_min"] for r in runs) / 60 * GPUS_PER_NODE * nodes
    print(f"{len(runs)} runs, ~{total_gpu_h:.0f} GPU-hours estimated (depth-dependent MFU assumptions, "
          f"{ASSUMED_MFU:.0%} fallback; {OVERHEAD_MIN}m compile/CORE/save scaled above d12; "
          f"{VALIDATION_MIN}m per 52M-token validation)")
    slots = [0.0] * min(max_parallel, len(runs))
    for r in sorted(runs, key=lambda r: -r["est_min"]):
        slots[slots.index(min(slots))] += r["est_min"]
    print(f"Run times ~{min(r['est_min'] for r in runs):.0f}-{max(r['est_min'] for r in runs):.0f}m; "
          f"~{max(slots)/60:.1f}h elapsed with {len(slots)} concurrent runs / {len(slots) * nodes} free nodes (queue time excluded)")
    need = max(r["tokens"] for r in runs) / TRAIN_TOKENS_PER_SHARD
    have = count_train_shards()
    print(f"Largest run needs ~{math.ceil(need)} train shards at {TRAIN_TOKENS_PER_SHARD/1e6:g}M packed tokens/shard; "
          f"{have} downloaded")
    if need > have:
        print(f"NOTE: the largest run needs ~{need:.0f} train shards, {have} are downloaded -> it will repeat data "
              f"(multi-epoch). Download more with: NUM_SHARDS={int(need * 1.2)} sbatch slurm/jobs/data.sbatch")


# -----------------------------------------------------------------------------
# Commands

def ladder_dir(name):
    return os.path.join(LADDERS_DIR, name)


def read_plan(ldir):
    with open(os.path.join(ldir, "plan.tsv")) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def squeue_states():
    """Map 'jobid_task' -> (state, elapsed, node/reason) for the current user's jobs."""
    try:
        out = subprocess.run(["squeue", "-h", "-r", "-u", os.environ.get("USER", ""), "-o", "%i|%T|%M|%R"],
                             capture_output=True, text=True, check=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as e:
        detail = (getattr(e, "stderr", None) or str(e)).strip()
        raise RuntimeError(f"cannot query Slurm: {detail}") from e
    states = {}
    for line in out.strip().splitlines():
        jid, st, el, where = line.split("|", 3)
        states[jid] = (st, el, where)
    return states


def cmd_plan(a):
    runs = estimate_runs(a.depths, a.mode, a.flops, a.ratio, a.extra, a.nodes)
    print_plan(runs, a.max_parallel, a.nodes)
    if runs:
        print("\nexample args:", " ".join(build_args(a.name, runs[0], a.extra, a.wandb)))


def run_dir(ldir, tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(f"invalid run tag: {tag!r}")
    return os.path.join(ldir, tag)


def prepare_run(ldir, tag, args, **details):
    directory = run_dir(ldir, tag)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "run.json")
    if not os.path.exists(path):
        git = lambda *c: subprocess.run(["git", "-C", NANOCHAT_DIR, *c], capture_output=True, text=True).stdout.strip()
        metadata = dict(tag=tag, args=list(args), source_checkout=NANOCHAT_DIR,
                        source_commit=git("rev-parse", "HEAD"), source_dirty=bool(git("status", "--porcelain")),
                        created=datetime.datetime.now().isoformat(timespec="seconds"), **details)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
    return directory


def submit_run(directory, tag, args, *, nodes, partition, time_min, dependencies=(), workers_per_node=GPUS_PER_NODE):
    if nodes < 1 or nodes > PARTITION_MAX_NODES.get(partition, math.inf):
        raise ValueError(f"invalid node count {nodes} for partition {partition}")
    if not 1 <= workers_per_node <= GPUS_PER_NODE:
        raise ValueError(f"workers per node must be between 1 and {GPUS_PER_NODE}")
    cmd = ["sbatch", "--parsable", f"--nodes={nodes}", "--ntasks-per-node=1", "--gres=gpu:b300:4",
           f"--job-name={tag}", f"--account={os.environ.get('SLURM_ACCOUNT_NAME', 'llm-team')}",
           f"--partition={partition}", f"--time={time_min}", f"--chdir={NANOCHAT_DIR}",
           f"--output={directory}/slurm-%j.out",
           f"--export=ALL,NANOCHAT_RUN_DIR={directory},TAG={tag},NANOCHAT_NPROC_PER_NODE={workers_per_node}"]
    if dependencies:
        cmd.append("--dependency=afterany:" + ":".join(dependencies))
    cmd += [os.path.join(NANOCHAT_DIR, "slurm", "jobs", "train.sbatch"), *args]
    print(shlex.join(cmd))
    job_id = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]
    return job_id


def submission_job_id(submission, task):
    return f"{submission['job_id']}_{task}" if submission.get("array", True) else submission['job_id']


def cmd_submit(a):
    ldir = ladder_dir(a.name)
    meta_path = os.path.join(ldir, "ladder.json")
    if a.resume:
        if not os.path.exists(meta_path):
            sys.exit(f"--resume: no ladder at {ldir}")
        meta = json.load(open(meta_path))
        if meta.get("layout_version") != 2:
            raise ValueError("unsupported legacy output layout; use a new ladder name")
        if a.nodes is not None and a.nodes != meta.get("nodes", 1):
            sys.exit("--resume preserves the original node count; use a new ladder name to change nodes")
    else:
        if os.path.exists(ldir):
            sys.exit(f"{ldir} already exists. Use `submit {a.name} --resume` or a new name.")
        runs = estimate_runs(a.depths, a.mode, a.flops, a.ratio, a.extra, a.nodes)
        runs.sort(key=lambda r: -r["est_min"])
        print_plan(runs, a.max_parallel, a.nodes)
        os.makedirs(ldir)
        cols = ["tag", "depth", "model_dim", "n_scaling", "n_total", "flops_budget", "ratio", "batch", "iters",
                "tokens", "flops", "est_min", "nodes", "world_size", "device_batch", "grad_accum_steps", "args"]
        with open(os.path.join(ldir, "plan.tsv"), "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in runs:
                args = build_args(a.name, r, a.extra, a.wandb)
                r["args"] = shlex.join(args)
                prepare_run(ldir, r['tag'], args, nodes=a.nodes, depth=r['depth'])
                f.write("\t".join(str(r[c]) if not isinstance(r[c], float) else f"{r[c]:.6g}" for c in cols) + "\n")
        time_min = a.time or min(PARTITION_MAX_MIN.get(a.partition, 720),
                                 int(math.ceil(max(r["est_min"] for r in runs) * 1.5 / 10) * 10))
        meta = dict(name=a.name, mode=a.mode, architecture="qwen3", layout_version=2,
                    created=datetime.datetime.now().isoformat(timespec="seconds"), extra_args=a.extra,
                    wandb=a.wandb, partition=a.partition, time_min=time_min, max_parallel=a.max_parallel,
                    nodes=a.nodes, gpus_per_node=GPUS_PER_NODE, submissions=[])
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    plan = read_plan(ldir)
    todo = [i for i, r in enumerate(plan) if not os.path.exists(os.path.join(run_dir(ldir, r['tag']), 'done'))]
    active = {}
    if a.resume and todo:
        queue = squeue_states()  # Never duplicate jobs if queue lookup fails.
        for submission in meta['submissions']:
            for i in submission['tasks']:
                job = submission_job_id(submission, i)
                if job in queue:
                    active[i] = job
        todo = [i for i in todo if i not in active]
    if not todo:
        print("nothing to submit: all runs are done or still in the queue")
        return
    partition = a.partition if a.resume and a.partition_set else meta['partition']
    nodes = meta.get('nodes', 1)
    if nodes > PARTITION_MAX_NODES.get(partition, math.inf):
        raise ValueError(f"partition {partition} does not allow {nodes} nodes per run")
    time_min = a.time or meta['time_min']
    parallel = a.max_parallel or meta['max_parallel']
    # One job per descriptive folder. Each lane is a dependency chain. On resume,
    # wait for existing jobs before opening new lanes, preserving the concurrency cap.
    tails = [list(dict.fromkeys(active.values())) for _ in range(min(parallel, len(todo)))]
    loads = [0.0] * len(tails)
    for i in todo:
        row = plan[i]
        lane = loads.index(min(loads))
        directory = prepare_run(ldir, row['tag'], shlex.split(row['args']), nodes=nodes)
        job_id = submit_run(directory, row['tag'], shlex.split(row['args']), nodes=nodes,
                            partition=partition, time_min=time_min, dependencies=tails[lane])
        submission = dict(job_id=job_id, tasks=[i], array=False, partition=partition,
                          time_min=time_min, nodes=nodes, at=datetime.datetime.now().isoformat(timespec="seconds"))
        meta['submissions'].append(submission)
        # Save after each accepted job, so a later sbatch failure remains resumable.
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        tails[lane] = [job_id]
        loads[lane] += float(row.get('est_min', 1))
    print(f"submitted ladder '{a.name}': {len(todo)} jobs, at most {parallel} concurrent runs -> {ldir}")
    print(f"  status: python slurm/ladder.py status {a.name}")


STEP_RE = re.compile(r"^step (\d+)/(\d+) \(([\d.]+)%\).*?bf16_mfu: ([\d.]+)")
FAILURE_RE = re.compile(r"Traceback \(most recent call last\)|ChildFailedError|OutOfMemoryError|DUE TO TIME LIMIT|CANCELLED AT")


def cmd_status(a):
    ldir = ladder_dir(a.name)
    meta = json.load(open(os.path.join(ldir, "ladder.json")))
    plan = read_plan(ldir)
    try:
        q = squeue_states()
    except RuntimeError as e:
        print(f"WARNING: {e}; showing log evidence only", file=sys.stderr)
        q = None
    task_job = {}   # array index -> latest job id that included it
    for s in meta["submissions"]:
        for t in s["tasks"]:
            task_job[t] = submission_job_id(s, t)
    print(f"{'#':>3} {'tag':<16} {'state':<12} {'progress':<54} node/reason")
    for i, r in enumerate(plan):
        log = os.path.join(run_dir(ldir, r["tag"]), "train.log")
        done = os.path.exists(os.path.join(run_dir(ldir, r["tag"]), "done"))
        jid = task_job.get(i)
        state, where, progress = "-", "", ""
        if done:
            state = "DONE"
            m = parse_log(log)
            progress = f"val_bpb {m.get('val_bpb', '?')} CORE {m.get('core_score', '?')}"
        elif q is not None and jid in q:
            state, elapsed, where = q[jid]
            progress = f"elapsed {elapsed}"
        elif i in task_job:
            state = "UNKNOWN" if q is None else "INCOMPLETE"
            where = f"see {run_dir(ldir, r['tag'])}/slurm-{task_job[i]}.out"
        if os.path.exists(log) and not done:
            tail = open(log, errors="replace").read().splitlines()[-200:]
            if (q is None or jid not in q) and any(FAILURE_RE.search(line) for line in tail):
                state = "FAILED"
            for line in reversed(tail):
                m = STEP_RE.match(line)
                if m:
                    eta = re.search(r"eta: [\d.]+m", line)
                    progress = f"step {m.group(1)}/{m.group(2)} ({m.group(3)}%) mfu {m.group(4)} {eta.group() if eta else ''}"
                    break
        print(f"{i:>3} {r['tag']:<16} {state:<12} {progress:<54} {where}")


def parse_log(path):
    """Pull the numbers we need for scaling-law analysis out of a base_train log."""
    m = {}
    if not os.path.exists(path):
        return m
    mfus, toks = [], []
    for line in open(path, errors="replace"):
        line = line.rstrip()
        if (g := re.match(r"^(wte|value_embeds|lm_head|transformer_matrices|scalars|total)\s+: ([\d,]+)$", line)):
            m["params_" + g.group(1)] = int(g.group(2).replace(",", ""))
        elif (g := re.search(r"Calculated number of iterations.*: ([\d,]+)|Using user-provided number of iterations: ([\d,]+)", line)):
            m["num_iterations"] = int((g.group(1) or g.group(2)).replace(",", ""))
        elif (g := re.search(r"Total number of training tokens: ([\d,]+)", line)):
            m["tokens_trained"] = int(g.group(1).replace(",", ""))
        elif (g := re.search(r"Total training FLOPs estimate: ([\d.e+]+)", line)):
            m["flops"] = float(g.group(1))
        elif (g := re.search(r'"n_embd": (\d+)', line)):
            m["model_dim"] = int(g.group(1))
        elif (g := re.search(r"Validation bpb: ([\d.]+)", line)):
            m["val_bpb"] = float(g.group(1))
        elif (g := re.search(r"CORE metric: ([\d.\-]+)", line)):
            m["core_score"] = float(g.group(1))
        elif (g := re.search(r"Total training time: ([\d.]+)m", line)):
            m["train_time_sec"] = round(float(g.group(1)) * 60, 2)
        elif (g := re.search(r"Peak memory usage: ([\d.]+)MiB", line)):
            m["peak_mem_gib"] = round(float(g.group(1)) / 1024, 1)
        elif (g := re.search(r"tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+)", line)):
            toks.append(int(g.group(1).replace(",", "")))
            mfus.append(float(g.group(2)))
    if mfus:   # median over the second half of training (skip compile/warmup)
        half = lambda xs: statistics.median(xs[len(xs) // 2:])
        m["mfu"], m["tok_per_sec"] = half(mfus), half(toks)
    return m


def cmd_report(a):
    import numpy as np
    ldir = ladder_dir(a.name)
    meta = json.load(open(os.path.join(ldir, "ladder.json")))
    plan = read_plan(ldir)
    rows = []
    for r in plan:
        if not os.path.exists(os.path.join(run_dir(ldir, r["tag"]), "done")):
            continue
        m = parse_log(os.path.join(run_dir(ldir, r["tag"]), "train.log"))
        done = open(os.path.join(run_dir(ldir, r["tag"]), "done")).read()
        wall = re.search(r"wall_time_sec=(\d+)", done)
        n_eff = m.get("params_transformer_matrices", 0) + m.get("params_lm_head", 0)
        rows.append(dict(
            # columns read by upstream dev/scaling_analysis.ipynb
            flops_budget=float(r["flops_budget"]), depth=int(r["depth"]), model_dim=m.get("model_dim"),
            params_wte=m.get("params_wte"), params_value_embeds=m.get("params_value_embeds"),
            params_lm_head=m.get("params_lm_head"), params_transformer=m.get("params_transformer_matrices"),
            params_scalars=m.get("params_scalars"), params_total=m.get("params_total"),
            num_iterations=m.get("num_iterations"), tokens_trained=m.get("tokens_trained"),
            val_bpb=m.get("val_bpb"), core_score=m.get("core_score"), train_time_sec=m.get("train_time_sec"),
            # extras
            nodes=meta.get("nodes", 1), world_size=meta.get("nodes", 1) * meta.get("gpus_per_node", GPUS_PER_NODE),
            tag=r["tag"], flops=m.get("flops"), tokens_per_param=round(m.get("tokens_trained", 0) / n_eff, 2) if n_eff else None,
            mfu=m.get("mfu"), tok_per_sec=m.get("tok_per_sec"), peak_mem_gib=m.get("peak_mem_gib"),
            wall_time_sec=int(wall.group(1)) if wall else None))
    if not rows:
        print(f"no finished runs yet in {ldir} (see `status`)")
        return
    out = os.path.join(ldir, "results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}/{len(plan)} runs finished -> {out}\n")
    print(f"{'tag':<16}{'N_eff':>9}{'tokens':>9}{'tok/N':>7}{'FLOPs':>10}{'val_bpb':>9}{'CORE':>8}{'MFU%':>6}{'mem':>7}{'wall':>7}")
    for x in sorted(rows, key=lambda x: (x["flops_budget"], x["depth"])):
        n_eff = (x["params_transformer"] or 0) + (x["params_lm_head"] or 0)
        val = x["val_bpb"] if x["val_bpb"] is not None else float("nan")
        core = x["core_score"] if x["core_score"] is not None else float("nan")
        print(f"{x['tag']:<16}{n_eff/1e6:>8.0f}M{(x['tokens_trained'] or 0)/1e9:>8.2f}B{x['tokens_per_param'] or 0:>7.1f}"
              f"{x['flops'] or 0:>10.2e}{val:>9.4f}{core:>8.4f}"
              f"{x['mfu'] or 0:>6.1f}{x['peak_mem_gib'] or 0:>6.0f}G{(x['wall_time_sec'] or 0)/60:>6.0f}m")

    if meta["mode"] != "isoflop":
        return
    # Chinchilla approach 2: per budget, fit val_bpb as a parabola in log10(N_eff) and take its minimum
    print("\nIso-FLOP fits (N_eff = transformer matrices + lm_head, as upstream):")
    opts = []
    for c in sorted({x["flops_budget"] for x in rows}):
        pts = [x for x in rows if x["flops_budget"] == c and x["val_bpb"] is not None]
        if len(pts) < 3:
            print(f"  C={c:.2e}: {len(pts)} finished runs, need >= 3 for a fit")
            continue
        logn = np.log10([(x["params_transformer"] + x["params_lm_head"]) for x in pts])
        bpb = np.array([x["val_bpb"] for x in pts])
        a2, a1, a0 = np.polyfit(logn, bpb, 2)
        if a2 <= 0:
            print(f"  C={c:.2e}: no interior minimum (parabola opens down) -> extend the depth range")
            continue
        lo = -a1 / (2 * a2)
        order = np.argsort(logn)
        d_opt = 10 ** np.interp(lo, logn[order], np.log10([pts[i]["tokens_trained"] for i in order]))
        edge = " (minimum outside sampled sizes: extend depths)" if not (logn.min() <= lo <= logn.max()) else ""
        if edge:
            print(f"  C={c:.2e}: fitted minimum outside sampled sizes -> extend depths; excluded from power laws")
            continue
        opts.append((c, 10 ** lo, d_opt))
        print(f"  C={c:.2e}: N*={10**lo/1e6:7.1f}M  D*={d_opt/1e9:6.2f}B  D*/N*={d_opt/10**lo:5.1f}  "
              f"bpb*={a2*lo**2 + a1*lo + a0:.4f}{edge}")
    if len(opts) >= 2:
        lc, ln, ld = (np.log10([o[i] for o in opts]) for i in range(3))
        sn, sd = np.polyfit(lc, ln, 1)[0], np.polyfit(lc, ld, 1)[0]
        print(f"\nPower laws over {len(opts)} budgets: N* ~ C^{sn:.3f}, D* ~ C^{sd:.3f} (Chinchilla: ~0.5 / ~0.5)")
    else:
        print(f"\nPower-law exponents unavailable: {len(opts)} interior minima; need >= 2 budgets with an interior minimum")


def cmd_list(a):
    if not os.path.isdir(LADDERS_DIR):
        return
    for name in sorted(os.listdir(LADDERS_DIR)):
        p = os.path.join(LADDERS_DIR, name, "plan.tsv")
        if os.path.exists(p):
            plan = read_plan(os.path.join(LADDERS_DIR, name))
            done = sum(os.path.exists(os.path.join(run_dir(os.path.join(LADDERS_DIR, name), r["tag"]), "done")) for r in plan)
            print(f"{name:<30}{done:>4}/{len(plan)} done")
        elif os.path.isdir(os.path.join(LADDERS_DIR, name)):
            children = [p for p in os.scandir(os.path.join(LADDERS_DIR, name)) if p.is_dir() and os.path.isfile(os.path.join(p.path, 'run.json'))]
            done = sum(os.path.exists(os.path.join(p.path, 'done')) for p in children)
            print(f"{name:<30}{done:>4}/{len(children)} done (standalone/benchmark runs)")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:   # everything after `--` is passed verbatim to every base_train run
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("plan", "submit"):
        s = sub.add_parser(c)
        s.add_argument("name")
        s.add_argument("--mode", choices=["isoflop", "ratio"], default="isoflop")
        s.add_argument("--flops", type=float, nargs="+", default=DEFAULT_FLOPS, help="isoflop: FLOPs budgets")
        s.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
        s.add_argument("--ratio", type=float, default=12.0, help="ratio mode: tokens per scaling param")
        s.add_argument("--partition", default=None, help="dev (default, <=8 nodes, 12h) | short (1h) | long (3d)")
        s.add_argument("--time", type=int, default=None, help="minutes per run (default: estimated)")
        s.add_argument("--nodes", type=int, default=None, help="nodes per run, four GPUs each (default 2; resume preserves original)")
        s.add_argument("--max-parallel", type=int, default=None, help="max concurrent runs (default 5; subject to Slurm quotas)")
        s.add_argument("--wandb", action="store_true", help="log every run to wandb (default: off)")
        if c == "submit":
            s.add_argument("--resume", action="store_true", help="resubmit unfinished runs of an existing ladder")
    for c in ("status", "report"):
        sub.add_parser(c).add_argument("name")
    sub.add_parser("list")
    a = p.parse_args(argv)
    a.extra = extra
    if hasattr(a, "name") and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.name):
        p.error("ladder name must start with a letter/digit and contain only letters, digits, _, . or -")
    if a.cmd in ("plan", "submit"):
        a.partition_set = a.partition is not None
        a.partition = a.partition or "dev"
        if a.nodes is not None and a.nodes <= 0:
            p.error("nodes must be positive")
        if not getattr(a, "resume", False):
            a.nodes = a.nodes or DEFAULT_NODES
            if a.nodes > PARTITION_MAX_NODES.get(a.partition, math.inf):
                p.error(f"partition {a.partition} allows at most {PARTITION_MAX_NODES[a.partition]} nodes per run")
        a.max_parallel = a.max_parallel if a.max_parallel is not None else (None if getattr(a, "resume", False) else DEFAULT_MAX_PARALLEL)
        if (a.time is not None and a.time <= 0) or (a.max_parallel is not None and a.max_parallel <= 0):
            p.error("time and max-parallel must be positive")
        if getattr(a, "resume", False) and extra:
            sys.exit("--resume reuses the original plan; extra base_train args are not allowed")
    try:
        {"plan": cmd_plan, "submit": cmd_submit, "status": cmd_status, "report": cmd_report, "list": cmd_list}[a.cmd](a)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(getattr(e, "stderr", None) or str(e))


if __name__ == "__main__":
    main()

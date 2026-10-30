"""Summarize node benchmarks, including speedup and GPU scaling efficiency.

Usage: python slurm/benchmarks/report.py [JOB_ID ...] [--ladder smoke-nodes]
Writes qwen3-node-smokes.{csv,json} inside the ladder folder; never submits jobs.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import LADDERS_DIR, parse_log


def summarize(log):
    tag = log.parent.name
    match = re.fullmatch(r'(?:smoke_nodes_)?d(\d+)_n(\d+)_(fixed|scaled)(?:_(\d+))?', tag)
    if not match:
        raise ValueError(f'Unexpected benchmark filename: {log}')
    depth, nodes, mode, job = match.groups()
    text = log.read_text()
    done = log.parent / 'done'
    submission = log.parent / 'submission.json'
    if job is None and submission.exists():
        job = json.loads(submission.read_text())['job_id']
    row = dict(job=int(job) if job else None, tag=tag, depth=int(depth), nodes=int(nodes), mode=mode,
               world_size=int(nodes) * 4, context=8192, state='DONE' if done.exists() else 'INCOMPLETE',
               **parse_log(log))
    steps = re.findall(r'^step (\d+)/(\d+) .*?loss: (\S+)', text, re.MULTILINE)
    row['completed_steps'] = len(steps)
    row['finite_losses'] = all(math.isfinite(float(loss)) for _, _, loss in steps)
    row['validation'] = [dict(step=int(s), bpb=float(b)) for s, b in
                         re.findall(r'Step (\d+) \| Validation bpb: ([\d.]+)', text)]
    if row['state'] == 'DONE':
        expected = row.get('num_iterations', 40)
        assert len(steps) == expected and row['finite_losses'], log
        assert [v['step'] for v in row['validation']] == [0, expected], log
        assert f'Distributed world size: {row["world_size"]}' in text, log
        ranks = {int(r) for r in re.findall(r'Saved optimizer state to: .*_rank(\d+)\.pt', text)}
        assert ranks == set(range(row['world_size'])), (log, ranks)
        row['wall_time_sec'] = int(re.search(r'wall_time_sec=(\d+)', done.read_text())[1])
    batch = re.search(r'Total batch size ([\d,]+) => gradient accumulation steps: (\d+)', text)
    micro = re.search(r'Tokens / micro-batch / rank: (\d+) x', text)
    if batch:
        row.update(global_batch_tokens=int(batch[1].replace(',', '')), accumulation_steps=int(batch[2]))
    if micro:
        row['device_batch'] = int(micro[1])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs', nargs='*', type=int)
    parser.add_argument('--ladder', default='smoke-nodes', help='ladder name or absolute directory')
    args = parser.parse_args()
    root = Path(LADDERS_DIR) / args.ladder
    if not root.is_dir():
        parser.error(f'ladder directory does not exist: {root}')
    rows = [summarize(log) for log in sorted(root.glob('*/train.log'))]
    if args.jobs:
        rows = [r for r in rows if r['job'] in args.jobs]
    if not rows:
        parser.error('no matching run logs')
    rows.sort(key=lambda r: (r['depth'], r['mode'], r['nodes']))
    baselines = {r['depth']: r['tok_per_sec'] for r in rows if r['nodes'] == 1 and r['state'] == 'DONE'}
    for row in rows:
        if row['state'] == 'DONE' and row['depth'] in baselines:
            row['speedup'] = row['tok_per_sec'] / baselines[row['depth']]
            row['scaling_efficiency_pct'] = 100 * row['speedup'] / row['nodes']
    (root / 'qwen3-node-smokes.json').write_text(json.dumps(rows, indent=2) + '\n')
    columns = ['job', 'state', 'depth', 'nodes', 'world_size', 'mode', 'context', 'global_batch_tokens',
               'device_batch', 'accumulation_steps', 'completed_steps', 'tok_per_sec', 'speedup',
               'scaling_efficiency_pct', 'mfu', 'peak_mem_gib', 'wall_time_sec', 'val_bpb', 'finite_losses']
    with (root / 'qwen3-node-smokes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps({key: row[key] for key in columns if key in row}))


if __name__ == '__main__':
    main()

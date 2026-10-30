"""Plan node-throughput probes. Nothing is submitted without --submit.

Each configuration gets a descriptive run folder and its own Slurm job. Jobs run
sequentially to keep the allocation bounded by the largest requested node count.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ladder


def configurations(nodes):
    for n in nodes:
        for mode in ('fixed', 'scaled') if n > 1 else ('fixed',):
            for depth in (12, 24):
                tag = f'd{depth}_n{n}_{mode}'
                batch = 524288 * (n if mode == 'scaled' else 1)
                args = [f'--depth={depth}', '--max-seq-len=8192', '--device-batch-size=16',
                        f'--total-batch-size={batch}', '--num-iterations=40', '--warmup-steps=5',
                        '--matrix-lr=0.005', '--unembedding-lr=0.002', '--scalar-lr=0.1',
                        '--eval-every=40', '--eval-tokens=2097152', '--core-metric-every=-1',
                        '--sample-every=-1', '--run=dummy', f'--model-tag={tag}']
                yield tag, n, batch, args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='new ladder name')
    parser.add_argument('--nodes', nargs='+', type=int, choices=(1, 2, 4), default=[ladder.DEFAULT_NODES],
                        help='default: 2; use --nodes 1 2 4 for a scaling comparison')
    parser.add_argument('--time', type=int, default=15, help='minutes per run')
    parser.add_argument('--submit', action='store_true')
    a = parser.parse_args()
    root = Path(ladder.run_dir(ladder.LADDERS_DIR, a.name))
    if len(set(a.nodes)) != len(a.nodes) or not 0 < a.time <= ladder.PARTITION_MAX_MIN['dev']:
        parser.error('node counts must be distinct and time must fit the dev partition')
    runs = list(configurations(a.nodes))
    for tag, nodes, batch, _ in runs:
        print(f'{tag}: {nodes} node(s), {batch:,} tokens/update, 40 updates -> {root / tag}')
    if not a.submit:
        print('Plan only. Add --submit to launch these probes.')
        return
    if root.exists():
        parser.error(f'{root} already exists; choose a new name')
    previous = []
    for tag, nodes, _, training in runs:
        directory = Path(ladder.prepare_run(str(root), tag, training, nodes=nodes))
        job = ladder.submit_run(str(directory), tag, training, nodes=nodes,
                                partition='dev', time_min=a.time, dependencies=previous)
        (directory / 'submission.json').write_text(json.dumps(dict(job_id=job, nodes=nodes,
            partition='dev', time_min=a.time), indent=2) + '\n')
        previous = [job]


if __name__ == '__main__':
    main()

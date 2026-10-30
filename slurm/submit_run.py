"""Submit one run with every artifact inside ladders/LADDER/TAG/.

Example: python slurm/submit_run.py trials d12 --nodes 2 -- --depth=12 --run=dummy
"""
import argparse
import json
from pathlib import Path
import re
import sys

import ladder


def main():
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ladder')
    parser.add_argument('tag')
    parser.add_argument('--nodes', type=int, default=ladder.DEFAULT_NODES, help='nodes per run (default: 2)')
    parser.add_argument('--workers-per-node', type=int, default=ladder.GPUS_PER_NODE,
                        choices=range(1, ladder.GPUS_PER_NODE + 1),
                        help='GPUs used per node (default: 4); Slurm still reserves the whole four-GPU node')
    parser.add_argument('--partition', default='dev')
    parser.add_argument('--time', type=int, default=60, help='minutes')
    args = parser.parse_args(argv[:split])
    training = argv[split + 1:]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.ladder):
        parser.error('invalid ladder name')
    if args.nodes < 1 or args.nodes > ladder.PARTITION_MAX_NODES.get(args.partition, 0) or args.time <= 0:
        parser.error('invalid nodes, partition or time')
    root = Path(ladder.ladder_dir(args.ladder))
    directory = Path(ladder.run_dir(str(root), args.tag))
    if directory.exists():
        parser.error(f'{directory} already exists; choose a new tag')
    if any(x.split('=',1)[0] == '--run-dir' for x in training):
        parser.error('the run directory is managed by this submission tool')
    training += [f'--model-tag={args.tag}']
    ladder.prepare_run(str(root), args.tag, training, nodes=args.nodes, workers_per_node=args.workers_per_node)
    job = ladder.submit_run(str(directory), args.tag, training, nodes=args.nodes,
                            partition=args.partition, time_min=args.time, workers_per_node=args.workers_per_node)
    (directory / 'submission.json').write_text(json.dumps(dict(job_id=job, nodes=args.nodes,
        workers_per_node=args.workers_per_node, partition=args.partition, time_min=args.time), indent=2) + '\n')
    print(f'Submitted {job}: {directory}')


if __name__ == '__main__':
    main()

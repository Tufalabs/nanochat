"""CPU-only regressions for cluster orchestration; never contact Slurm or launch training."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location('ladder', Path(__file__).parents[1] / 'slurm/ladder.py')
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)


def make_ladder(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, 'LADDERS_DIR', str(tmp_path))
    root = tmp_path / 'test'
    root.mkdir(parents=True)
    for i in range(3):
        (root / f'd{i}').mkdir()
        (root / f'd{i}/run.json').write_text('{}')
    meta = dict(layout_version=2, mode='isoflop', partition='dev', nodes=2,
                gpus_per_node=4, time_min=40, max_parallel=2,
                submissions=[dict(job_id=str(123 + i), tasks=[i], array=False) for i in range(3)])
    (root / 'ladder.json').write_text(json.dumps(meta))
    with (root / 'plan.tsv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['tag', 'depth', 'flops_budget', 'args'], delimiter='\t')
        w.writeheader()
        for i in range(3):
            w.writerow(dict(tag=f'd{i}', depth=i+8, flops_budget=1e17, args='--run=dummy'))
    return root


def resume_args():
    return argparse.Namespace(name='test', resume=True, partition_set=False,
                              partition='dev', time=None, max_parallel=None, nodes=None)


def test_resume_skips_done_and_active(tmp_path, monkeypatch):
    root = make_ladder(tmp_path, monkeypatch)
    (root / 'd0/done').touch()
    monkeypatch.setattr(ladder, 'squeue_states', lambda: {'124': ('RUNNING', '1:00', 'node')})
    calls = []
    def submit(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='456;thur\n')
    monkeypatch.setattr(ladder.subprocess, 'run', submit)
    ladder.cmd_submit(resume_args())
    assert len(calls) == 1
    assert '--dependency=afterany:124' in calls[0]
    assert f'--output={root}/d2/slurm-%j.out' in calls[0]
    assert '--partition=dev' in calls[0]
    assert '--nodes=2' in calls[0]
    assert str(Path(ladder.NANOCHAT_DIR) / 'slurm/jobs/train.sbatch') in calls[0]
    assert json.loads((root / 'ladder.json').read_text())['submissions'][-1]['tasks'] == [2]


def test_dependency_lanes_bound_parallelism_and_save_partial_submission(tmp_path, monkeypatch):
    root = make_ladder(tmp_path, monkeypatch)
    monkeypatch.setattr(ladder, 'squeue_states', lambda: {})
    calls = []
    def submit(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 3:
            raise subprocess.CalledProcessError(1, cmd, stderr='temporary failure')
        return subprocess.CompletedProcess(cmd, 0, stdout=f'{200 + len(calls)}\n')
    monkeypatch.setattr(ladder.subprocess, 'run', submit)
    with pytest.raises(subprocess.CalledProcessError):
        ladder.cmd_submit(resume_args())
    assert not any(x.startswith('--dependency=') for cmd in calls[:2] for x in cmd)
    assert '--dependency=afterany:201' in calls[2]
    saved = json.loads((root / 'ladder.json').read_text())['submissions']
    assert [s['job_id'] for s in saved] == ['123', '124', '125', '201', '202']
    # A retry must not resubmit either accepted job and must wait for them.
    monkeypatch.setattr(ladder, 'squeue_states', lambda: {
        '201': ('RUNNING', '0:01', 'node-a'), '202': ('PENDING', '0:00', 'Resources')})
    ladder.cmd_submit(resume_args())
    assert len(calls) == 4
    assert '--dependency=afterany:201:202' in calls[3]
    assert f'--output={root}/d2/slurm-%j.out' in calls[3]


def test_queue_failure_blocks_resume_but_status_reads_logs(tmp_path, monkeypatch, capsys):
    root = make_ladder(tmp_path, monkeypatch)
    (root / 'd0/train.log').write_text('Traceback (most recent call last):\n')
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'squeue', stderr='no controller')
    monkeypatch.setattr(ladder.subprocess, 'run', unavailable)
    with pytest.raises(RuntimeError, match='no controller'):
        ladder.cmd_submit(resume_args())
    ladder.cmd_status(argparse.Namespace(name='test'))
    out = capsys.readouterr()
    assert 'FAILED' in out.out and 'UNKNOWN' in out.out
    assert 'log evidence only' in out.err
    assert len(json.loads((root / 'ladder.json').read_text())['submissions']) == 3


def test_report_and_status_parse_training_output(tmp_path, monkeypatch, capsys):
    root = make_ladder(tmp_path, monkeypatch)
    log = '''wte                     : 1,000
lm_head                 : 20,000
transformer_matrices    : 80,000
scalars                 : 42
total                   : 101,042
  "n_embd": 768,
Calculated number of iterations from target FLOPs: 1,000
Total number of training tokens: 524,288,000
Total training FLOPs estimate: 1.000000e+17
Step 00000 | Validation bpb: 3.160000
step 00000/01000 (0.00%) | loss: 8.000000 | lrm: 0.05 | dt: 10000.00ms | tok/sec: 52,428 | bf16_mfu: 0.40 | epoch: 1 pq: 0 rg: 0
step 00001/01000 (0.10%) | loss: 7.000000 | lrm: 0.05 | dt: 1000.00ms | tok/sec: 524,288 | bf16_mfu: 4.00 | epoch: 1 pq: 0 rg: 1
step 00998/01000 (99.80%) | loss: 2.000000 | lrm: 0.05 | dt: 200.00ms | tok/sec: 2,621,440 | bf16_mfu: 20.00 | epoch: 1 pq: 14 rg: 8 | total time: 1.23m | eta: 0.1m
step 00999/01000 (99.90%) | loss: 1.999999 | lrm: 0.05 | dt: 200.00ms | tok/sec: 2,621,440 | bf16_mfu: 20.00 | epoch: 1 pq: 14 rg: 8 | total time: 1.23m | eta: 0.0m | optimizer_dt: 10.00ms | optimizer_time_pct: 5.00%
Step 01000 | Validation bpb: 1.123456
Step 01000 | CORE metric: -0.0010
Peak memory usage: 102400.00MiB
Total training time: 1.23m
'''
    p = root / 'd0/train.log'
    p.write_text(log)
    m = ladder.parse_log(p)
    assert m['params_transformer_matrices'] == 80000
    assert m['num_iterations'] == 1000 and m['tokens_trained'] == 524288000
    assert m['model_dim'] == 768 and m['val_bpb'] == 1.123456
    assert m['train_time_sec'] == 73.8 and m['peak_mem_gib'] == 100
    assert m['mfu'] == 20 and m['tok_per_sec'] == 2621440
    assert m['core_score'] == -0.001
    monkeypatch.setattr(ladder, 'squeue_states', lambda: {'123': ('RUNNING', '1:00', 'node')})
    ladder.cmd_status(argparse.Namespace(name='test'))
    output = capsys.readouterr().out
    assert 'step 00999/01000' in output and 'mfu 20.00' in output and 'eta: 0.0m' in output
    (root / 'd0/done').write_text('wall_time_sec=180 job=123\n')
    ladder.cmd_report(argparse.Namespace(name='test'))
    rows = list(csv.DictReader((root / 'results.csv').open()))
    assert len(rows) == 1 and rows[0]['train_time_sec'] == '73.8'
    assert rows[0]['wall_time_sec'] == '180'


def test_batch_script_quotes_success_failure_and_skip(tmp_path):
    import os
    root = tmp_path / 'run'
    root.mkdir()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    nanochat = tmp_path / 'nanochat'
    (nanochat / 'slurm').mkdir(parents=True)
    (nanochat / 'slurm/env.sh').write_text(f'export PATH={shlex.quote(str(bin_dir))}:"$PATH"\nexport GPUS_PER_NODE=4\n')
    (bin_dir / 'nvidia-smi').write_text('#!/bin/bash\nexit 0\n')
    (bin_dir / 'torchrun').write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\nsys.exit(int(os.environ.get("FAIL", "0")))\n')
    for file in bin_dir.iterdir():
        file.chmod(0o755)
    argv = ['--run=dummy', '--model-tag=contains spaces', '--window-pattern=L']
    env = dict(os.environ, NANOCHAT_DIR=str(nanochat), NANOCHAT_RUN_DIR=str(root),
               SLURM_JOB_NUM_NODES='1', SLURM_JOB_ID='test', CAPTURE=str(tmp_path / 'args.json'))
    script = Path(__file__).parents[1] / 'slurm/jobs/train.sbatch'
    run = lambda: subprocess.run(['bash', str(script), *argv], env=env, capture_output=True, text=True)
    env['FAIL'] = '1'
    assert run().returncode == 1
    assert not (root / 'done').exists()
    env['FAIL'] = '0'
    assert run().returncode == 0
    captured = json.loads(Path(env['CAPTURE']).read_text())
    assert captured[-4:-1] == argv and captured[-1] == f'--run-dir={root}'
    assert (root / 'train.log').exists() and (root / 'slurm-test.out').exists()
    assert 'wall_time_sec=' in (root / 'done').read_text()
    Path(env['CAPTURE']).unlink()
    assert run().returncode == 0 and not Path(env['CAPTURE']).exists()


def test_plan_rejects_overrides_that_would_mislabel_results():
    with pytest.raises(ValueError, match='ladder-managed'):
        ladder.estimate_runs([10], 'isoflop', [1e18], 12, ['--depth=32'])
    with pytest.raises(ValueError, match='distinct'):
        ladder.estimate_runs([10, 10], 'isoflop', [1e18], 12, [])
    with pytest.raises(ValueError, match='positive'):
        ladder.estimate_runs([10], 'isoflop', [float('nan')], 12, [])


@pytest.mark.parametrize('context_args', [['--max-seq-len=8192'], ['--max-seq-len', '8192']])
def test_context_batch_rule_reaches_training_and_override_wins(context_args):
    run = dict(depth=32, tag='d32', horizon=['--num-iterations=24'])
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-batch-size', type=int)
    argv = ladder.build_args('context', run, context_args, False)
    parsed, _ = parser.parse_known_args(argv)
    assert parsed.device_batch_size == 8
    argv = ladder.build_args('context', run, context_args + ['--device-batch-size=4'], False)
    parsed, _ = parser.parse_known_args(argv)
    assert parsed.device_batch_size == 4


def test_report_excludes_extrapolated_minima(tmp_path, monkeypatch, capsys):
    root = make_ladder(tmp_path, monkeypatch)
    import math
    for i, n in enumerate((100, 1000, 10000)):
        # Convex parabola whose minimum at N=10**6 is outside the sampled range.
        bpb = 1 + (math.log10(n) - 6)**2
        (root / f'd{i}/train.log').write_text(
            f'transformer_matrices    : {n}\nlm_head                 : 0\n'
            f'Total number of training tokens: 100000\nStep 00010 | Validation bpb: {bpb}\n')
        (root / f'd{i}/done').write_text('wall_time_sec=10\n')
    ladder.cmd_report(argparse.Namespace(name='test'))
    out = capsys.readouterr().out
    assert 'outside sampled sizes' in out
    assert 'Power laws over' not in out


def test_failed_initial_submission_remains_resumable_without_code_copy(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'example.py').write_text('original source\n')
    monkeypatch.setattr(ladder, 'NANOCHAT_DIR', str(source))
    monkeypatch.setattr(ladder, 'LADDERS_DIR', str(tmp_path / 'ladders'))
    run = dict(tag='d8', depth=8, model_dim=512, n_scaling=1, n_total=2,
               flops_budget=1e17, ratio=12, batch=262144, iters=1, tokens=262144,
               flops=1e17, est_min=20, horizon=['--num-iterations=1'], nodes=2,
               world_size=8, device_batch=16, grad_accum_steps=1)
    monkeypatch.setattr(ladder, 'estimate_runs', lambda *args: [run.copy()])
    monkeypatch.setattr(ladder, 'print_plan', lambda *args: None)
    def unavailable(cmd, **kwargs):
        if cmd[0] == 'git':
            return subprocess.CompletedProcess(cmd, 0, stdout='fixture\n')
        raise subprocess.CalledProcessError(1, cmd, stderr='submission unavailable')
    monkeypatch.setattr(ladder.subprocess, 'run', unavailable)
    a = argparse.Namespace(name='test', resume=False, depths=[8], mode='isoflop',
                           flops=[1e17], ratio=12, extra=[], wandb=False, time=40,
                           partition='dev', max_parallel=2, nodes=2)
    with pytest.raises(subprocess.CalledProcessError):
        ladder.cmd_submit(a)
    root = tmp_path / 'ladders/test'
    assert json.loads((root / 'ladder.json').read_text())['submissions'] == []
    assert (root / 'plan.tsv').exists()
    assert not (root / 'code').exists()
    assert not (root / 'code.diff').exists()
    assert json.loads((root / 'd8/run.json').read_text())['source_checkout'] == str(source)
    monkeypatch.setattr(ladder, 'squeue_states', lambda: {})
    monkeypatch.setattr(ladder.subprocess, 'run', lambda cmd, **kwargs:
                        subprocess.CompletedProcess(cmd, 0, stdout='999\n'))
    ladder.cmd_submit(resume_args())
    assert json.loads((root / 'ladder.json').read_text())['submissions'][0]['job_id'] == '999'
    assert json.loads((root / 'ladder.json').read_text())['nodes'] == 2


def test_resume_preserves_nodes_and_rejects_changed_count(tmp_path, monkeypatch):
    root = make_ladder(tmp_path, monkeypatch)
    meta = json.loads((root / 'ladder.json').read_text())
    meta['nodes'] = 4
    meta['partition'] = 'dev'
    (root / 'ladder.json').write_text(json.dumps(meta))
    monkeypatch.setattr(ladder, 'squeue_states', lambda: {})
    calls = []
    def submit(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='456\n')
    monkeypatch.setattr(ladder.subprocess, 'run', submit)
    ladder.cmd_submit(resume_args())
    assert '--nodes=4' in calls[0]
    a = resume_args()
    a.nodes = 2
    with pytest.raises(SystemExit, match='preserves the original node count'):
        ladder.cmd_submit(a)
    assert len(calls) == 3


def test_multinode_plan_preserves_tokens_and_adjusts_microbatch():
    extra = ['--max-seq-len=8192', '--total-batch-size=524288', '--device-batch-size=16']
    plans = [ladder.estimate_runs([12], 'isoflop', [1e18], 12, extra, nodes=n)[0] for n in (1, 2, 4)]
    assert [r['device_batch'] for r in plans] == [16, 8, 4]
    assert all(r['grad_accum_steps'] == 1 for r in plans)
    assert len({r['tokens'] for r in plans}) == 1
    assert plans[2]['est_min'] < plans[1]['est_min'] < plans[0]['est_min']
    with pytest.raises(ValueError, match='optimizer sharding'):
        ladder.estimate_runs([12], 'isoflop', [1e18], 12, extra, nodes=3)
    with pytest.raises(ValueError, match='global batch'):
        ladder.estimate_runs([12], 'isoflop', [1e18], 12,
                             ['--max-seq-len=8192', '--total-batch-size=65536'], nodes=4)


def test_multinode_launcher_ranks_quoting_and_remote_failure(tmp_path):
    import os
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    setup = tmp_path / 'checkout'
    (setup / 'slurm').mkdir(parents=True)
    logs = tmp_path / 'logs'
    logs.mkdir()
    (setup / 'slurm/env.sh').write_text(
        f'export PATH={shlex.quote(str(bin_dir))}:"$PATH"\nexport GPUS_PER_NODE=4\n'
        f'export CODE_DIR={shlex.quote(str(setup))}\nexport LOGS_DIR={shlex.quote(str(logs))}\n')
    (bin_dir / 'nvidia-smi').write_text('#!/bin/bash\nexit 0\n')
    (bin_dir / 'scontrol').write_text('#!/bin/bash\nprintf "node-a\\nnode-b\\n"\n')
    (bin_dir / 'srun').write_text('''#!/usr/bin/env python3
import os, subprocess, sys
from pathlib import Path
Path(os.environ['CAPTURE'] + '.srun').write_text(' '.join(sys.argv[1:]))
command = sys.argv[sys.argv.index('bash'):]
for rank in range(2):
    result = subprocess.run(command, env=dict(os.environ, SLURM_PROCID=str(rank)))
    if result.returncode:
        sys.exit(result.returncode)
''')
    (bin_dir / 'torchrun').write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
rank = os.environ['SLURM_PROCID']
Path(os.environ['CAPTURE'] + rank).write_text(json.dumps(sys.argv[1:]))
sys.exit(7 if rank == '1' and os.environ.get('FAIL_REMOTE') else 0)
''')
    for file in bin_dir.iterdir():
        file.chmod(0o755)
    env = dict(os.environ, NANOCHAT_DIR=str(setup), SLURM_JOB_NUM_NODES='2',
               SLURM_JOB_NODELIST='node-[a,b]', SLURM_JOB_ID='fixture', TAG='multi',
               CAPTURE=str(tmp_path / 'capture'), MASTER_PORT='29500', NANOCHAT_RUN_DIR=str(logs),
               TRITON_CACHE_DIR=str(tmp_path / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(tmp_path / 'inductor'))
    argv = ['--run=dummy', '--model-tag=with spaces']
    script = Path(__file__).parents[1] / 'slurm/jobs/train.sbatch'
    run = lambda: subprocess.run(['bash', str(script), *argv], env=env, capture_output=True, text=True)
    result = run()
    assert result.returncode == 0, result.stderr
    for rank in range(2):
        command = json.loads((tmp_path / f'capture{rank}').read_text())
        assert f'--node_rank={rank}' in command and '--nnodes=2' in command
        assert '--nproc_per_node=4' in command and '--master_addr=node-a' in command
        assert command[-3:-1] == argv
    assert '--kill-on-bad-exit=1' in (tmp_path / 'capture.srun').read_text()
    assert 'world_size=8' in (logs / 'done').read_text()
    (logs / 'done').unlink()
    env['FAIL_REMOTE'] = '1'
    assert run().returncode == 7
    assert not (logs / 'done').exists()


@pytest.mark.parametrize('action', ['plan', 'submit'])
def test_cli_defaults_to_two_nodes_and_five_concurrent_runs(monkeypatch, action):
    calls = []
    monkeypatch.setattr(ladder, f'cmd_{action}', calls.append)
    monkeypatch.setattr(sys, 'argv', ['ladder.py', action, 'example'])
    ladder.main()
    assert calls[-1].nodes == 2 and calls[-1].max_parallel == 5
    monkeypatch.setattr(sys, 'argv', ['ladder.py', action, 'example', '--nodes=1', '--partition=short'])
    ladder.main()
    assert calls[-1].nodes == 1
    monkeypatch.setattr(sys, 'argv', ['ladder.py', action, 'example', '--partition=short'])
    with pytest.raises(SystemExit):
        ladder.main()
    assert len(calls) == 2  # invalid default/partition combination never reaches submission


def test_single_gpu_cli_reserves_whole_node_and_launches_one_worker(tmp_path, monkeypatch):
    import os
    commands = []
    def sbatch(cmd, **kwargs):
        if cmd[0] == 'sbatch':
            commands.append(cmd)
        else:
            assert cmd[0] == 'git', cmd
        return subprocess.CompletedProcess(cmd, 0, stdout='fixture\n')
    path = Path(__file__).parents[1] / 'slurm/submit_run.py'
    spec = importlib.util.spec_from_file_location('submit_single', path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'ladder', ladder)
        spec.loader.exec_module(module)
        patch.setattr(ladder, 'LADDERS_DIR', str(tmp_path))
        patch.setattr(ladder.subprocess, 'run', sbatch)
        patch.setattr(sys, 'argv', [str(path), 'trials', 'd24-gpu1', '--nodes=1',
                                   '--workers-per-node=1', '--partition=short', '--', '--depth=24'])
        module.main()
    assert len(commands) == 1
    assert '--gres=gpu:b300:4' in commands[0]
    assert '--nodes=1' in commands[0]
    exports = next(arg.removeprefix('--export=ALL,') for arg in commands[0] if arg.startswith('--export='))
    job_environment = dict(pair.split('=', 1) for pair in exports.split(','))
    run = tmp_path / 'trials/d24-gpu1'
    assert json.loads((run / 'run.json').read_text())['workers_per_node'] == 1
    assert json.loads((run / 'submission.json').read_text())['job_id'] == 'fixture'
    scripts = tmp_path / 'checkout/slurm'
    scripts.mkdir(parents=True)
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    (scripts / 'env.sh').write_text(f'export PATH={shlex.quote(str(binaries))}:"$PATH"\nexport GPUS_PER_NODE=4\n')
    (binaries / 'nvidia-smi').write_text('#!/bin/bash\nexit 0\n')
    (binaries / 'torchrun').write_text('#!/usr/bin/env python3\nimport os, sys\nfrom pathlib import Path\n'
                                      'Path(os.environ["CAPTURE"]).write_text("\\n".join(sys.argv[1:]))\n')
    for file in binaries.iterdir():
        file.chmod(0o755)
    env = dict(os.environ, **job_environment)
    env.update(NANOCHAT_DIR=str(scripts.parent), SLURM_JOB_NUM_NODES='1', SLURM_JOB_ID='fixture',
               CAPTURE=str(tmp_path / 'args'))
    result = subprocess.run(['bash', str(Path(__file__).parents[1] / 'slurm/jobs/train.sbatch'), '--run=dummy'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--nproc_per_node=1' in (tmp_path / 'args').read_text().splitlines()
    assert json.loads((run / 'execution-fixture.json').read_text())['workers_per_node'] == 1
    assert 'world_size=1' in (run / 'done').read_text()


def test_preset_runner_defaults_to_plan_and_loads_custom_config(tmp_path):
    import os
    import shutil
    source = Path(__file__).parents[1] / 'slurm'
    scripts = tmp_path / 'checkout/slurm'
    (scripts / 'configs').mkdir(parents=True)
    shutil.copy2(source / 'run_scaling_ladder.sh', scripts)
    shutil.copy2(source / 'configs/qwen3_scaling.sh', scripts / 'configs')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    (scripts / 'env.sh').write_text(f'export PATH={shlex.quote(str(binaries))}:"$PATH"\n')
    fake_python = binaries / 'python'
    fake_python.write_text('#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n'
                          'Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
    fake_python.chmod(0o755)
    capture = tmp_path / 'args.json'
    env = dict(os.environ, CAPTURE=str(capture))
    env.pop('LADDER_NAME', None)
    command = ['bash', str(scripts / 'run_scaling_ladder.sh')]
    subprocess.run(command, env=env, cwd=tmp_path, check=True, capture_output=True)
    args = json.loads(capture.read_text())
    assert args[1:3] == ['plan', 'qwen3-scaling-v1']
    assert args[args.index('--nodes') + 1] == '2'
    assert args[args.index('--max-parallel') + 1] == '5'
    assert '--total-batch-size=524288' in args and '--wandb' in args
    custom = tmp_path / 'custom preset.sh'
    custom.write_text((scripts / 'configs/qwen3_scaling.sh').read_text().replace('NODES=2', 'NODES=1'))
    subprocess.run(command + ['plan', str(custom)], env=dict(env, LADDER_NAME='custom-run'),
                   cwd=tmp_path, check=True, capture_output=True)
    args = json.loads(capture.read_text())
    assert args[1:3] == ['plan', 'custom-run']
    assert args[args.index('--nodes') + 1] == '1'

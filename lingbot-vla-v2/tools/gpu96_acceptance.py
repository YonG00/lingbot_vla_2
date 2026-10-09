#!/usr/bin/env python3
"""Single-GPU 96GiB acceptance harness.  PLAN is the default: never rents/starts a GPU.

Uses the real optimizer trainer, RoboTwin open-loop evaluator and HF acceptance entry.
Each GPU subcommand needs --execute, writes to a new path, and never shuts down a host.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = '/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt'
DEFAULT_CONFIG = '/data/train/configs/robotwin_official_paths.yaml'
DEFAULT_SPLIT = '/data/train/task_splits_50'
DEFAULT_DATA = '/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30'
DEFAULT_NORM = str(ROOT / 'assets/norm_stats/robotwin_competition_clean.json')
STEP_RX = re.compile(r'\bStep\s+(\d+)/\d+.*?\bLoss\s+([^,\s]+).*?\bStepTime\s+([^,\s]+)s')
MSE_RX = re.compile(r'MSE for trajectory\s+(\d+):\s*([^,\s]+)')


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def fresh_dir(path: Path) -> Path:
    # Never silently reuse/overwrite a previous acceptance run.
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_files(*paths: str) -> None:
    missing = [s for s in paths if not Path(s).exists()]
    if missing:
        raise ValueError('required file/dir missing: ' + ', '.join(missing))


def require_gpu() -> dict:
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('nvidia-smi not installed; GPU work refused')
    p = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True)
    cards = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    if len(cards) != 1:
        raise RuntimeError(f'expected exactly one visible GPU (CUDA_VISIBLE_DEVICES), got {cards}')
    name, size = [x.strip() for x in cards[0].split(',', 1)]
    mib = float(size)
    # 诊断放行（默认关闭，保持 96G fail-closed）：仅用于小卡上的**数值诊断**，不放行正式验收。
    if mib < 90000 and os.environ.get('AL_ALLOW_SMALL_CARD') != '1':
        raise RuntimeError(f'expected >= 90000 MiB 96G-class card; got {name}: {mib} MiB '
                           f'(set AL_ALLOW_SMALL_CARD=1 only for diagnostics)')
    if mib < 90000:
        print(f'[gpu96] WARNING: 小卡诊断模式（{name} {mib} MiB）；仅数值诊断，非正式验收', flush=True)
    used = gpu_used_mib()
    if used is not None and used > 5120:
        raise RuntimeError(f'GPU is not idle (used={used:.0f} MiB). Refusing contaminated benchmark.')
    return {'name': name, 'memory_total_mib': mib, 'initial_used_mib': used}


def current_git_head() -> str | None:
    try:
        return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,
                              text=True,check=True,timeout=3).stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def gpu_used_mib() -> float | None:
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=4, check=True)
        return float(p.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def monitored_run(cmd: list[str], *, log: Path, env: dict, timeout: int,
                  memory_limit_mib: float | None = None) -> dict:
    """Only kill our process group; never kill someone else's training processes."""
    start = time.monotonic()
    peak = None
    reason = 'exit'
    rc = 1
    with log.open('w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        high = 0
        try:
            while proc.poll() is None:
                used = gpu_used_mib()
                if used is not None:
                    peak = used if peak is None else max(peak, used)
                    high = high + 1 if memory_limit_mib is not None and used > memory_limit_mib else 0
                if high >= 2:
                    reason = 'vram_headroom_guard'
                    break
                if time.monotonic() - start >= timeout:
                    reason = 'timeout'
                    break
                time.sleep(2)
            if reason != 'exit':
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    return {'returncode': rc, 'reason': reason, 'elapsed_seconds': round(time.monotonic()-start, 2),
            'peak_nvidia_smi_mib': peak}


def trainer_command(a: argparse.Namespace) -> list[str]:
    """Match the real al_50task_bf16 launcher, minus --train.auto_learning for pure throughput."""
    gbs = a.micro * a.gas
    return [a.python, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=1',
            '--master-addr=127.0.0.1', f'--master-port={a.master_port}',
            'tasks/vla/train_lingbotvla.py', a.train_config,
            '--model.model_path', a.model_path,
            '--data.train_path', str(Path(a.phases) / 'datasets.txt'),
            '--data.episode_ids_file', str(Path(a.split_dir) / 'combined.train_ids.json'),
            '--data.image_augment', 'false',
            '--train.output_dir', str(a.output),
            '--train.micro_batch_size', str(a.micro),
            '--train.gradient_accumulation_steps', str(a.gas),
            '--train.global_batch_size', str(gbs),
            '--train.num_train_epochs', '1000',
            '--train.max_steps', str(a.step_offset + a.warmup + a.measure),
            '--train.step_offset', str(a.step_offset),
            '--train.save_steps', '0', '--train.save_epochs', '0',
            '--train.save_hf_weights', 'false', '--train.hf_pass_interval', '0',
            '--train.smoke_no_checkpoint', 'true', '--train.disk_guard', 'false',
            '--train.enable_resume', 'false',
            '--train.train_expert_only', 'false', '--train.freeze_vision_encoder', 'true',
            '--train.enable_mixed_precision', 'false',
            '--train.rmpad', 'false', '--train.rmpad_with_pos_ids', 'false']


def parse_steps(log: str, *, warmup: int, measure: int, gbs: int) -> dict:
    rows = []
    for line in log.splitlines():
        match = STEP_RX.search(line)
        if match:
            step, loss, seconds = int(match[1]), float(match[2]), float(match[3])
            rows.append((step, loss, seconds))
    unique = {s: (l, t) for s, l, t in rows}
    usable = sorted(unique.items())
    if len(usable) < warmup + measure:
        raise ValueError(f'only {len(usable)} optimizer step records, need {warmup+measure}')
    if any(not math.isfinite(l) or not math.isfinite(t) or t <= 0 for _, (l, t) in usable):
        raise ValueError('nonfinite loss/steptime or nonpositive steptime')
    last = usable[-measure:]
    elapsed = sum(t for _, (_, t) in last)
    return {'steps_seen': len(usable), 'measured_step_range': [last[0][0], last[-1][0]],
            'mean_step_seconds': round(elapsed / measure, 4),
            'samples_per_second': round(measure*gbs/elapsed, 4),
            'last_loss': round(last[-1][1][0], 6)}


def run_bench(a: argparse.Namespace) -> int:
    a.output = a.out_root / f'micro{a.micro}_gas{a.gas}_bf16_compile{a.compile}'
    a.master_port = a.master_port or free_port()
    cmd = trainer_command(a)
    print('GBS:', a.micro*a.gas, '| AL=OFF | no DCP/HF | fresh optimizer from source weights')
    print('COMMAND:', ' '.join(cmd))
    if not a.execute:
        print('PLAN ONLY: pass --execute after GPU is ready.')
        return 0
    if a.micro <= 0 or a.gas <= 0 or a.measure < 1 or a.warmup < 1:
        raise ValueError('micro/gas/measure/warmup must all be positive')
    gpu = require_gpu()
    require_files(a.python, a.train_config, a.model_path,
                  str(Path(a.phases)/'datasets.txt'),
                  str(Path(a.split_dir)/'combined.train_ids.json'))
    if not Path(a.python).is_file():
        raise ValueError('python must be an absolute interpreter path')
    fresh_dir(a.output)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '0', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                'TORCH_COMPILE_DISABLE': '1' if a.compile == 'off' else '0',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'HF_DATASETS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                'MASTER_PORT': str(a.master_port)})
    env['PATH'] = str(Path(a.python).parent) + os.pathsep + env.get('PATH', '')
    env.setdefault('QWEN3VL_PATH', a.qwen3vl)
    headroom = gpu['memory_total_mib'] - a.reserve_gib*1024
    result = {'kind': 'bench', 'status': 'FAIL', 'git_head': current_git_head(),
              'gpu': gpu, 'micro': a.micro, 'gas': a.gas, 'gbs': a.micro*a.gas,
              'compile': a.compile, 'warmup_steps': a.warmup, 'measured_steps': a.measure,
              'command': cmd}
    write_json(a.output/'plan.json', result)
    attempt = monitored_run(cmd, log=a.output/'train.log', env=env, timeout=a.timeout_sec,
                            memory_limit_mib=headroom)
    result.update(attempt)
    if attempt['returncode'] == 0 and attempt['reason'] == 'exit':
        try:
            result.update(parse_steps((a.output/'train.log').read_text(errors='replace'),
                                      warmup=a.warmup, measure=a.measure, gbs=a.micro*a.gas))
            artifacts = list(a.output.glob('checkpoints/global_step_*')) + list(a.output.glob('hf_milestones/*'))
            if artifacts:
                raise RuntimeError(f'no-checkpoint mode leaked artifacts: {artifacts[:3]}')
            if attempt['peak_nvidia_smi_mib'] is None:
                raise RuntimeError('no GPU memory samples')
            result['status'] = 'PASS'
        except (ValueError, RuntimeError) as exc:
            result['error'] = str(exc)
    write_json(a.output/'result.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


def ratio_check(a: argparse.Namespace) -> int:
    from lingbotvla.auto_learning.batch_ratio import ratio_plan
    gbs = a.micro * a.gas * a.dp
    plans = [ratio_plan(global_batch_size=gbs, dp_size=a.dp, dp_rank=i, new_ratio=a.new_ratio)
             for i in range(a.dp)]
    out = {'kind': 'ratio_contract_CPU_ONLY', 'micro': a.micro, 'gas': a.gas,
           'dp': a.dp, 'gbs': gbs, 'new_total': plans[0].global_new,
           'replay_total': plans[0].global_replay,
           'per_rank': [{'new': p.local_new, 'replay': p.local_replay} for p in plans]}
    assert sum(p.local_new for p in plans) == plans[0].global_new
    assert sum(p.local_replay for p in plans) == plans[0].global_replay
    assert all(p.local_batch_size == a.micro*a.gas for p in plans)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print('This validates allocation logic only, NOT real GPU DataLoader/Replay provenance.')
    return 0


def extract_trajectory_mse(raw: str, expected_ids: list[int]) -> dict[int, float]:
    seen = {}
    for line in raw.splitlines():
        match = MSE_RX.search(line)
        if match:
            key = int(match[1]); value = float(match[2])
            if key in seen:
                raise ValueError(f'duplicate evaluated trajectory ID {key}')
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid trajectory {key} MSE {value}')
            seen[key] = value
    if set(seen) != set(expected_ids):
        raise ValueError(f'ID mismatch: missing {sorted(set(expected_ids)-set(seen))}, '
                         f'extra {sorted(set(seen)-set(expected_ids))}')
    return seen


def load_task_ids(split_dir: Path, tasks: list[str], n: int = 10) -> dict[str, list[int]]:
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('select unique task names')
    d = {}
    for task in tasks:
        if '/' in task or '\\' in task or task.startswith('.'):
            raise ValueError(f'invalid task {task!r}')
        path = split_dir / f'{task}.val_ids.json'
        ids = json.loads(path.read_text())
        if not isinstance(ids, list) or len(ids) < n:
            raise ValueError(f'{task}: expected >= {n} val IDs; got {len(ids) if isinstance(ids,list) else type(ids)}')
        chosen = [int(x) for x in ids[:n]]
        if len(set(chosen)) != n:
            raise ValueError(f'duplicate IDs in {task}')
        d[task] = chosen
    flat = [id for ids in d.values() for id in ids]
    if len(set(flat)) != len(flat):
        raise ValueError('trajectory ID shared across tasks; cannot assign task safely')
    return d


def load_thresholds(path: str | None) -> dict:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if raw.get('metric') != 'mse' or raw.get('stat') != 'geomean':
        raise ValueError('threshold table must declare metric=mse, stat=geomean')
    return raw['tasks']


def gmean_summary(task_ids: dict[str,list[int]], measured: dict[str,dict[int,float]],
                  thresholds: dict) -> list[dict]:
    from lingbotvla.auto_learning.decision.gmean import geometric_mse
    records = []
    for task, ids in task_ids.items():
        for n in (2,4,10):
            row = {'task': task, 'n_traj': n, 'traj_ids': ids[:n],
                   'threshold': thresholds.get(task)}
            for precision, values in measured.items():
                g = geometric_mse([values[i] for i in ids[:n]], expected_count=n, ids=ids[:n])
                if g is None:
                    raise ValueError(f'invalid {precision} GMean for {task} n={n}')
                row[precision + '_gmean_mse'] = g
                if row['threshold'] is not None:
                    t = float(row['threshold'])
                    if not math.isfinite(t) or t <= 0:
                        raise ValueError(f'invalid threshold for {task}: {t}')
                    row[precision+'_threshold_ratio'] = g/t
                    row[precision+'_passes_experimental_threshold'] = g <= t
            if len(measured) == 2:
                x = row['bf16_gmean_mse']; y = row['fp32_gmean_mse']
                row['bf16_vs_fp32_relative_difference'] = (x-y)/max(y, 1e-30)
                if row['threshold'] is not None:
                    row['pass_flip'] = (row['bf16_passes_experimental_threshold'] !=
                                        row['fp32_passes_experimental_threshold'])
            records.append(row)
    return records


def run_gmean(a: argparse.Namespace) -> int:
    task_ids = load_task_ids(Path(a.split_dir), a.tasks, 10)
    threshold = load_thresholds(a.thresholds)
    ids = [i for values in task_ids.values() for i in values]
    print(f'PAIR PLAN: {len(a.tasks)} tasks, {len(ids)} unique trajectories, BF16/FP32, '
          f'seed={a.seed_base}, repeats={a.noise_repeats}; derive 2/4/10 from same 10.')
    if not a.execute:
        print(json.dumps(task_ids, indent=2, ensure_ascii=False))
        print('PLAN ONLY: pass --execute to run the two actual open-loop evaluations.')
        return 0
    require_gpu()
    require_files(a.python, a.model_path, a.dataset, a.norm_path)
    if a.thresholds:
        require_files(a.thresholds)
    fresh_dir(a.out_dir)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                'TORCH_COMPILE_DISABLE': '1'})
    env.setdefault('QWEN3VL_PATH', a.qwen3vl)
    outputs: dict[str,dict[int,float]] = {}
    for dtype in ('bf16', 'fp32'):
        cmd = [a.python, '-u', 'scripts/open_loop_eval.py', '--model_path', a.model_path,
               '--robo_name', 'robotwin', '--norm_path', a.norm_path,
               '--data_path', a.dataset, '--traj_ids', *map(str, ids),
               '--save_plot_path', str(a.out_dir/f'plots_{dtype}'), '--no_plot',
               '--fixed_seed_per_traj', '--seed_base', str(a.seed_base),
               '--noise_repeats', str(a.noise_repeats)]
        if dtype == 'bf16':
            cmd.append('--use_bf16')
        print('EVAL:', dtype, ' '.join(cmd))
        r = monitored_run(cmd, log=a.out_dir/f'{dtype}.log', env=env, timeout=a.timeout_sec)
        if r['returncode'] != 0 or r['reason'] != 'exit':
            write_json(a.out_dir/'status.json', {'status': 'BLOCKED', 'precision': dtype, **r})
            raise RuntimeError(f'{dtype} evaluator failed ({r}); see {a.out_dir/dtype}.log; '
                               'no dtype workarounds applied')
        outputs[dtype] = extract_trajectory_mse((a.out_dir/f'{dtype}.log').read_text(errors='replace'), ids)
    summary = {'status': 'PASS', 'kind': 'gmean_bf16_fp32_candidate_paired',
               'git_head':current_git_head(), 'model_path': a.model_path, 'candidate_norm_path': a.norm_path,
               'paired_fixed_seed_per_traj': True, 'seed_base': a.seed_base,
               'noise_repeats': a.noise_repeats, 'thresholds_experimental_only': bool(a.thresholds),
               'reference_model_not_run': True,
               'results': gmean_summary(task_ids, outputs, threshold)}
    write_json(a.out_dir/'result.json', summary)
    print('RESULT:', a.out_dir/'result.json')
    return 0


def hf_plan(a: argparse.Namespace) -> int:
    """Validate an Agent-authored command instead of inventing risky train flags."""
    script = ROOT/'tools/hf_direct_export_acceptance.py'
    if not script.exists():
        raise ValueError(f'HF acceptance entry missing: {script}')
    text = (
        'HF GPU validation uses existing tools/hf_direct_export_acceptance.py.\n'
        'Agent must provide a vetted short AL+BF16 trainer command (same real model / split / norm)\n'
        'with this Python entry INSTEAD of tasks/vla/train_lingbotvla.py.\n'
        'Set --train.hf_export_dtype bf16, --train.hf_pass_interval 1,\n'
        '--train.save_hf_weights false, --train.async_save_hf_weights false,\n'
        '--train.save_steps (far larger than the run), --train.save_epochs 0,\n'
        '--train.skip_final_save_on_max_steps true (acceptance only),\n'
        '--train.smoke_no_checkpoint false, and an isolated output directory.\n'
        'Do NOT forge PASS, modify Scheduler, or call direct_hf_checkpoint from a fake model.\n'
        'Do NOT call the production launcher with SMOKE_NO_CHECKPOINT=1 (it would disable HF).\n'
        'Verify trainer integration and short-AL config using --help/DRY_RUN before executing;\n'
        'if no verified short AL path exists, report BLOCKED rather than spending GPU time.\n')
    print(text)
    return 0




def ratio_unit_evidence(event_rows: list[dict], *, gbs: int, new_ratio: float) -> dict:
    """Match consumed TrainUnit events to actual per-unit sampling metric (no faked PASS)."""
    from lingbotvla.auto_learning.batch_ratio import ratio_plan
    plan = ratio_plan(global_batch_size=gbs, dp_size=1, dp_rank=0, new_ratio=new_ratio)
    old_by_step = {int(r['step']):int(round(float(r['value'])))
                   for r in event_rows if r.get('kind') == 'metric'
                   and r.get('name') == 'sampling/replay_samples_per_unit'
                   and r.get('value') is not None}
    units = []
    for r in event_rows:
        if r.get('kind') != 'event' or r.get('action') != 'train_unit':
            continue
        step = int(r['step'])
        steps = int(r.get('batches_built', 0))
        samples = int(r.get('samples_seen', 0))
        replay = old_by_step.get(step)
        expected = steps*plan.local_replay
        unit = {'step':step,'optimizer_steps':steps,'samples_seen':samples,
                'actual_replay':replay,'expected_replay_when_pool_nonempty':expected,
                'actual_new':samples-replay if replay is not None else None,
                'expected_new_when_pool_nonempty':steps*plan.local_new,
                'replay_tasks':r.get('old_tasks', [])}
        unit['full_batch_valid'] = (steps > 0 and samples == steps*gbs)
        unit['replay_ratio_valid'] = (bool(unit['replay_tasks']) and replay == expected)
        units.append(unit)
    verdict = 'PASS' if any(x['full_batch_valid'] and x['replay_ratio_valid'] for x in units) else 'BLOCKED'
    if not units:
        diagnosis = 'no_consumed_train_unit'
    elif not any(x['replay_tasks'] for x in units):
        diagnosis = 'train_unit_without_replay_pool'
    else:
        diagnosis = 'replay_present_but_consumption_unverified'
    return {'status':verdict,'kind':'actual_AL_ratio_unit_evidence', 'gbs':gbs,
            'expected_per_step':{'new':plan.local_new,'replay':plan.local_replay},
            'units':units, 'diagnosis': 'verified' if verdict == 'PASS' else diagnosis,
            'note':'Only observed consumed Replay can PASS; bootstrap and ratio plans alone never PASS.'}


def build_ratio_smoke_config(raw: dict, *, target: int, max_global_steps: int,
                             max_named_tasks: int = 3) -> dict:
    """Build an isolated AL smoke config; never mutate the formal/source YAML."""
    original = raw.get('auto_learning', raw)
    names = original.get('task_names') or ()
    if original.get('pass_metric', 'nmse') != 'nmse' or not 1 <= len(names) <= max_named_tasks:
        raise ValueError(f'tiny AL ratio probe requires 1..{max_named_tasks} explicitly named '
                         'tasks and the NMSE metric (raise --max-named-tasks deliberately)')
    if isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= len(names):
        raise ValueError(f'target must be 1..{len(names)} named tasks; got {target!r}')
    if not isinstance(max_global_steps, int) or max_global_steps <= 0:
        raise ValueError('max_global_steps must be positive')
    b = dict(original)
    b['new_ratio'] = 0.7
    b['hardness_probe_fraction'] = min(float(b.get('hardness_probe_fraction', .1)), .1)
    b['target_total_passed_tasks'] = target
    b['max_global_steps'] = max_global_steps
    return b


def run_ratio_gpu(a: argparse.Namespace) -> int:
    """Real tiny-task AL training with no checkpoint; inspect consumed unit accounting."""
    a.output = a.out_root / f'ratio_real_micro{a.micro}_gas{a.gas}_target{a.target_total_passed_tasks}_steps{a.steps}'
    a.master_port = a.master_port or free_port()
    cmd = trainer_command(a)
    cmd[cmd.index('--train.max_steps')+1] = str(a.step_offset+a.steps)
    cfg_path = a.output/'isolated_ratio_smoke.yaml'
    cmd.extend(['--train.auto_learning',str(cfg_path),
                '--train.auto_learning_manifest',str(Path(a.split_dir)/'manifest.json'),
                '--train.auto_learning_baseline',str(Path(a.split_dir)/'task_baseline.json')])
    if a.steps < 1 or a.target_total_passed_tasks < 1 or a.max_named_tasks < 1:
        raise ValueError('steps, target_total_passed_tasks and max_named_tasks must be positive')
    print('REAL RATIO AL SMOKE:', a.micro*a.gas, 'GBS; isolated NMSE tasks, no DCP/HF')
    print('TARGET:', a.target_total_passed_tasks, 'MAX OPTIMIZER STEPS:', a.steps,
          'WARNING: target does not guarantee Bootstrap PASS or consumed Replay')
    print('COMMAND:', ' '.join(cmd))
    if not a.execute:
        print('PLAN ONLY; no GPU, files, or outputs touched.')
        return 0
    gpu = require_gpu()
    require_files(a.python,a.model_path,a.train_config,a.al_config,
                  str(Path(a.split_dir)/'manifest.json'),
                  str(Path(a.split_dir)/'task_baseline.json'),
                  str(Path(a.split_dir)/'combined.train_ids.json'),
                  str(Path(a.phases)/'datasets.txt'))
    import yaml
    raw = yaml.safe_load(Path(a.al_config).read_text(encoding='utf-8')) or {}
    b = build_ratio_smoke_config(raw, target=a.target_total_passed_tasks,
                                   max_global_steps=a.step_offset+a.steps,
                                   max_named_tasks=a.max_named_tasks)
    fresh_dir(a.output)
    cfg_path.write_text(yaml.safe_dump(b,allow_unicode=True,sort_keys=False),encoding='utf-8')
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'0','TORCH_COMPILE_DISABLE':'1',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
                'MASTER_PORT':str(a.master_port),
                'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
    env.setdefault('QWEN3VL_PATH',a.qwen3vl)
    env['PATH'] = str(Path(a.python).parent)+os.pathsep+env.get('PATH','')
    r = monitored_run(cmd,log=a.output/'train.log',env=env,timeout=a.timeout_sec,
                      memory_limit_mib=gpu['memory_total_mib']-a.reserve_gib*1024)
    events_path=a.output/'auto_learning_events.jsonl'
    rows = [json.loads(x) for x in events_path.read_text().splitlines() if x.strip()] if events_path.exists() else []
    evidence = ratio_unit_evidence(rows,gbs=a.micro*a.gas,new_ratio=.7)
    evidence.update(r)
    evidence['gpu']=gpu
    evidence['git_head']=current_git_head()
    evidence['test_target_total_passed_tasks']=a.target_total_passed_tasks
    evidence['test_task_names']=b['task_names']
    evidence['configured_max_global_steps']=b['max_global_steps']
    if r['returncode'] != 0 or r['reason'] != 'exit':
        evidence['status']='FAIL'
    elif list(a.output.glob('checkpoints/global_step_*')) or list(a.output.glob('hf_milestones/*')):
        evidence['status']='FAIL'; evidence['error']='no-checkpoint smoke wrote a checkpoint'
    write_json(a.output/'result.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False,indent=2))
    return 0 if evidence['status']=='PASS' else 2


#: 同一阈值 pass_nmse=1.66 下**历史实测不会 PASS**的任务（多次测量稳定）：
#: turn_switch 1.762~1.763（4 次）、put_object_cabinet 2.167。
#: 隔离配置必须含至少一个此类任务，否则全部任务会在 Bootstrap 通过 ⇒ `all_tasks_resolved`
#: 零训练步收工 ⇒ **里程碑判定点根本不会被触达** ⇒ 不会导出（必须避免）。
HF_NON_PASSING_TASKS = ('turn_switch', 'put_object_cabinet')


#: 训练器 hf_pass_interval 的启动前置条件（与 tasks/vla/train_lingbotvla.py 的校验一一对应）。
HF_REQUIRED_FLAGS = {
    '--train.smoke_no_checkpoint': lambda v: v == 'false',
    '--train.save_hf_weights': lambda v: v == 'false',
    '--train.async_save_hf_weights': lambda v: v == 'false',
    '--train.dcp_save_mode': lambda v: v == 'always',
    '--train.save_steps': lambda v: int(v) >= 1,
    '--train.save_epochs': lambda v: int(v) == 0,
    '--train.hf_pass_interval': lambda v: int(v) >= 1,
    '--train.hf_export_dtype': lambda v: v in ('bf16', 'fp32', 'native'),
}


def hf_flag_value(cmd: list, flag: str):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def apply_hf_trainer_overrides(cmd: list) -> list:
    """把 HF 验收命令归一化到满足训练器全部前置条件（否则启动即 ValueError ⇒ 零步失败）。

    save_steps 取远大于本次步数的值：既满足 save_steps >= 1，又保证不会触发周期性 DCP；
    配合 skip_final_save_on_max_steps=true ⇒ 整个验收只写一个 HF 里程碑、不写任何 DCP。
    """
    def set_flag(flag: str, value: str) -> None:
        if flag in cmd:
            cmd[cmd.index(flag) + 1] = value
        else:
            cmd.extend([flag, value])
    for flag, value in (('--train.smoke_no_checkpoint', 'false'),
                        ('--train.hf_pass_interval', '1'),
                        ('--train.save_epochs', '0'),
                        ('--train.save_hf_weights', 'false'),
                        ('--train.async_save_hf_weights', 'false'),
                        ('--train.dcp_save_mode', 'always'),
                        ('--train.save_steps', '1000000000'),
                        ('--train.skip_final_save_on_max_steps', 'true'),
                        ('--train.hf_export_dtype', 'bf16')):
        set_flag(flag, value)
    return cmd


def build_hf_smoke_config(base: dict, *, step_offset: int, steps: int) -> dict:
    """构造 HF 验收专用隔离 AL 配置（纯函数，便于 CPU 测试）。

    * 判定口径显式 **NMSE**（pass_metric='nmse'、pass_nmse=1.66），**不引入实验性阈值表**；
    * 任务集必含历史不通过任务，且 target = 任务总数 ⇒ 调度器必须尝试它 ⇒ **≥1 个 Train Unit**；
    * 不含 new_ratio 分支时要求 batch_size == GBS10（由调用方先行校验）。
    """
    if type(step_offset) is not int or type(steps) is not int or steps < 1:
        raise ValueError('step_offset/steps must be ints with steps >= 1')
    cfg = dict(base)
    # 正式 YAML 用 task_names: null 表示全部任务 ⇒ 必须容忍 None，否则解包 NoneType 崩溃。
    base_tasks = list(cfg.get('task_names') or [])
    tasks = list(dict.fromkeys([*base_tasks, *HF_NON_PASSING_TASKS]))
    cfg['task_names'] = tasks
    cfg['pass_metric'] = 'nmse'
    cfg['pass_nmse'] = 1.66
    cfg['pass_thresholds_file'] = None
    cfg['target_total_passed_tasks'] = len(tasks)
    cfg['max_global_steps'] = step_offset + steps
    cfg['eval_interval_steps'] = 3
    cfg['min_steps_before_defer'] = 6
    cfg['defer_retry_steps'] = 3
    cfg['hardness_probe_fraction'] = 0.10
    cfg['review_after_task_transitions'] = 100
    return cfg


def hf_step_evidence(out_dir) -> dict:
    """判断"里程碑判定点是否被触达"（fail-closed：读不到即 0）。"""
    import json as _json
    import re as _re
    from pathlib import Path as _Path
    units_run, steps_seen = 0, 0
    ev = _Path(out_dir) / 'auto_learning_events.jsonl'
    if ev.exists():
        for line in ev.read_text(encoding='utf-8', errors='ignore').splitlines():
            try:
                row = _json.loads(line)
            except Exception:
                continue
            if row.get('kind') == 'metric' and row.get('name') == 'system/units_run':
                try:
                    units_run = max(units_run, int(round(float(row.get('value')))))
                except Exception:
                    pass
    log = _Path(out_dir) / 'train_hf.log'
    if log.exists():
        for m in _re.finditer(r'Step:\s*(\d+)/', log.read_text(encoding='utf-8', errors='ignore')):
            steps_seen = max(steps_seen, int(m.group(1)))
    return {'units_run': units_run, 'steps_seen': steps_seen}


def hf_verdict(*, returncode: int, reason: str, n_milestones: int, n_shards: int,
               size_gib: float, leaked_dcp: int, units_run: int, steps_seen: int):
    """HF 验收判定（纯谓词）：任一硬条件不满足即 FAIL 并给出原因。"""
    if returncode != 0 or reason != 'exit':
        return False, 'entry_nonzero_exit'
    if units_run < 1 and steps_seen < 1:
        return False, 'no_optimizer_step_so_milestone_decision_point_never_reached'
    if n_milestones != 1:
        return False, f'expected_exactly_one_hf_milestone_got_{n_milestones}'
    if n_shards < 1:
        return False, 'no_weight_shard_published'
    if not (5 < size_gib < 19):
        return False, f'bf16_size_out_of_band:{size_gib:.3f}GiB'
    if leaked_dcp:
        return False, f'dcp_leak:{leaked_dcp}'
    return True, 'verified'


def run_hf(a: argparse.Namespace) -> int:
    """One actual BF16 HF export through the EXISTING production acceptance entry."""
    a.output = a.out_root / f'hf_bf16_micro{a.micro}_gas{a.gas}'
    a.master_port = a.master_port or free_port()
    if a.micro*a.gas != 10:
        raise ValueError('HF acceptance uses GBS=10 to control startup cost')
    if a.steps < 3:
        raise ValueError('HF acceptance needs at least three optimizer steps')
    cfg_path = a.output/'isolated_hf_nmse_smoke.yaml'
    cmd = trainer_command(a)
    cmd[cmd.index('tasks/vla/train_lingbotvla.py')] = 'tools/hf_direct_export_acceptance.py'
    cmd[cmd.index('--train.max_steps')+1] = str(a.step_offset+a.steps)
    # 归一化到训练器 hf_pass_interval 的全部前置条件（save_steps>0 / dcp_save_mode=always / async=false …）
    apply_hf_trainer_overrides(cmd)
    cmd.extend(['--train.auto_learning', str(cfg_path),
                '--train.auto_learning_manifest', str(Path(a.split_dir)/'manifest.json'),
                '--train.auto_learning_baseline', str(Path(a.split_dir)/'task_baseline.json')])
    print('HF PRODUCTION ACCEPTANCE: BF16, short AL, 1 forced policy decision, '
          'no forged PASS, no final DCP; expected model-only ~12 GiB.')
    print('COMMAND:', ' '.join(cmd))
    if not a.execute:
        print('PLAN ONLY: inspect the tiny-AL config and GPU availability before --execute.')
        return 0
    gpu = require_gpu()
    require_files(a.python, a.model_path, a.train_config, a.al_config,
                  str(Path(a.split_dir)/'combined.train_ids.json'),
                  str(Path(a.split_dir)/'manifest.json'),
                  str(Path(a.split_dir)/'task_baseline.json'),
                  str(Path(a.phases)/'datasets.txt'))
    # A short, explicitly NMSE-based AL test config. Do not enable experimental thresholds.
    import yaml
    cfg = yaml.safe_load(Path(a.al_config).read_text(encoding='utf-8')) or {}
    cfg = cfg.get('auto_learning', cfg)
    if cfg.get('pass_metric', 'nmse') != 'nmse':
        raise ValueError('HF isolated acceptance requires NMSE (not experimental GMean table)')
    if cfg.get('new_ratio') is None and int(cfg.get('batch_size', -1)) != 10:
        raise ValueError('HF AL config batch_size must equal the GBS10 smoke')
    # 隔离配置：显式 NMSE + **含历史不通过任务** ⇒ 保证 ≥1 个 Train Unit ⇒ 里程碑判定点可达。
    cfg = build_hf_smoke_config(cfg, step_offset=a.step_offset, steps=a.steps)
    fresh_dir(a.output)
    cfg_path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'0', 'TORCH_COMPILE_DISABLE':'1',
                'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                'HF_DATASETS_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false',
                'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
                'MASTER_PORT':str(a.master_port)})
    env['PATH'] = str(Path(a.python).parent)+os.pathsep+env.get('PATH','')
    env.setdefault('QWEN3VL_PATH', a.qwen3vl)
    r = monitored_run(cmd,log=a.output/'train_hf.log',env=env,timeout=a.timeout_sec)
    paths = sorted(a.output.glob('hf_milestones/global_step_*/hf_ckpt'))
    shards = list(paths[-1].glob('*.safetensors')) if paths else []
    leaked_dcp = list(a.output.glob('checkpoints/global_step_*'))
    size_gib = sum(p.stat().st_size for p in shards)/2**30
    # Standalone acceptance executable now returns rc != 0 on tensor mismatch.
    step_ev = hf_step_evidence(a.output)
    passed, why = hf_verdict(returncode=r['returncode'], reason=r['reason'],
                             n_milestones=len(paths), n_shards=len(shards),
                             size_gib=size_gib, leaked_dcp=len(leaked_dcp),
                             units_run=step_ev['units_run'], steps_seen=step_ev['steps_seen'])
    result = {'kind':'bf16_hf_direct', 'status':'PASS' if passed else 'FAIL', 'verdict_reason':why,
              'gpu':gpu, 'git_head':current_git_head(), 'command':cmd, 'hf_paths':[str(p) for p in paths],
              'weight_shards':len(shards), 'weight_size_gib':round(size_gib,3),
              'dcp_leaks':[str(x) for x in leaked_dcp], 'units_run':step_ev['units_run'],
              'max_global_step_seen':step_ev['steps_seen'], **r}
    write_json(a.output/'result.json',result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    bench = sub.add_parser('bench', help='one real BF16 optimization benchmark; plan-only by default')
    bench.add_argument('--execute', action='store_true')
    bench.add_argument('--micro', type=int, default=28)
    bench.add_argument('--gas', type=int, default=1)
    bench.add_argument('--warmup', type=int, default=5)
    bench.add_argument('--measure', type=int, default=20)
    bench.add_argument('--step-offset', type=int, default=500)
    bench.add_argument('--compile', choices=['off', 'on'], default='off')
    bench.add_argument('--model-path', default=DEFAULT_MODEL)
    bench.add_argument('--train-config', default=DEFAULT_CONFIG)
    bench.add_argument('--split-dir', default=DEFAULT_SPLIT)
    bench.add_argument('--phases', default='/data/train/phases')
    bench.add_argument('--python', default='/data/miniconda3/envs/lingbotvla/bin/python')
    bench.add_argument('--qwen3vl', default='/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct')
    bench.add_argument('--out-root', type=Path, default=Path('/data/outputs/gpu96_acceptance'))
    bench.add_argument('--reserve-gib', type=float, default=8.0)
    bench.add_argument('--timeout-sec', type=int, default=1800)
    bench.add_argument('--master-port', type=int, default=None)
    ratio = sub.add_parser('ratio', help='CPU-only exact optimizer-step allocation contract')
    ratio.add_argument('--micro', type=int, default=28)
    ratio.add_argument('--gas', type=int, default=2)
    ratio.add_argument('--dp', type=int, default=1)
    ratio.add_argument('--new-ratio', type=float, default=0.7)
    gm = sub.add_parser('gmean', help='paired physical-action evaluation (explicit execute only)')
    gm.add_argument('--execute', action='store_true')
    gm.add_argument('--tasks', nargs='+', default=['click_bell','click_alarmclock','adjust_bottle'])
    gm.add_argument('--model-path', default=DEFAULT_MODEL)
    gm.add_argument('--norm-path', default=DEFAULT_NORM)
    gm.add_argument('--dataset', default=DEFAULT_DATA)
    gm.add_argument('--split-dir', default=DEFAULT_SPLIT)
    gm.add_argument('--thresholds', default=None)
    gm.add_argument('--seed-base', type=int, default=1234)
    gm.add_argument('--noise-repeats', type=int, default=1)
    gm.add_argument('--python', default='/data/miniconda3/envs/lingbotvla/bin/python')
    gm.add_argument('--qwen3vl', default='/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct')
    gm.add_argument('--out-dir', type=Path, default=Path('/data/outputs/gpu96_acceptance/gmean_pair'))
    gm.add_argument('--timeout-sec', type=int, default=3600)
    ratio_real = sub.add_parser('ratio-gpu', help='real 2-task AL/Replay probe; plan-only by default')
    ratio_real.add_argument('--execute',action='store_true')
    ratio_real.add_argument('--micro',type=int,default=28)
    ratio_real.add_argument('--gas',type=int,default=2)
    ratio_real.add_argument('--warmup',type=int,default=1)
    ratio_real.add_argument('--measure',type=int,default=1)
    ratio_real.add_argument('--steps',type=int,default=9)
    ratio_real.add_argument('--target-total-passed-tasks',type=int,default=2,
                            help='isolated test goal; default 2, must not exceed named task count')
    ratio_real.add_argument('--max-named-tasks',type=int,default=3,
                            help='safety cap on named AL tasks; raise deliberately (e.g. 4)')
    ratio_real.add_argument('--step-offset',type=int,default=500)
    ratio_real.add_argument('--compile',choices=['off'],default='off')
    ratio_real.add_argument('--model-path',default=DEFAULT_MODEL)
    ratio_real.add_argument('--train-config',default=DEFAULT_CONFIG)
    ratio_real.add_argument('--eval-inference-dtype', default=None,
        help='诊断用：auto/bf16/fp32；透传 --train.eval_inference_dtype')
    ratio_real.add_argument('--al-config',default=str(ROOT/'configs/auto_learning/smoke_2task.yaml'))
    ratio_real.add_argument('--split-dir',default=DEFAULT_SPLIT)
    ratio_real.add_argument('--phases',default='/data/train/phases')
    ratio_real.add_argument('--python',default='/data/miniconda3/envs/lingbotvla/bin/python')
    ratio_real.add_argument('--qwen3vl',default='/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct')
    ratio_real.add_argument('--out-root',type=Path,default=Path('/data/outputs/gpu96_acceptance'))
    ratio_real.add_argument('--reserve-gib',type=float,default=8)
    ratio_real.add_argument('--master-port',type=int,default=None)
    ratio_real.add_argument('--timeout-sec',type=int,default=2400)
    hf = sub.add_parser('hf', help='real single-GPU BF16 HF production-path export; plan-only by default')
    hf.add_argument('--execute',action='store_true')
    hf.add_argument('--micro',type=int,default=10)
    hf.add_argument('--gas',type=int,default=1)
    hf.add_argument('--warmup',type=int,default=1)
    hf.add_argument('--measure',type=int,default=1)
    hf.add_argument('--steps',type=int,default=3)
    hf.add_argument('--step-offset',type=int,default=500)
    hf.add_argument('--compile',choices=['off','on'],default='off')
    hf.add_argument('--model-path',default=DEFAULT_MODEL)
    hf.add_argument('--al-config', default=str(ROOT/'configs/auto_learning/formal_50task_4pass.yaml'))
    hf.add_argument('--train-config',default=DEFAULT_CONFIG)
    hf.add_argument('--split-dir',default=DEFAULT_SPLIT)
    hf.add_argument('--phases',default='/data/train/phases')
    hf.add_argument('--python',default='/data/miniconda3/envs/lingbotvla/bin/python')
    hf.add_argument('--qwen3vl',default='/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct')
    hf.add_argument('--out-root',type=Path,default=Path('/data/outputs/gpu96_acceptance'))
    hf.add_argument('--master-port',type=int,default=None)
    hf.add_argument('--timeout-sec',type=int,default=2400)
    sub.add_parser('hf-plan', help='print vetted HF acceptance entry + requirements; never starts GPU')
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.command == 'bench': return run_bench(a)
        if a.command == 'ratio': return ratio_check(a)
        if a.command == 'gmean': return run_gmean(a)
        if a.command == 'hf': return run_hf(a)
        if a.command == 'ratio-gpu': return run_ratio_gpu(a)
        if a.command == 'hf-plan': return hf_plan(a)
        raise AssertionError(a.command)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f'[gpu96] FAIL/BLOCKED: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

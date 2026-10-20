"""Sequential post-training comparison; never runs a second GPU job in parallel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.paired_control.methods import validate_pair


def verify_checkpoint(path):
    path = Path(path)
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    seen = {}
    for name in set(index['weight_map'].values()):
        file = path / name
        assert file.resolve().parent == path.resolve()
        with file.open('rb') as stream:
            n = struct.unpack('<Q', stream.read(8))[0]
            assert 0 < n < min(file.stat().st_size, 32 * 1024**2)
            header = json.loads(stream.read(n))
        for key, tensor in header.items():
            if key == '__metadata__':
                continue
            assert key not in seen and tensor['dtype'] not in ('BF16', 'F16', 'F64')
            assert 0 <= tensor['data_offsets'][0] <= tensor['data_offsets'][1] <= file.stat().st_size - n - 8
            seen[key] = name
    assert seen == index['weight_map'] and seen
    return dict(path=str(path), tensors=len(seen), floating_dtype='F32')


def gpu_idle(timeout=90):
    start = time.monotonic()
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
        if time.monotonic() - start > timeout:
            raise RuntimeError('GPU occupied; refuse overlapping job')
        time.sleep(2)


def worker(job):
    queue = Path(job['queue'])
    official = Path(job['official_run'])
    enhanced = Path(job['enhanced_run'])
    enhanced_root = Path(job['enhanced_root'])
    py = job['python']
    def state(value, **data):
        report = dict(state=value, **data)
        temporary = queue / 'status.tmp'
        temporary.write_text(json.dumps(report, indent=2));temporary.replace(queue / 'status.json')
        print('PAIR_STATE', json.dumps(report), flush=True)
    def run(command, log, cwd, *, timeout=14400):
        env = dict(os.environ, PYTHONPATH=str(cwd), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
                   AL_DYNAMO_CACHE_SIZE_LIMIT='0', QWEN3VL_PATH=job['qwen'])
        for key in ('LINGBOT_PRECISION_AUDIT', 'LINGBOTVLA_TRAINING_CONFIG', 'AL_VLM_ATTENTION'):
            env.pop(key, None)
        with log.open('x') as output:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            print('PAIR_CHILD', process.pid, str(log), flush=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                raise
        if code:
            raise RuntimeError((str(log), code))
    def evaluate(label, model, reference=None):
        directory = queue / label
        directory.mkdir(exist_ok=False)
        evaluation = dict(label=label, model=str(model), destination=str(directory), episodes=3,
            require_baseline=False, robotwin=str(ROOT.parent / 'RoboTwin-lingbot'), qwen=job['qwen'],
            sim_python=job['sim_python'])
        if reference:
            evaluation['reference_summary'] = str(reference)
        path = directory / 'job.json';path.write_text(json.dumps(evaluation, indent=2))
        gpu_idle()
        run([py, '-u', str(ROOT / 'experiment/official_control/evaluate.py'), '--job', str(path)],
            directory / 'pipeline.log', ROOT, timeout=4500)
        gpu_idle()
        return json.loads((directory / 'summary.json').read_text())
    try:
        state('waiting_official_training', parent_status=str(official / 'direct_training.status.json'))
        while True:
            parent = json.loads((official / 'direct_training.status.json').read_text())
            if parent['state'] == 'failed':
                raise RuntimeError(('official training failed', parent))
            if parent['state'] == 'complete':
                assert parent.get('training_complete')
                break
            time.sleep(10)
        assert hashlib.sha256((official / 'trainer_config.json').read_bytes()).hexdigest() == job['official_config_sha256']
        checkpoints = [verify_checkpoint(official / f'checkpoints/global_step_{step}/hf_ckpt') for step in (250, 500)]
        state('official_evaluation_3', verified_checkpoints=checkpoints)
        first = evaluate('official_500', official / 'checkpoints/global_step_500/hf_ckpt')
        state('enhanced_cpu_preflight', official_result=first)
        gpu_idle()
        # Prepare cache from raw PD drive-target data, never from model inference.
        run([py, str(enhanced_root / 'experiment/robotwin/prepare_target_change_windows.py'),
             '--al-config', str(enhanced / 'window_preparation.json'), '--dataset-root', job['dataset'],
             '--manifest', str(enhanced_root / 'refs/al_artifacts/al/task_splits_50/manifest.json'),
             '--tasks', 'adjust_bottle'], enhanced / 'window_prepare.log', enhanced_root, timeout=900)
        run([py, '-u', str(enhanced_root / 'experiment/paired_control/preflight.py'), str(enhanced)],
            enhanced / 'cpu_preflight.log', enhanced_root, timeout=900)
        assert json.loads((enhanced / 'cpu_report.json').read_text())['state'] == 'cpu_pass'
        import shutil
        assert shutil.disk_usage(enhanced).free >= 56 * 2**30, 'not enough space for both enhanced FP32 checkpoints'
        state('enhanced_gpu_probe', official_result=first)
        command = json.loads((enhanced / 'plan.json').read_text())['train_command']
        smoke = enhanced.with_name(enhanced.name + '_smoke')
        gpu_idle()
        run(command + ['--smoke-out', str(smoke)], enhanced / 'gpu_probe.log', enhanced_root, timeout=2400)
        assert json.loads((smoke / 'summary.json').read_text())['state'] == 'three_step_probe_complete'
        gpu_idle()
        state('enhanced_training', steps=500, save_steps=[250, 500], flags=job['flags'], official_result=first)
        run(['bash', str(enhanced / 'train.sh')], enhanced / 'train.log', enhanced_root)
        enhanced_checkpoints = [verify_checkpoint(enhanced / f'checkpoints/global_step_{step}/hf_ckpt') for step in (250, 500)]
        (enhanced / 'verified_checkpoints.json').write_text(json.dumps(enhanced_checkpoints, indent=2))
        state('enhanced_evaluation_3', verified_checkpoints=enhanced_checkpoints)
        second = evaluate('enhanced_500', enhanced / 'checkpoints/global_step_500/hf_ckpt', queue / 'official_500/summary.json')
        comparison = validate_pair(first, second)
        (queue / 'comparison.json').write_text(json.dumps(comparison, indent=2))
        state('complete', official=first, enhanced=second, comparison=comparison, shutdown=False)
        return 0
    except BaseException as error:
        state('failed', error=repr(error));traceback.print_exc();return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    job = json.loads(args.job.read_text())
    if args.worker:
        raise SystemExit(worker(job))
    print(json.dumps(job, indent=2))
    if args.execute:
        queue = Path(job['queue']);queue.mkdir(parents=True, exist_ok=False)
        with (queue / 'pipeline.log').open('x') as log:
            child = subprocess.Popen(['setsid', 'nohup', job['python'], '-u', str(Path(__file__).resolve()),
                         '--job', str(args.job), '--worker'], cwd=ROOT, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT)
        print('PAIRED_QUEUE_STARTED', child.pid, str(queue))


if __name__ == '__main__':
    main()

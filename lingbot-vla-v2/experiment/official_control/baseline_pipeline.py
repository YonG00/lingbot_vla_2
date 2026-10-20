"""已验收的adjust_bottle BF16基准；默认只打印计划，--execute后台执行。"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve()


def verify_checkpoint(path):
    path = Path(path)
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    seen = {}
    for shard in set(index['weight_map'].values()):
        file = path / shard
        assert file.resolve().parent == path.resolve()
        with file.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            assert 0 < size < min(file.stat().st_size, 32 * 1024**2)
            header = json.loads(stream.read(size))
        for key, tensor in header.items():
            if key == '__metadata__':
                continue
            assert key not in seen and tensor['dtype'] in ('BF16', 'I64', 'I32', 'BOOL')
            assert 0 <= tensor['data_offsets'][0] <= tensor['data_offsets'][1] <= file.stat().st_size - 8 - size
            seen[key] = shard
    assert seen and seen == index['weight_map']
    return dict(path=str(path), tensors=len(seen), floating_dtype='BF16')


def worker(job):
    out = Path(job['out'])
    launch = Path(job['launch'])
    def state(value, **extra):
        (launch / 'status.json').write_text(json.dumps(dict(state=value, **extra), indent=2))
        print('BASELINE_STATE', value, extra, flush=True)
    def run(command, log):
        env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', AL_DYNAMO_CACHE_SIZE_LIMIT='0',
            PYTHONUNBUFFERED='1', LOCAL_RANK='0', RANK='0', WORLD_SIZE='1',
            MASTER_ADDR='127.0.0.1', MASTER_PORT='29546')
        for key in ('LINGBOT_PRECISION_AUDIT', 'LINGBOTVLA_TRAINING_CONFIG', 'AL_VLM_ATTENTION'):
            env.pop(key, None)
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            print('BASELINE_CHILD', child.pid, str(log), flush=True)
            code = child.wait()
        if code:
            raise RuntimeError((str(log), code))
    try:
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip(), 'GPU occupied'
        helper = str(ROOT / 'experiment/official_control/bf16_trial.py')
        state('preparing')
        run([sys.executable, helper, 'prepare', '--out', str(out)], launch / 'prepare.log')
        state('cpu_preflight')
        run([sys.executable, helper, 'cpu', '--out', str(out)], out / 'cpu.log')
        state('gpu_probe')
        run([sys.executable, helper, 'smoke', '--out', str(out)], out / 'probe.log')
        probe = json.loads((out / 'probe/summary.json').read_text())
        assert probe['state'] == 'three_step_probe_complete'
        state('training', probe=probe)
        run(['bash', str(out / 'train.sh')], out / 'train.log')
        checkpoint = verify_checkpoint(out / 'checkpoints/global_step_200/hf_ckpt')
        (out / 'verified_checkpoint.json').write_text(json.dumps(checkpoint, indent=2))
        state('evaluation', checkpoint=checkpoint)
        destination = out / 'evaluation_200'
        destination.mkdir(exist_ok=False)
        evaluation = dict(label='official_bf16_200', model=checkpoint['path'], destination=str(destination),
            episodes=3, require_baseline=False, expected_seeds=[100001, 100002, 100005],
            robotwin=str(ROOT.parent / 'RoboTwin-lingbot'), sim_python=job['sim_python'],
            qwen=json.loads((out / 'trainer_config.json').read_text())['model']['tokenizer_path'])
        path = destination / 'job.json'
        path.write_text(json.dumps(evaluation, indent=2))
        run([sys.executable, str(ROOT / 'experiment/official_control/evaluate.py'), '--job', str(path)],
            destination / 'pipeline.log')
        state('complete', checkpoint=checkpoint,
            evaluation=json.loads((destination / 'summary.json').read_text()), shutdown=False)
        return 0
    except BaseException as error:
        state('failed', error=repr(error))
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--sim-python', default='/data/miniconda3/envs/RoboTwin/bin/python')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(json.loads(args.worker.read_text())))
    if args.out is None:
        parser.error('--out is required')
    out = args.out.resolve()
    job = dict(out=str(out), launch=str(out.with_name(out.name + '_pipeline')), sim_python=args.sim_python,
        source='global_step_9680', steps=200, micro=56, gas=1, save_steps=250,
        final_checkpoint=200, hf_export_dtype='bf16', evaluation_episodes=3, shutdown=False)
    print(json.dumps(job, indent=2))
    if args.execute:
        assert not out.exists(), 'refuse to overwrite a run'
        launch = Path(job['launch'])
        launch.mkdir(parents=True, exist_ok=False)
        path = launch / 'job.json'
        path.write_text(json.dumps(job, indent=2))
        with (launch / 'pipeline.log').open('x') as stream:
            child = subprocess.Popen(['setsid', 'nohup', sys.executable, '-u', str(HERE), '--worker', str(path)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        print('BASELINE_STARTED', child.pid, str(launch))


if __name__ == '__main__':
    main()

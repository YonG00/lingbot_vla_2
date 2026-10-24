"""adjust_bottle 对照评测入口；默认只打印计划，--execute才后台执行。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve()
sys.path.insert(0,str(ROOT))
from experiment.official_control.recording import video_count,install_recording_limit


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=20)


def execute(job, shared=None):
    destination = Path(job['destination'])
    episodes = int(job.get('episodes', 10))
    expected_videos=video_count(episodes,job.get('video_episodes'))
    if job.get('render_sync','eager') not in ['eager','on_demand']:raise ValueError('invalid render_sync')
    task = job.get('task', 'adjust_bottle')
    assert re.fullmatch(r'[a-z][a-z0-9_]*', task), 'invalid task name'
    assert episodes > 0
    status = destination / 'status.json'
    server = simulation = None
    def state(value, **extra):
        status.write_text(json.dumps(dict(state=value, **extra), indent=2))
    try:
        if shared is None:
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip(), 'GPU busy'
        else:
            assert Path(shared['model']).resolve() == Path(job['model']).resolve(), 'shared model differs'
            assert shared['process'].poll() is None, 'shared inference server exited'
        robo = Path(job['robotwin'])
        client = robo / 'script/eval_policy_client_lingbotvla.py'
        client_hash = hashlib.sha256(client.read_bytes()).hexdigest()
        if shared is not None:
            port = shared['port']
        else:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
        env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='0',
                   QWEN3VL_PATH=job['qwen'], LINGBOT_INFER_NOISE_SEED='0',
                   HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0', PYTHONUNBUFFERED='1',
                   LINGBOT_DEPLOY_FAST_LOAD='0',
                   LINGBOT_EVAL_PRECISION_AUDIT=str(destination / 'precision_audit.json'))
        for key in ('LINGBOTVLA_TRAINING_CONFIG', 'AL_VLM_ATTENTION', 'LINGBOT_PRECISION_AUDIT'):
            env.pop(key, None)
        if shared is not None:
            env['LINGBOT_EVAL_PRECISION_AUDIT'] = str(shared['audit'])
        server_command = [sys.executable, '-u', str(ROOT / 'experiment/robotwin/precision_eval_server.py'),
                          '--model_path', job['model'], '--use_length', '50', '--chunk_ret', 'True',
                          '--use_bf16', 'False', '--use_fp32', 'True', '--use_compile', 'False', '--port', str(port)]
        sim_command = [job['sim_python'], '-u', str(HERE), '--sim-client', str(robo),
                       '--config', 'policy/ACT/deploy_policy.yml', '--overrides',
                       '--task_name', task, '--task_config', 'demo_clean',
                       '--train_config_name', 'official_control', '--seed', '0', '--policy_name', 'ACT',
                       '--port', str(port), '--robo_name', 'robotwin', '--video_fps', '10',
                       '--eval_video_log', 'False', '--eval_global_view_log', 'True', '--global_view_fps', '2',
                       '--output_dir', str(destination / 'results'), '--test_num', str(episodes)]
        temporal=job.get('temporal_ensemble')
        if temporal is not None:
            from experiment.official_control.temporal_ensemble import TemporalEnsemble
            TemporalEnsemble(**temporal)
            env['LINGBOT_TEMPORAL_ENSEMBLE']=json.dumps(temporal)
            env['LINGBOT_TEMPORAL_ENSEMBLE_AUDIT']=str(destination/'temporal_ensemble_audit.json')
        else:
            env.pop('LINGBOT_TEMPORAL_ENSEMBLE',None)
            env.pop('LINGBOT_TEMPORAL_ENSEMBLE_AUDIT',None)
        if job.get('video_episodes') is not None:
            sim_command.extend(['--global_view_episodes',str(job['video_episodes'])])
        if 'scene_seed' in job:
            if episodes!=1 or job.get('expected_seeds')!=[job['scene_seed']]:raise ValueError('episode seed contract')
            env['LINGBOT_EPISODE_SCENE_SEED']=str(job['scene_seed'])
            env['LINGBOT_DESCRIPTION_EPISODE_COUNT']=str(job['description_episode_count'])
            env['LINGBOT_EPISODE_INDEX']=str(job['episode_index'])
        if 'render_sync' in job:sim_command.extend(['--render_sync',job['render_sync']])
        (destination / 'commands.json').write_text(json.dumps(dict(server=server_command, simulation=sim_command), indent=2))
        state('starting')
        with (destination / 'server.log').open('x') as slog, (destination / 'sim.log').open('x') as elog:
            if shared is None:
                server = subprocess.Popen(server_command, cwd=ROOT, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
            else:
                slog.write('Shared inference server pid=' + str(shared['process'].pid) + '\n')
            started = time.monotonic()
            while True:
                assert (server if shared is None else shared['process']).poll() is None, 'inference server exited'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=1) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                if time.monotonic() - started > 600:
                    raise TimeoutError('inference startup')
                time.sleep(2)
            site = subprocess.check_output([job['sim_python'], '-c', 'import site; print(site.getsitepackages()[0])'], text=True).strip()
            simenv = dict(env, PYTHONPATH=os.pathsep.join([site, str(robo), str(robo / 'script')]),
                          PYTHONNOUSERSITE='1', PYTHONWARNINGS='ignore::UserWarning')
            if 'render_sync' in job:simenv['LINGBOT_RENDER_SYNC_AUDIT']=str(destination/'render_sync_audit.json')
            if job.get('trace_policy'):simenv['LINGBOT_POLICY_TRACE']=str(destination/'policy_trace.jsonl')
            state('evaluating')
            simulation = subprocess.Popen(sim_command, cwd=robo, env=simenv, stdout=elog, stderr=subprocess.STDOUT, start_new_session=True)
            assert simulation.wait(timeout=3600) == 0, 'simulation failed'
        clean = re.sub(r'\x1b\[[0-9;]*m', '', (destination / 'sim.log').read_text(errors='replace'))
        rates = re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)', clean)
        assert len(rates) == episodes and int(rates[-1][1]) == episodes, 'incomplete episodes'
        assert 'OIDN 2.5.1 CUDA=' in clean and 'OIDN Error' not in clean, 'renderer check failed'
        seeds = [int(row[2]) for row in rates]
        assert len(set(seeds)) == episodes
        if job.get('expected_seeds') is not None:
            assert seeds == job['expected_seeds'], 'scene seeds differ from recorded baseline'
        baseline = destination.parent / '9680/summary.json'
        if job.get('reference_summary'):
            assert seeds == json.loads(Path(job['reference_summary']).read_text())['seeds'], 'scene seeds differ between paired models'
        elif job.get('require_baseline', True) and job['label'] != '9680':
            assert seeds == json.loads(baseline.read_text())['seeds'], 'scene seeds differ from baseline'
        results = destination / 'results' / task
        videos = list(results.glob('episode*_global.mp4'))
        assert {p.name for p in videos}=={f'episode{i}_global.mp4' for i in range(expected_videos)} and all(p.stat().st_size for p in videos), 'missing or unexpected video'
        rate = float((results / '_result.txt').read_text().strip().splitlines()[-1])
        assert abs(rate - int(rates[-1][0]) / episodes) < 1e-6
        if shared is not None:
            import shutil
            shutil.copyfile(shared['audit'], destination / 'precision_audit.json')
        precision = json.loads((destination / 'precision_audit.json').read_text())
        assert set(precision['parameter_numel_by_dtype']) == {'torch.float32'} and precision['forward_samples']
        assert hashlib.sha256(client.read_bytes()).hexdigest() == client_hash
        summary = dict(state='complete', task=task, label=job['label'], model=job['model'],
                       successes=int(rates[-1][0]), episodes=episodes, success_rate=rate, seeds=seeds,
                       oidn_errors=0, client_sha256=client_hash, dtype='float32', action_chunk=50,recorded_episodes=expected_videos)
        if temporal is not None:
            temporal_audit=json.loads((destination/'temporal_ensemble_audit.json').read_text())
            assert temporal_audit['enabled'] and temporal_audit['ensemble_horizon']==temporal.get('horizon',8) and temporal_audit['decay']==temporal.get('decay',0.1)
            assert temporal_audit['replan_every_actions']==1 and temporal_audit['executed_actions_per_response']==1 and temporal_audit['source_chunk_length']==50 and temporal_audit['requests']>0
            summary['temporal_ensemble']=temporal_audit
        if 'job_id' in job:summary.update(job_id=job['job_id'],episode_index=job['episode_index'],scene_seed=job['scene_seed'])
        if shared is not None:
            summary.update(server_pid=shared['process'].pid, shared_server=True)
        (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
        state('complete', summary=summary)
        print(json.dumps(summary), flush=True)
        return 0
    except BaseException as error:
        state('failed', error=repr(error))
        raise
    finally:
        stop(simulation)
        stop(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', choices=['9680', '250', '500'], required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads((args.out / 'plan.json').read_text())
    assert json.loads((args.out / 'cpu_report.json').read_text())['state'] == 'cpu_pass'
    model = Path(plan['source_model']) if args.label == '9680' else args.out / f'checkpoints/global_step_{args.label}/hf_ckpt'
    assert (model / 'model.safetensors.index.json').is_file()
    directory = args.out / 'evaluation' / args.label
    job = dict(label=args.label, model=str(model), destination=str(directory),
               robotwin=str(ROOT.parent / 'RoboTwin-lingbot'),
               qwen=json.loads((args.out / 'trainer_config.json').read_text())['model']['tokenizer_path'],
               sim_python='/data/miniconda3/envs/RoboTwin/bin/python')
    print(json.dumps(job, indent=2))
    if args.execute:
        if args.label != '9680':
            assert (directory.parent / '9680/summary.json').is_file(), 'baseline evaluation required first'
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / 'job.json'
        path.write_text(json.dumps(job, indent=2))
        with (directory / 'pipeline.log').open('x') as log:
            process = subprocess.Popen(['setsid', 'nohup', sys.executable, '-u', str(HERE), '--job', str(path)],
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print('EVALUATION_STARTED', process.pid, str(directory))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--sim-client':
        import random
        import runpy
        robo = Path(sys.argv[2])
        os.chdir(robo)
        sys.path[:0] = [str(robo), str(robo / 'script')]
        random.seed(0)
        import numpy as np
        np.random.seed(0)
        import importlib
        task = sys.argv[sys.argv.index('--task_name') + 1]
        importlib.import_module('envs.' + task)
        if '--global_view_episodes' in sys.argv:
            limit=int(sys.argv[sys.argv.index('--global_view_episodes')+1])
            from envs._base_task import Base_Task
            install_recording_limit(Base_Task,limit)
            print('EVAL_RECORDING_POLICY first_episodes='+str(limit),flush=True)
        if '--render_sync' in sys.argv:
            from envs._base_task import Base_Task
            from experiment.official_control.render_sync import install_render_sync
            mode=sys.argv[sys.argv.index('--render_sync')+1]
            install_render_sync(Base_Task,mode,os.environ['LINGBOT_RENDER_SYNC_AUDIT'])
            print('EVAL_RENDER_SYNC '+mode,flush=True)
        if os.environ.get('LINGBOT_POLICY_TRACE'):
            from script.deploy.websocket_client_policy import WebsocketClientPolicy
            from experiment.official_control.render_sync import install_policy_trace
            install_policy_trace(WebsocketClientPolicy,os.environ['LINGBOT_POLICY_TRACE'])
        if os.environ.get('LINGBOT_TEMPORAL_ENSEMBLE'):
            from script.deploy.websocket_client_policy import WebsocketClientPolicy
            from experiment.official_control.temporal_ensemble import install_client
            install_client(WebsocketClientPolicy,**json.loads(os.environ['LINGBOT_TEMPORAL_ENSEMBLE']),audit_path=os.environ['LINGBOT_TEMPORAL_ENSEMBLE_AUDIT'])
            print('EVAL_TEMPORAL_ENSEMBLE '+os.environ['LINGBOT_TEMPORAL_ENSEMBLE'],flush=True)
        sys.argv = [str(robo / 'script/eval_policy_client_lingbotvla.py')] + sys.argv[3:]
        if os.environ.get('LINGBOT_EPISODE_SCENE_SEED') is None:
            runpy.run_path(sys.argv[0], run_name='__main__')
        else:
            from experiment.official_control.episode_parallel import compile_episode_client
            code=compile_episode_client(Path(sys.argv[0]).read_text(),sys.argv[0],int(os.environ['LINGBOT_EPISODE_SCENE_SEED']),int(os.environ['LINGBOT_DESCRIPTION_EPISODE_COUNT']),int(os.environ['LINGBOT_EPISODE_INDEX']))
            exec(code,dict(__name__='__main__',__file__=sys.argv[0]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--job':
        raise SystemExit(execute(json.loads(Path(sys.argv[2]).read_text())))
    else:
        main()

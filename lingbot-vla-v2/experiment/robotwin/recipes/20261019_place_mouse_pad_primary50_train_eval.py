#!/usr/bin/env python3
"""纯主任务50条续练；默认只预览，--execute后台训练→3回合评测→可选关机。"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import runpy
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('primary50_source', HERE.with_name(
    '20261018_place_mouse_pad_all50_9680_ee_lnfm.py'))
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
recipe = source.recipe
continuation = recipe.continuation
PACKAGE = recipe.ROOT


def validate_plan(args, job, plan):
    assert args.steps == 500 and args.hf_save_steps == 250, '此入口固定500步、250/500两份存档'
    assert args.micro > 0 and args.gas > 0 and args.gradient_checkpointing
    if not getattr(args, 'fp32_master', False):
        assert args.micro * args.gas == 56
    assert plan['training_config']['train']['enable_mixed_precision'] == bool(getattr(args, 'fp32_master', False))
    assert plan['training_config']['train']['enable_fp32'] is False
    assert args.train_all_episodes and not args.auxiliary_manifest and not plan.get('auxiliary')
    assert plan['counts']['place_mouse_pad'] == {'train': 50, 'val': 0}
    assert args.ee_position_weight == 10 and args.ee_orientation_weight == .8 and args.ee_gripper_weight == 1.5
    assert args.alignment_loss_scale == 1
    align = plan['training_config']['train']['align_params']
    assert [align['depth_loss_weight'], align['future_depth_loss_weight'],
            align['video']['future_video_loss_weight']] == [.004, .004, .004]
    assert Path(job['source_model']).resolve() == (source.SOURCE_RUN / 'checkpoints/global_step_500/hf_ckpt').resolve()
    assert args.closed_loop_episodes == 3


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=20)


def validate_evaluation(directory, episodes, client_sha):
    log = (directory / 'sim.log').read_text(errors='replace')
    clean = re.sub(r'\x1b\[[0-9;]*m', '', log)
    rates = re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)', clean)
    if len(rates) != episodes or int(rates[-1][1]) != episodes:
        raise RuntimeError('闭环评测未完成指定回合数')
    if 'OIDN 2.5.1 CUDA=' not in log or 'OIDN Error' in log:
        raise RuntimeError('渲染兼容检查失败，不自动关机')
    result = directory / 'results/place_mouse_pad/_result.txt'
    rate = float(result.read_text().strip().splitlines()[-1])
    expected = int(rates[-1][0]) / episodes
    if not math.isfinite(rate) or abs(rate - expected) > 1e-6:
        raise RuntimeError('评测结果文件与完成日志不一致')
    videos = sorted((directory / 'results/place_mouse_pad').glob('episode*_global.mp4'))
    if len(videos) != episodes or any(p.stat().st_size == 0 for p in videos):
        raise RuntimeError('闭环视频不完整')
    return dict(episodes=episodes, successes=int(rates[-1][0]), success_rate=rate,
                seeds=[int(r[2]) for r in rates], client_sha256=client_sha,
                oidn_errors=0, videos=[str(p) for p in videos])


def evaluation_server_command(model, port, precision):
    fp32 = precision == 'fp32'
    entry = ([str(PACKAGE / 'experiment/robotwin/precision_eval_server.py')] if fp32
             else ['-m', 'deploy.lingbot_vla_v2_policy'])
    return [sys.executable, '-u', *entry, '--model_path', str(model),
            '--use_length', '50', '--chunk_ret', 'True', '--use_bf16', str(not fp32),
            '--use_fp32', str(fp32), '--use_compile', 'False', '--port', str(port)]


def evaluate(config, model):
    directory = Path(config['eval_dir'])
    directory.mkdir(parents=True, exist_ok=False)
    robo = Path(config['robotwin_root'])
    client = robo / 'script/eval_policy_client_lingbotvla.py'
    original_sha = hashlib.sha256(client.read_bytes()).hexdigest()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    simpy = config['sim_python']
    site = subprocess.check_output([simpy, '-c', 'import site;print(site.getsitepackages()[0])'], text=True).strip()
    env = dict(os.environ, PYTHONPATH=str(PACKAGE), CUDA_VISIBLE_DEVICES='0',
               QWEN3VL_PATH=config['qwen3vl'], LINGBOT_INFER_NOISE_SEED='0',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0', PYTHONUNBUFFERED='1')
    env.pop('LINGBOTVLA_TRAINING_CONFIG', None)
    fp32_eval = config.get('eval_precision', 'bf16') == 'fp32'
    env['LINGBOT_DEPLOY_FAST_LOAD'] = '0'
    env['LINGBOT_EVAL_PRECISION_AUDIT'] = str(directory / 'precision_audit.json')
    server_cmd = evaluation_server_command(model, port, config.get('eval_precision', 'bf16'))
    sim_cmd = [simpy, '-u', str(HERE), '--sim-client', str(robo),
               '--config', 'policy/ACT/deploy_policy.yml', '--overrides',
               '--task_name', 'place_mouse_pad', '--task_config', 'demo_clean',
               '--train_config_name', 'primary50_control', '--seed', '0', '--policy_name', 'ACT',
               '--port', str(port), '--robo_name', 'robotwin', '--video_fps', '10',
               '--eval_video_log', 'False', '--eval_global_view_log', 'True', '--global_view_fps', '2',
               '--output_dir', str(directory / 'results'), '--test_num', str(config['episodes'])]
    continuation.write_json(directory / 'commands.json', dict(server=server_cmd, sim=sim_cmd))
    server = sim = None
    try:
        with (directory / 'server.log').open('x') as slog, (directory / 'sim.log').open('x') as elog:
            server = subprocess.Popen(server_cmd, cwd=PACKAGE, env=env, stdout=slog,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            start = time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError('推理服务启动失败')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=1) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                if time.monotonic() - start > 420:
                    raise TimeoutError('推理服务启动超时')
                time.sleep(2)
            simenv = dict(env, PYTHONPATH=os.pathsep.join([site, str(robo), str(robo / 'script')]),
                          PYTHONNOUSERSITE='1', PYTHONWARNINGS='ignore::UserWarning',
                          XLA_PYTHON_CLIENT_MEM_FRACTION='0.9')
            sim = subprocess.Popen(sim_cmd, cwd=robo, env=simenv, stdout=elog,
                                   stderr=subprocess.STDOUT, start_new_session=True)
            if sim.wait(timeout=1200) != 0:
                raise RuntimeError('闭环仿真异常退出')
        if hashlib.sha256(client.read_bytes()).hexdigest() != original_sha:
            raise RuntimeError('评测客户端被更改')
        result = validate_evaluation(directory, config['episodes'], original_sha)
        if fp32_eval:
            audit = json.loads((directory / 'precision_audit.json').read_text())
            if set(audit['parameter_numel_by_dtype']) != {'torch.float32'} or not audit['forward_samples']:
                raise RuntimeError('FP32评测实际dtype审计失败')
            result['precision_audit'] = audit
        continuation.write_json(directory / 'summary.json', result)
        return result
    finally:
        stop(sim)
        stop(server)


def run_pipeline(config):
    state = dict(state='training', training_complete=False, evaluation_complete=False,
                 shutdown=config['shutdown'])
    def save():
        continuation.write_json(config['pipeline_status'], state)
    save()
    try:
        # Inner worker must never shut down before the closed-loop evaluation.
        if config['training_job']['shutdown']:
            raise RuntimeError('训练子任务不得直接关机')
        if config.get('fp32_master'):
            os.environ['LINGBOT_PRECISION_AUDIT'] = str(Path(config['training_job']['output_dir']) / 'precision_audit.json')
        if continuation.run_job(config['training_job']) != 0:
            raise RuntimeError('训练或最终存档校验失败')
        continuation.verify_final(dict(config['training_job'], steps=250))
        model = continuation.verify_final(config['training_job'])
        state.update(state='evaluating', training_complete=True, final_hf=str(model))
        save()
        result = evaluate(config, model)
        state.update(state='complete', evaluation_complete=True, evaluation=result)
        save()
        if config['shutdown']:
            state['state'] = 'shutdown_requested'
            save()
            subprocess.run(['bash', '/usr/bin/shutdown'], check=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        state.update(state='failed', error=repr(exc))
        save()
        print(f'流程失败，保留机器：{exc!r}', flush=True)
        return 1


def main(argv=None):
    parser = recipe.parser()
    parser.set_defaults(model=source.SOURCE_RUN / 'checkpoints/global_step_500/hf_ckpt',
                        train_config=source.SOURCE_RUN / 'lingbotvla_cli.yaml',
                        train_all_episodes=True, ee_position_weight=10., micro=56,
                        gradient_checkpointing=True)
    parser.add_argument('--closed-loop-episodes', type=int, default=3)
    parser.add_argument('--eval-precision', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--sim-python', default='/data/miniconda3/envs/RoboTwin/bin/python')
    parser.add_argument('--robotwin-root', type=Path, default=PACKAGE.parent / 'RoboTwin-lingbot')
    args = parser.parse_args(argv)
    args.run_name = args.run_name or f'al_mouse_pad_primary50_g56_from_mouse500_add500_{datetime.now():%Y%m%d_%H%M%S}'
    job, plan, snapshot = recipe.build_job(args)
    validate_plan(args, job, plan)
    job['shutdown'] = False
    config = dict(training_job=job, shutdown=args.shutdown, episodes=args.closed_loop_episodes,
                  fp32_master=args.fp32_master, eval_precision=args.eval_precision,
                  robotwin_root=str(args.robotwin_root.resolve()), sim_python=args.sim_python,
                  qwen3vl=str(args.qwen3vl), eval_dir=str(Path(job['output_dir']) / 'closed_loop_eval'),
                  pipeline_status=str(args.runtime_dir / f'{args.run_name}.pipeline.status.json'))
    path = args.runtime_dir / f'{args.run_name}.pipeline.json'
    print(json.dumps(dict(config=config, split=plan['counts'], auxiliary=plan.get('auxiliary'),
                          checkpoint_count=2, save_steps=250, micro=args.micro, gas=args.gas,
                          alignment=plan['training_config']['train']['align_params']), indent=2))
    if not args.execute:
        return 0
    continuation.check_machine(job, plan)
    # Budget two HFs via the existing guard, plus 2 GiB for eval/logs.
    import shutil
    weight_bytes = sum(row['size'] for row in plan['checkpoint_inventory'])
    required = int(2 * weight_bytes * 1.1) + 3 * 1024**3
    free = shutil.disk_usage(Path(job['output_dir']).parent).free
    if free < required:
        raise RuntimeError(f'需要{required/1024**3:.2f}GiB，现有{free/1024**3:.2f}GiB；不自动删除')
    al_snapshot = Path(job['job_file']).with_suffix('.al.json')
    artifacts = [snapshot, al_snapshot, Path(job['job_file']), path,
                 Path(job['log']), Path(job['status_file']), Path(config['pipeline_status'])]
    if any(p.exists() for p in artifacts):
        raise FileExistsError('启动产物已存在，禁止覆盖')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    continuation.write_json(snapshot, plan['training_config'])
    continuation.write_json(al_snapshot, plan['config'])
    job['command'][job['command'].index('--al-config') + 1] = str(al_snapshot)
    continuation.write_json(job['job_file'], job)
    continuation.write_json(path, config)
    with Path(job['log']).open('x') as log:
        child = subprocess.Popen(['setsid', 'nohup', sys.executable, str(HERE), '--job', str(path)],
                                 cwd=PACKAGE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    print(f'后台流程PID={child.pid}；日志={job["log"]}；容量预算={required/1024**3:.2f}GiB', flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--sim-client':
        # Avoid importing the training dependency tree into the simulation env.
        robo = Path(sys.argv[2]); os.chdir(robo)
        sys.path[:0] = [str(robo), str(robo / 'script')]
        random.seed(0)
        import numpy as np
        np.random.seed(0)
        from envs.place_mouse_pad import place_mouse_pad  # same import order as paired diagnostics
        sys.argv = [str(robo / 'script/eval_policy_client_lingbotvla.py')] + sys.argv[3:]
        runpy.run_path(sys.argv[0], run_name='__main__')
    elif len(sys.argv) == 3 and sys.argv[1] == '--job':
        raise SystemExit(run_pipeline(json.loads(Path(sys.argv[2]).read_text())))
    else:
        raise SystemExit(main())

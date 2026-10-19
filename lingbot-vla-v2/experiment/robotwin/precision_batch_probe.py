"""独立3步精度/显存探针；默认CPU预览，--execute才后台运行，不存档、不关机。"""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location('fp32_probe_recipe', HERE.parent / 'recipes/20261018_place_mouse_pad_9680_ee_lnfm.py')
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)
PACKAGE = recipe.ROOT
SOURCE = Path('/data/outputs/al_mouse_pad_9680_ee_lnfm_add500_20261018_150402')


def probe_command(command):
    cmd = list(command)
    for key, value in {'--train.dcp_save_mode': 'always', '--train.smoke_no_checkpoint': 'true',
                       '--train.disk_guard': 'false'}.items():
        if key in cmd:
            cmd[cmd.index(key)+1] = value
        else:
            cmd += [key, value]
    return cmd


def worker(job):
    env = dict(os.environ, PYTHONPATH=str(PACKAGE), LINGBOT_PRECISION_AUDIT=job['audit'])
    rc = subprocess.run(job['command'], cwd=PACKAGE, env=env, check=False).returncode
    report = dict(returncode=rc, micro=job['micro'], gas=job['gas'], gbs=job['micro']*job['gas'])
    audit = Path(job['audit'])
    if audit.exists():
        report['precision_audit'] = json.loads(audit.read_text())
    log = Path(job['log']).read_text(errors='replace')
    import re
    completed = [int(x) for x in re.findall(r'Step:\s*(\d+)/3', log)]
    report['last_step'] = max(completed, default=0)
    report['passed'] = rc == 0 and report['last_step'] == 3 and audit.exists()
    recipe.continuation.write_json(job['result'], report)
    return 0 if report['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--micro', type=int, default=32)
    parser.add_argument('--gas', type=int, default=1)
    parser.add_argument('--run-name')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    name = args.run_name or f'mouse_fp32master_probe_m{args.micro}_g{args.gas}_{datetime.now():%Y%m%d_%H%M%S}'
    parsed = recipe.parser().parse_args(['--model', str(SOURCE/'checkpoints/global_step_500/hf_ckpt'),
        '--train-config', str(SOURCE/'lingbotvla_cli.yaml'), '--train-all-episodes', '--ee-position-weight', '10',
        '--fp32-master', '--gradient-checkpointing', '--micro', str(args.micro), '--gas', str(args.gas),
        '--steps', '3', '--run-name', name])
    job, plan, snapshot = recipe.build_job(parsed)
    output = Path(job['output_dir'])
    probe = dict(command=probe_command(plan['command']), micro=args.micro, gas=args.gas,
                 audit=str(output/'precision_audit.json'), result=str(output/'probe_result.json'),
                 log=job['log'])
    print(json.dumps(probe, indent=2))
    if not args.execute:
        return 0
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:
        raise RuntimeError('GPU非空闲，不启动探针')
    if output.exists() or snapshot.exists() or Path(job['log']).exists():
        raise FileExistsError('探针产物已存在')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    recipe.continuation.write_json(snapshot, plan['training_config'])
    prepare = list(job['command'][2:])
    prepare[prepare.index('--execute')] = '--prepare-only'
    recipe.launcher.main(prepare)
    path = output / 'probe_job.json'
    recipe.continuation.write_json(path, probe)
    with Path(job['log']).open('x') as log:
        child = subprocess.Popen(['setsid','nohup',sys.executable,str(HERE),'--job',str(path)],
            cwd=PACKAGE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    print(f'后台探针PID={child.pid}；结果={probe["result"]}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--job':
        raise SystemExit(worker(json.loads(Path(sys.argv[2]).read_text())))
    raise SystemExit(main())

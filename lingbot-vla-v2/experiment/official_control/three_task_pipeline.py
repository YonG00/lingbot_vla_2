"""三任务400步，保留200/400 BF16 HF，只测最终400的8任务×3回合。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment.official_control.three_task import EVAL_TASKS
from experiment.official_control.baseline_pipeline import verify_checkpoint


def worker(out):
    launch=out.with_name(out.name+'_pipeline')
    def state(value,**data):
        (launch/'status.json').write_text(json.dumps(dict(state=value,**data),indent=2))
        print('THREE_TASK_STATE',value,data,flush=True)
    def run(command,log):
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',
            HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',AL_DYNAMO_CACHE_SIZE_LIMIT='0',
            PYTHONUNBUFFERED='1',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='29548')
        for k in ['LINGBOT_PRECISION_AUDIT','LINGBOTVLA_TRAINING_CONFIG','AL_VLM_ATTENTION']:env.pop(k,None)
        with log.open('x') as f:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            print('CHILD',child.pid,str(log),flush=True)
            code=child.wait()
        if code:raise RuntimeError((str(log),code))
    try:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'GPU busy'
        robo=ROOT.parent/'RoboTwin-lingbot'
        assert all((robo/'envs'/f'{task}.py').is_file() for task in EVAL_TASKS)
        helper=str(ROOT/'experiment/official_control/three_task.py')
        state('prepare')
        run([sys.executable,helper,'prepare','--out',str(out)],launch/'prepare.log')
        logs=Path('/root/tf-logs')
        if logs.is_dir():
            link=logs/out.name
            assert not link.exists() and not link.is_symlink(), 'TensorBoard run link already exists'
            link.symlink_to(out/'runs',target_is_directory=True)
        state('cpu_preflight')
        run([sys.executable,helper,'cpu','--out',str(out)],out/'cpu.log')
        state('gpu_probe')
        run([sys.executable,helper,'smoke','--out',str(out)],out/'probe.log')
        probe=json.loads((out/'probe/summary.json').read_text())
        assert probe['state']=='three_step_probe_complete'
        state('training',probe=probe,steps=400,hf_steps=[200,400])
        run(['bash',str(out/'train.sh')],out/'train.log')
        verified=[verify_checkpoint(out/f'checkpoints/global_step_{step}/hf_ckpt') for step in [200,400]]
        (out/'verified_checkpoints.json').write_text(json.dumps(verified,indent=2))
        results=[]
        for task in EVAL_TASKS:
            state('evaluation',task=task,completed=results)
            dest=out/'evaluation_400'/task;dest.mkdir(parents=True,exist_ok=False)
            cfg=json.loads((out/'trainer_config.json').read_text())
            job=dict(task=task,label='three_task_400',model=verified[-1]['path'],destination=str(dest),
                episodes=3,require_baseline=False,robotwin=str(robo),qwen=cfg['model']['tokenizer_path'],
                sim_python='/data/miniconda3/envs/RoboTwin/bin/python')
            if task=='adjust_bottle':job['expected_seeds']=[100001,100002,100005]
            path=dest/'job.json';path.write_text(json.dumps(job,indent=2))
            run([sys.executable,str(ROOT/'experiment/official_control/evaluate.py'),'--job',str(path)],dest/'pipeline.log')
            results.append(json.loads((dest/'summary.json').read_text()))
            (out/'evaluation_400/summary.json').write_text(json.dumps(results,indent=2))
        state('complete',verified_checkpoints=verified,evaluation=results,shutdown=False)
    except BaseException as error:
        state('failed',error=repr(error));traceback.print_exc();return 1
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=a.out.resolve()
    if a.worker:raise SystemExit(worker(out))
    print(json.dumps(dict(out=str(out),steps=400,hf_steps=[200,400],evaluate_step=400,tasks=EVAL_TASKS,episodes=3),indent=2))
    if a.execute:
        assert not out.exists()
        launch=out.with_name(out.name+'_pipeline');launch.mkdir(parents=True,exist_ok=False)
        with (launch/'pipeline.log').open('x') as f:
            child=subprocess.Popen(['setsid','nohup',sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(out),'--worker'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        print('STARTED',child.pid,str(launch))

if __name__=='__main__':main()

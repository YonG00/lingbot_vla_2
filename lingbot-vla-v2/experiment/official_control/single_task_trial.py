"""先执行单任务续训；复用现有BF16探针与闭环，不引入新调度器。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def prepare(out,source,task):
    from experiment.official_control.baseline_pipeline import verify_checkpoint
    from experiment.official_control.prepare_adjust_bottle import validate_identity
    import shutil
    validate_identity()
    checkpoint=source/'checkpoints/global_step_200/hf_ckpt'
    verify_checkpoint(checkpoint)
    assert shutil.disk_usage(out.parent).free>17*2**30
    cfg=json.loads((source/'trainer_config.json').read_text())
    entry=json.loads((ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())['tasks'][task]
    ids=sorted(entry['train_ids']+entry['val_ids']);assert len(ids)==len(set(ids))==50
    cfg['model'].update(model_path=str(checkpoint),config_path=str(checkpoint/'config.json'))
    cfg['data'].update(train_path=str(out/'datasets.txt'),episode_ids_file=str(out/'train_ids.json'),robot_config_root=str(ROOT/'configs/robot_configs'))
    cfg['train'].update(output_dir=str(out),max_steps=200,save_steps=200,enable_resume=False,load_checkpoint_path=None,step_offset=0,reset_step_counter=True)
    t=cfg['train']
    assert t['micro_batch_size']==t['global_batch_size']==56 and t['gradient_accumulation_steps']==1
    assert not any(t[k] for k in ['auto_learning','ee_fk_loss_enabled','low_noise_fm_enabled','enable_fp32','enable_mixed_precision','action_fp32'])
    assert t['hf_export_dtype']=='bf16'
    out.mkdir(exist_ok=False)
    for name,value in [('trainer_config.json',cfg),('train_ids.json',ids),('plan.json',dict(task=task,source_model=str(checkpoint),steps=200,save_steps=[200],expected_frames=entry['train_frames']+entry['val_frames']))]:
        (out/name).write_text(json.dumps(value,indent=2))
    (out/'datasets.txt').write_text((source/'datasets.txt').read_text())
    command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port=29551','tasks/vla/train_lingbotvla.py',str(out/'trainer_config.json')]
    (out/'train.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join(command)+'\n')


def worker(args):
    out=args.out;launch=out.with_name(out.name+'_pipeline')
    def state(value,**extra):
        (launch/'status.json').write_text(json.dumps(dict(state=value,**extra),indent=2));print(value,extra,flush=True)
    def run(command,log):
        cfg=json.loads((out/'trainer_config.json').read_text()) if out.exists() else None
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            AL_DYNAMO_CACHE_SIZE_LIMIT='0',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='29550',PYTHONUNBUFFERED='1')
        if cfg:env['QWEN3VL_PATH']=cfg['model']['tokenizer_path']
        for key in ['LINGBOT_PRECISION_AUDIT','LINGBOTVLA_TRAINING_CONFIG','AL_VLM_ATTENTION']:env.pop(key,None)
        with log.open('x') as f:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            print('CHILD',child.pid,str(log),flush=True);code=child.wait()
        if code:raise RuntimeError((str(log),code))
    try:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'GPU busy'
        state('prepare');prepare(out,args.source,args.task)
        logs=Path('/root/tf-logs')
        if logs.is_dir():
            link=logs/out.name;assert not link.exists() and not link.is_symlink();link.symlink_to(out/'runs',target_is_directory=True)
        here=str(Path(__file__).resolve())
        state('cpu_preflight');run([sys.executable,here,'--out',str(out),'--cpu'],out/'cpu.log')
        state('gpu_probe');run([sys.executable,str(ROOT/'experiment/official_control/bf16_trial.py'),'smoke','--out',str(out)],out/'probe.log')
        assert json.loads((out/'probe/summary.json').read_text())['state']=='three_step_probe_complete'
        state('training',steps=200);run(['bash',str(out/'train.sh')],out/'train.log')
        from experiment.official_control.baseline_pipeline import verify_checkpoint
        checked=verify_checkpoint(out/'checkpoints/global_step_200/hf_ckpt')
        (out/'verified_checkpoint.json').write_text(json.dumps(checked,indent=2))
        state('evaluation',checkpoint=checked)
        dest=out/'evaluation_200';dest.mkdir(exist_ok=False)
        cfg=json.loads((out/'trainer_config.json').read_text())
        job=dict(task=args.task,label='single_task_200',model=checked['path'],destination=str(dest),episodes=3,require_baseline=False,
            robotwin=str(ROOT.parent/'RoboTwin-lingbot'),qwen=cfg['model']['tokenizer_path'],sim_python='/data/miniconda3/envs/RoboTwin/bin/python')
        path=dest/'job.json';path.write_text(json.dumps(job,indent=2))
        run([sys.executable,str(ROOT/'experiment/official_control/evaluate.py'),'--job',str(path)],dest/'pipeline.log')
        state('complete',checkpoint=checked,evaluation=json.loads((dest/'summary.json').read_text()),shutdown=False)
    except BaseException as error:
        state('failed',error=repr(error));traceback.print_exc();return 1
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('/data/outputs/official_adjust_bottle_9680_bf16_m56_add200_20261020'))
    p.add_argument('--task',default='place_container_plate')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--cpu',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();args.out=args.out.resolve()
    if args.cpu:
        from experiment.official_control.prepare_adjust_bottle import cpu_check
        return cpu_check(args.out,expected_steps=200,expected_save_steps=200,expected_fp32=False,expected_mixed=False,expected_export='bf16')
    if args.worker:return worker(args)
    print(json.dumps(dict(task=args.task,out=str(args.out),source=str(args.source),steps=200,evaluation_episodes=3),indent=2))
    if args.execute:
        assert not args.out.exists()
        launch=args.out.with_name(args.out.name+'_pipeline');launch.mkdir(parents=True,exist_ok=False)
        with (launch/'pipeline.log').open('x') as f:
            child=subprocess.Popen(['setsid','nohup',sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(args.out),'--source',str(args.source),'--task',args.task,'--worker'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        print('STARTED',child.pid,str(launch))

if __name__=='__main__':raise SystemExit(main())

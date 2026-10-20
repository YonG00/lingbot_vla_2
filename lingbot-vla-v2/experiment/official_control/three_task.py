"""BF16-200续训三任务；任务等概率，任务内帧均匀，不启AL/窗口/EE。"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
TRAIN_TASKS=['place_container_plate','place_empty_cup','move_can_pot']
EVAL_TASKS=TRAIN_TASKS+['adjust_bottle','click_bell','click_alarmclock','press_stapler','turn_switch']
SOURCE=Path('/data/outputs/official_adjust_bottle_9680_bf16_m56_add200_20261020')


def uniform_weights(groups, episodes):
    owner={}
    for task, ids in groups.items():
        if not ids or len(set(ids))!=len(ids): raise ValueError('empty/duplicate episode group')
        for episode in ids:
            if episode in owner: raise ValueError('episode in multiple tasks')
            owner[episode]=task
    counts=Counter(episodes)
    if set(counts)!=set(owner): raise ValueError('filtered episodes differ from manifest')
    frames=Counter()
    for episode,n in counts.items(): frames[owner[episode]]+=n
    weights=[1.0/frames[owner[episode]] for episode in episodes]
    mass={t:sum(w for w,e in zip(weights,episodes) if owner[e]==t)/len(groups) for t in groups}
    if not all(math.isclose(v,1/len(groups),abs_tol=1e-10) for v in mass.values()): raise ValueError('unequal task mass')
    return weights,dict(frames=frames,task_probability=mass)


def install_sampler(out):
    import torch
    from lingbotvla import data
    groups=json.loads((out/'task_groups.json').read_text())
    original=data.build_dataloader
    def build(*args,**kwargs):
        dataset=kwargs.get('dataset',args[0] if args else None)
        assert len(dataset._datasets)==1 and kwargs.get('sampler') is None
        raw=dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index'])
        weights,summary=uniform_weights(groups,[int(row['episode_index']) for row in raw])
        kwargs['sampler']=torch.utils.data.WeightedRandomSampler(weights,len(weights),replacement=True,
            generator=torch.Generator().manual_seed(kwargs.get('seed',3)))
        print('UNIFORM_TASK_SAMPLER',json.dumps(summary),flush=True)
        return original(*args,**kwargs)
    data.build_dataloader=build


def prepare(out):
    import shutil
    from experiment.official_control.prepare_adjust_bottle import validate_identity
    from experiment.official_control.baseline_pipeline import verify_checkpoint
    validate_identity()
    source=SOURCE/'checkpoints/global_step_200/hf_ckpt'
    verify_checkpoint(source)
    assert shutil.disk_usage(out.parent).free > 30*2**30,'need two BF16 HF saves plus margin'
    cfg=json.loads((SOURCE/'trainer_config.json').read_text())
    manifest=json.loads((ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())['tasks']
    groups={t:sorted(manifest[t]['train_ids']+manifest[t]['val_ids']) for t in TRAIN_TASKS}
    assert all(len(ids)==50 for ids in groups.values())
    ids=sorted(e for values in groups.values() for e in values)
    assert len(ids)==len(set(ids))==150
    cfg['model'].update(model_path=str(source),config_path=str(source/'config.json'))
    cfg['data'].update(train_path=str(out/'datasets.txt'),episode_ids_file=str(out/'train_ids.json'),
        robot_config_root=str(ROOT/'configs/robot_configs'))
    cfg['train'].update(output_dir=str(out),max_steps=400,save_steps=200,enable_resume=False,
        load_checkpoint_path=None,step_offset=0,reset_step_counter=True)
    t=cfg['train']
    assert (t['micro_batch_size'],t['global_batch_size'],t['gradient_accumulation_steps'])==(56,56,1)
    assert not any(t[k] for k in ['auto_learning','ee_fk_loss_enabled','low_noise_fm_enabled','enable_fp32','enable_mixed_precision','action_fp32'])
    assert t['hf_export_dtype']=='bf16'
    out.mkdir(exist_ok=False)
    for name,value in [('trainer_config.json',cfg),('train_ids.json',ids),('task_groups.json',groups),
        ('plan.json',dict(source_model=str(source),expected_frames=sum(manifest[t]['train_frames']+manifest[t]['val_frames'] for t in TRAIN_TASKS),
        train_tasks=TRAIN_TASKS,evaluation_tasks=EVAL_TASKS,steps=400,save_steps=[200,400],evaluate_step=400))]:
        (out/name).write_text(json.dumps(value,indent=2))
    (out/'datasets.txt').write_text((SOURCE/'datasets.txt').read_text())
    command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port=29549',str(Path(__file__).resolve()),'train','--out',str(out)]
    (out/'train.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join(command)+'\n')


def cpu(out):
    from experiment.official_control.prepare_adjust_bottle import cpu_check
    def check(dataset,cfg):
        import torch
        from types import SimpleNamespace
        from lingbotvla import data
        from lingbotvla.data import data_loader
        raw=dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index'])
        groups=json.loads((out/'task_groups.json').read_text())
        weights,summary=uniform_weights(groups,[int(r['episode_index']) for r in raw])
        data_loader.get_parallel_state=lambda:SimpleNamespace(dp_size=1,dp_rank=0,sp_enabled=False,sp_size=1)
        install_sampler(out)
        loader=data.build_dataloader(dataset=dataset,micro_batch_size=56,global_batch_size=56,dataloader_batch_size=56,
            max_seq_len=2048,train_steps=len(dataset)//56,rmpad=False,rmpad_with_pos_ids=False,
            collate_fn=[data.VLADataCollatorWithPacking()],num_workers=0,prefetch_factor=None,pin_memory=False,drop_last=True,seed=3)
        batch=next(iter(loader))
        assert batch[0]['actions'].shape==(56,50,55) and torch.isfinite(batch[0]['actions']).all()
        assert isinstance(loader.state_dict(),dict)
        return dict(uniform_task_summary=summary,loader_batch_shape=[56,50,55])
    cpu_check(out,expected_steps=400,expected_save_steps=200,expected_fp32=False,expected_mixed=False,
        expected_export='bf16',expected_episodes=150,dataset_check=check)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','cpu','smoke','train'])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out
    if a.mode=='prepare': return prepare(out)
    if a.mode=='cpu': return cpu(out)
    install_sampler(out)
    cfg=json.loads((out/'trainer_config.json').read_text())
    os.environ['QWEN3VL_PATH']=cfg['model']['tokenizer_path']
    os.environ['AL_DYNAMO_CACHE_SIZE_LIMIT']='0'
    for key in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE']:os.environ[key]='1'
    if a.mode=='smoke':
        from experiment.official_control.bf16_trial import smoke
        return smoke(out)
    assert json.loads((out/'cpu_report.json').read_text())['state']=='cpu_pass'
    sys.argv=['tasks/vla/train_lingbotvla.py',str(out/'trainer_config.json')]
    os.chdir(ROOT)
    runpy.run_path(sys.argv[0],run_name='__main__')

if __name__=='__main__':main()

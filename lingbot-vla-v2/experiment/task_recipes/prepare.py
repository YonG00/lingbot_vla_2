"""把用户配方编译为冻结配置与确定性步计划；不隐式开训练。"""
import json
from pathlib import Path
import shlex
import sys
from experiment.task_recipes.schema import validate,milestones,save_interval
from experiment.task_recipes.planning import build_plan,task_groups,digest
from experiment.task_recipes.replay_pool import resolve_pool

ROOT=Path(__file__).resolve().parents[2]


def path(value):
    p=Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p


def preview(raw):
    c=validate(resolve_pool(raw,ROOT))
    for key in c['source']:c['source'][key]=str(path(c['source'][key]).resolve())
    c['output_dir']=str(path(c['output_dir']).resolve())
    c['data']['dataset_root']=str(path(c['data']['dataset_root']).resolve())
    manifest=json.loads((ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())
    groups,frames=task_groups(c,manifest)
    plan=build_plan(c)
    return c,plan,groups,frames


def prepare(raw):
    import shutil
    import os
    from experiment.official_control.prepare_adjust_bottle import validate_identity
    from experiment.official_control.baseline_pipeline import verify_checkpoint
    c,plan,groups,frames=preview(raw)
    validate_identity()
    source=path(c['source']['checkpoint']);verify_checkpoint(source)
    template=json.loads(path(c['source']['trainer_config']).read_text())
    cfg=template.get('trainer_config',template)
    cfg=json.loads(json.dumps(cfg))
    t=cfg['train']
    if any(t.get(k) for k in ['auto_learning','ee_fk_loss_enabled','low_noise_fm_enabled','train_expert_only','use_lora','freeze_vision_encoder','rmpad','rmpad_with_pos_ids']):raise ValueError('v1 requires standard full-model trainer without AL/EE/LN-FM/packing')
    out=Path(c['output_dir']);saves=milestones(c)
    model_bytes=sum(p.stat().st_size for p in source.glob('*.safetensors'))
    extra_saves=0
    if c['remediation']['enabled']:
        worst_steps=100*len(c['evaluation']['tasks'])
        if c['checkpoint']['mode']=='task':
            count=len(c['evaluation']['tasks']);interval=c['checkpoint'].get('every_tasks',1)
            per_round=(count+interval-1)//interval
        else:
            interval=c['checkpoint']['every_steps']
            per_round=len(set(list(range(interval,worst_steps+1,interval))+[worst_steps]))
        extra_saves=per_round*c['remediation']['max_rounds']
    required=int(model_bytes*(len(saves)+extra_saves)*1.1)+3*2**30
    if shutil.disk_usage(out.parent).free<required:raise ValueError('insufficient space; will not drop checkpoint milestones')
    cfg['model'].update(model_path=str(source),config_path=str(source/'config.json'))
    t.update(output_dir=str(out),max_steps=c['train']['steps'],save_steps=save_interval(c),save_epochs=0,
        save_at_steps=saves if c['checkpoint']['mode']=='task' else None,
        micro_batch_size=c['train']['micro_batch_size'],gradient_accumulation_steps=c['train']['gradient_accumulation_steps'],
        global_batch_size=plan['global_batch_size'],seed=c['train']['seed'],enable_mixed_precision=False,enable_fp32=False,action_fp32=False,
        hf_export_dtype='bf16',save_hf_weights=True,dcp_save_mode='hf_only',async_save_hf_weights=False,
        enable_resume=False,load_checkpoint_path=None,step_offset=0,reset_step_counter=True,num_train_epochs=1,open_loop_eval_steps=0)
    cfg['data'].update(train_path=str(out/'datasets.txt'),episode_ids_file=str(out/'train_ids.json'),robot_config_root=str(ROOT/'configs/robot_configs'))
    if not (Path(c['data']['dataset_root'])/'meta').is_dir():raise ValueError('dataset metadata missing')
    dataset_lines='robotwin '+c['data']['dataset_root']+'\n'
    out.mkdir(parents=True,exist_ok=False)
    from experiment.data_selection.inheritance import freeze_registry,registry_digest
    from experiment.data_selection.fixed_importance import fixed_policies
    fixed=fixed_policies(ROOT,groups,c['data']['dataset_root'])
    importance=freeze_registry(source,out,
        explicit=os.environ.get('LINGBOT_SAMPLE_WEIGHT_CONFIG'),fixed=fixed)
    if importance is not None:
        from experiment.data_selection.weights import SelectionWeights
        loaded_importance=SelectionWeights(importance)
        staged=loaded_importance.stage_tables or loaded_importance.weight_sum
        if staged and any(key in c['train'] for key in ['role_segments','target_segments']):raise ValueError('stage weighting requires uniform task sampling; legacy role/target modes are mutually exclusive')
        if staged and c['train'].get('importance_sampling_multiplier',1)!=1:raise ValueError('stage weighting does not implicitly change sampling probabilities')
    ids=sorted(e for values in groups.values() for e in values)
    report=dict(expected_frames=frames,episodes=len(ids),hf_steps=saves,recipe_sha256=digest(c),required_disk_gib=required/2**30,
        remediation_reserved_checkpoints=extra_saves,
        source_model=str(source),trainer_config_sha256=digest(cfg),task_groups_sha256=digest(groups),train_ids_sha256=digest(ids),
        evaluation_steps=[c['train']['steps'] if x=='final' else x for x in c['evaluation']['checkpoints']])
    if importance is not None:report['sample_weight_registry_sha256']=registry_digest(importance)
    for name,value in [('recipe.json',c),('step_plan.json',plan),('task_groups.json',groups),('train_ids.json',ids),('trainer_config.json',cfg),('plan.json',report)]:
        (out/name).write_text(json.dumps(value,indent=2))
    (out/'datasets.txt').write_text(dataset_lines)
    command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=1','--master_port=29557',str(ROOT/'experiment/task_recipes/run.py'),'--stage','train','--out',str(out)]
    importance_export=('export LINGBOT_SAMPLE_WEIGHT_CONFIG='+shlex.quote(str(importance))+'\n') if importance else ''
    (out/'train.sh').write_text('#!/bin/bash\nset -euo pipefail\n'+importance_export+'cd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join(command)+'\n')
    return report

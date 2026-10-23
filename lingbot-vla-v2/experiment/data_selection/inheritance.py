"""Freeze task importance with a run; preserve unrelated tasks across continuations."""
import hashlib
import json
from pathlib import Path
import shutil
from .weights import SelectionWeights

NAME='sample_weight_registry.json'


def read_policies(path):
    path=Path(path).resolve();value=json.loads(path.read_text())
    if 'policies' not in value:
        SelectionWeights(path)
        task=value.get('task')
        if not isinstance(task,str) or not task or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in task):
            raise ValueError('task identity required for importance inheritance')
        return {task:path}
    if value.get('version')!=1 or not isinstance(value['policies'],dict):raise ValueError('invalid importance registry')
    result={}
    for task,relative in value['policies'].items():
        child=Path(relative);child=child if child.is_absolute() else path.parent/child
        if hashlib.sha256(child.read_bytes()).hexdigest()!=value.get('policy_sha256',{}).get(task):raise ValueError('task importance config fingerprint mismatch')
        data=json.loads(child.read_text())
        if 'policies' in data or data.get('task')!=task:raise ValueError('registry task/config mismatch')
        result.update(read_policies(child))
    return result


def find_source_registry(source_run):
    source_run=Path(source_run)
    registry=source_run/NAME
    if registry.exists():return registry
    if source_run.name=='hf_ckpt' or (source_run/'model.safetensors.index.json').exists():
        if source_run.parent.parent.name!='checkpoints':return None
        source_run=source_run.parents[2]
        registry=source_run/NAME
        if registry.exists():return registry
    # Backward-compatible import for runs predating the registry. No global search.
    audit=source_run.with_name(source_run.name+'_pipeline')/'sample_weight_audit.json'
    if audit.exists():
        data=json.loads(audit.read_text());config=data.get('weight_config',data.get('config'))
        if config:
            p=Path(config)
            if not p.is_file():raise ValueError('source importance config missing: '+str(p))
            return p
    return None


def freeze_registry(source_run,destination,explicit=None,fixed=None):
    policies={};source=find_source_registry(source_run)
    if source:policies.update(read_policies(source))
    if explicit:policies.update(read_policies(explicit)) # Replace this task, preserve all others.
    if fixed:
        for task,path in fixed.items():
            selected=read_policies(path)
            if set(selected)!={task}:raise ValueError("fixed importance task mismatch")
            if explicit and task in read_policies(explicit):
                requested=SelectionWeights(read_policies(explicit)[task]);locked=SelectionWeights(path)
                if requested.coordinate_weights!=locked.coordinate_weights:raise ValueError("explicit importance conflicts with fixed task policy: "+task)
            policies.update(selected)
    if not policies:return None
    destination=Path(destination);registry=destination/NAME
    if registry.exists():raise ValueError('refuse overwrite frozen importance registry')
    configurations={};coordinate_owner={};dataset_identity=None
    for task,path in sorted(policies.items()):
        cfg=json.loads(path.read_text());loaded=SelectionWeights(path)
        identity=(loaded.dataset_name,loaded.dataset_root)
        if dataset_identity is not None and identity!=dataset_identity:raise ValueError('importance registry currently supports one merged dataset')
        dataset_identity=identity
        for coordinate in loaded.keys:
            if coordinate in coordinate_owner:raise ValueError('different task importance selections overlap')
            coordinate_owner[coordinate]=task
        selected=Path(cfg['selected_frames']);selected=selected if selected.is_absolute() else path.parent/selected
        configurations[task]=(cfg,selected)
    relative={}
    for task,(cfg,selected) in configurations.items():
        target=destination/'sample_weights'/task;target.mkdir(parents=True,exist_ok=False)
        shutil.copy2(selected,target/'selected_frames.json')
        cfg=dict(cfg,selected_frames='selected_frames.json')
        for field in ['stage_table','stage_policy']:
            if field in cfg:
                dependency=Path(cfg[field]);dependency=dependency if dependency.is_absolute()else policies[task].parent/dependency
                name=field+'.json';shutil.copy2(dependency,target/name);cfg[field]=name
        (target/'weight_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        relative[task]=str((target/'weight_config.json').relative_to(destination))
    registry.write_text(json.dumps(dict(version=1,policies=relative,policy_sha256={task:registry_digest(destination/path) for task,path in relative.items()}),indent=2)+'\n')
    return registry


def registry_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

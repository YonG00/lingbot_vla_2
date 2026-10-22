"""配方进程内适配器；不改原训练器或默认数据加载器。"""
from pathlib import Path
import json
from experiment.task_recipes.planning import PlannedSampler,index_pools,digest,build_plan


class StreamDataset:
    """Virtual epoch length; indices always remain in the real filtered frame space."""
    def __init__(self,dataset,length):
        self.dataset=dataset;self.length=length
    def __len__(self):return self.length
    def __getitem__(self,index):return self.dataset[index]
    def __getattr__(self,name):return getattr(object.__getattribute__(self,'dataset'),name)


def load_run(out):
    out=Path(out)
    c=json.loads((out/'recipe.json').read_text())
    p=json.loads((out/'step_plan.json').read_text())
    if p['recipe_sha256']!=digest(c) or p!=build_plan(c):raise ValueError('frozen recipe/plan mismatch')
    groups=json.loads((out/'task_groups.json').read_text())
    report=json.loads((out/'plan.json').read_text())
    for name,value in [('trainer_config',json.loads((out/'trainer_config.json').read_text())),('task_groups',groups),('train_ids',json.loads((out/'train_ids.json').read_text()))]:
        if digest(value)!=report[name+'_sha256']:raise ValueError('frozen '+name+' differs from prepared run')
    if 'sample_weight_registry_sha256' in report:
        from experiment.data_selection.inheritance import registry_digest
        from experiment.data_selection.weights import SelectionWeights
        registry=out/'sample_weight_registry.json'
        if registry_digest(registry)!=report['sample_weight_registry_sha256']:raise ValueError('frozen task importance registry changed')
        SelectionWeights(registry) # Includes selected-coordinate fingerprint checks.
    return c,p,groups


def pools_for(dataset,groups):
    if len(dataset._datasets)!=1:raise ValueError('one merged dataset required')
    raw=dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index'])
    return index_pools(groups,[int(row['episode_index']) for row in raw])


def sampling_weights_for(dataset,c,out):
    multiplier=c['train'].get('importance_sampling_multiplier',1.)
    if multiplier==1:return {}
    from experiment.data_selection.weights import SelectionWeights
    policy=SelectionWeights(Path(out)/'sample_weight_registry.json')
    raw=dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index','frame_index'])
    weights={i:float(multiplier) for i,row in enumerate(raw)
             if (int(row['episode_index']),int(row['frame_index'])) in policy.keys}
    if len(weights)!=len(policy.keys):raise ValueError('sampling selection not fully represented in filtered dataset')
    print('IMPORTANCE_SAMPLING',json.dumps(dict(multiplier=multiplier,selected=len(weights),normalization='within_task')),flush=True)
    return weights


def make_sampler(dataset,c,p,groups,out):
    pools=pools_for(dataset,groups)
    if 'target_segments' not in c['train']:
        return PlannedSampler(p,pools,c['train']['seed'],sampling_weights_for(dataset,c,out))
    from experiment.data_selection.inheritance import read_policies
    from experiment.data_selection.weights import SelectionWeights
    from experiment.task_recipes.segment_sampling import SegmentSampler
    setting=c['train']['target_segments'];task=setting['task']
    config=read_policies(Path(out)/'sample_weight_registry.json')[task]
    policy=SelectionWeights(config);value=json.loads(config.read_text())
    raw=dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index','frame_index'])
    coordinates={(int(row['episode_index']),int(row['frame_index'])):i for i,row in enumerate(raw)}
    events={kind:[] for kind in setting['ratios']}
    for segment in value['sampling_segments']:
        events[segment['kind']].append([coordinates[(segment['episode_index'],frame)] for frame in range(segment['start_frame'],segment['end_frame']+1)])
    selected={i for groups_ in events.values() for event in groups_ for i in event}
    if selected!={coordinates[key] for key in policy.keys}:raise ValueError('segment/weight selection mismatch')
    background=[i for i in pools[task] if i not in selected]
    sampler=SegmentSampler(p,pools,c['train']['seed'],task,events,background,setting['fraction'],setting['ratios'])
    print('TARGET_SEGMENT_SAMPLING',json.dumps(dict(fraction=setting['fraction'],ratios=setting['ratios'],events={k:len(v) for k,v in events.items()},selected=len(selected),background=len(background),within_event='uniform_frame',between_events='uniform')),flush=True)
    return sampler


def install(out):
    c,p,groups=load_run(out)
    from lingbotvla import data
    original_dataset,original_loader=data.build_vla_dataset,data.build_dataloader
    holder={}
    def build_dataset(*args,**kwargs):
        dataset=original_dataset(*args,**kwargs)
        dataset.strict_getitem=True  # never silently replace a corrupt frame with another task
        holder['sampler']=make_sampler(dataset,c,p,groups,out)
        return StreamDataset(dataset,len(p['steps'])*p['global_batch_size'])
    def build_loader(*args,**kwargs):
        from lingbotvla.distributed.parallel_state import get_parallel_state
        if get_parallel_state().dp_size!=1:raise ValueError('recipe v1 supports single GPU only')
        if kwargs.get('sampler') is not None or kwargs.get('rmpad',False) or kwargs.get('rmpad_with_pos_ids',False):raise ValueError('recipe sampler conflicts with existing sampler/packing')
        if kwargs['global_batch_size']!=p['global_batch_size']:raise ValueError('effective GBS differs from plan')
        kwargs['sampler']=holder['sampler']
        kwargs['train_steps']=len(p['steps'])
        print('RECIPE_SAMPLER',json.dumps(dict(recipe_sha256=p['recipe_sha256'],steps=len(p['steps']),batch=p['global_batch_size'])),flush=True)
        return original_loader(*args,**kwargs)
    data.build_vla_dataset=build_dataset;data.build_dataloader=build_loader

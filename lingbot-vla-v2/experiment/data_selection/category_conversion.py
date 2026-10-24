"""Offline adapters into one immutable per-frame category contract."""
from .full_stages import StageTable
from .category_table import CategoryTable,ARMS,BRANCHES
import json,hashlib,tempfile,shutil
from pathlib import Path
SOURCE_CHANNELS={'action':{'left':list(range(7)),'right':list(range(7,14))},'observation.state':{'left':list(range(7)),'right':list(range(7,14))}}
def base_policy(classes):return dict(version=1,weighting_unit='action_timestep',normalization='weight_sum',overlap_rule='max',future_video_rule='target_endpoint',classes=classes)
def from_stages(config,base):
    old=StageTable(config,base)
    if old.policy['future_video_rule']!='target_endpoint':raise ValueError('legacy endpoint_max must be resolved explicitly before migration')
    classes={'background':dict(fm=1.,visual=1.)};positions={}
    for ep,arms in old.sub_labels.items():
        for arm,labels in arms.items():
            for f,name in enumerate(labels):positions.setdefault(name,{}).setdefault(arm,(ep,f))
    for name,arms in positions.items():
        main=name.split('/')[0]
        def weight(branch,arm):
            if arm in arms:
                ep,f=arms[arm];return old.frame_weights[branch][ep][arm][f]
            base_override=old.policy.get('overrides',{}).get('visual',{}).get(arm,{})if branch not in ('fm','visual')else {}
            return old.policy.get('overrides',{}).get(branch,{}).get(arm,{}).get(name,base_override.get(name,old.policy.get('visual_branches',{}).get(branch,old.policy.get(branch,old.policy['visual']))[main]))
        def compact(values):return next(iter(values.values()))if len(set(values.values()))==1 else values
        fm={a:weight('fm',a)for a in ARMS};visual={a:weight('visual',a)for a in ARMS}
        entry=dict(fm=compact(fm),visual=compact(visual));branches={b:compact({a:weight(b,a)for a in ARMS})for b in BRANCHES[1:]if {a:weight(b,a)for a in ARMS}!=visual}
        if branches:entry['visual_branches']=branches
        classes[name]=entry
    table={k:v for k,v in old.table.items()if k!='episodes'};table.update(label_source='trajectory_stages',episodes=[])
    for ep in old.table['episodes']:
        table['episodes'].append(dict(episode_index=ep['episode_index'],length=ep['length'],arms={a:[dict(seg,classes=[seg['stage']+'/'+seg['substage']],legacy_stage=seg['stage'])for seg in segs]for a,segs in ep['arms'].items()}))
    return table,base_policy(classes)
def from_targets(config,rows,catalog,policy,fps=15):
    lengths={v['episode_index']:v['length']for v in catalog}
    if len(lengths)!=len(catalog):raise ValueError('duplicate catalog episodes')
    coordinates={}
    for row in rows:
        key=(row['episode_index'],row['frame_index'])
        if key in coordinates or key[0]not in lengths or not 0<=key[1]<lengths[key[0]]:raise ValueError('invalid target coordinate')
        coordinates[key]=row
    episodes=[]
    for ep,n in sorted(lengths.items()):
        arms={}
        for arm in ARMS:
            values=[]
            for f in range(n):
                row=coordinates.get((ep,f),{});names=[]
                for label in row.get('labels',[]):
                    if 'arm'in label and label['arm']not in ARMS:raise ValueError('target class arm identity')
                    if label.get('arm',arm)!=arm:continue
                    name=label.get('category',label.get('source',label.get('stage')))
                    if name not in policy['classes']:raise ValueError('target category absent from weight config: '+str(name))
                    names.append(name)
                values.append(sorted(set(names))or ['background'])
            start=0;segs=[]
            for end in range(1,n+1):
                if end==n or values[end]!=values[start]:segs.append(dict(start=start,end=end,classes=values[start]));start=end
            arms[arm]=segs
        episodes.append(dict(episode_index=ep,length=n,arms=arms))
    return dict(schema_version=1,dataset_name=config['dataset_name'],task=config['task'],coordinate_space='original_episode_frame',label_source='target_classes',fps=fps,source_action_channels=SOURCE_CHANNELS,episodes=episodes),policy

def write_snapshot(config,table,policy,out):
    out=Path(out)
    if out.exists():raise ValueError('refuse overwrite category snapshot')
    out.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.category_weights_',dir=out.parent))
    try:
        selected=[dict(episode_index=e['episode_index'],frame_index=f)for e in table['episodes']for f in range(e['length'])]
        for name,data in [('frame_classes.json',table),('category_loss_weights.json',policy),('selected_frames.json',selected)]:(tmp/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        cfg=dict(version=1,task=config['task'],dataset_name=config['dataset_name'],dataset_root=config['dataset_root'],weighting_unit='action_timestep',normalization='weight_sum',multiplier=1)
        for field,name in [('selected_frames','selected_frames.json'),('category_table','frame_classes.json'),('category_weights','category_loss_weights.json')]:cfg[field]=name;cfg[field+'_sha256']=hashlib.sha256((tmp/name).read_bytes()).hexdigest()
        (tmp/'weight_config.json').write_text(json.dumps(cfg,indent=2)+'\n');result=CategoryTable(cfg,tmp);tmp.rename(out);return result
    finally:
        if tmp.exists():shutil.rmtree(tmp)

import json,sys
from pathlib import Path
import numpy as np
import pyarrow.dataset as ds
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
from experiment.data_selection.grasp_sequence import forward_kinematics
m=json.loads((r/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())['tasks']['move_pillbottle_pad'];ids=sorted(m['train_ids']+m['val_ids']);root=Path('/root/autodl-tmp/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30');rows=ds.dataset(str(root/'data'),format='parquet').to_table(columns=['episode_index','frame_index','observation.state'],filter=ds.field('episode_index').isin(ids),use_threads=False).to_pylist();spec=json.loads((r/'configs/kinematics/robotwin_aloha_agilex_ee.json').read_text());group={}
for v in rows:group.setdefault(v['episode_index'],[]).append(v)
assert json.loads((root/'meta/info.json').read_text())['fps']==15
result=[]
for ep,items in sorted(group.items()):
 items.sort(key=lambda v:v['frame_index']);assert [v['frame_index']for v in items]==list(range(len(items)));s=np.asarray([v['observation.state']for v in items]);e=[]
 for arm,j in [('left',0),('right',7)]:
  g=s[:,j+6];cross=np.flatnonzero((g[:-1]>.5)&(g[1:]<=.5))+1
  for c in cross:
   op=np.flatnonzero(g[:c]>=.95)
   if not len(op):continue
   cs=int(op[-1])+1;cl=np.flatnonzero(g[cs:]<=.2)
   if not len(cl):continue
   ce=cs+int(cl[0]);p,_=forward_kinematics(s[:,j:j+6],spec[arm]);release=np.flatnonzero(g[ce+1:]>.2)
   if not len(release):continue
   rs=ce+1+int(release[0]);opens=np.flatnonzero(g[rs:]>=.95)
   if not len(opens):continue
   re=rs+int(opens[0]);lifts=np.flatnonzero(p[ce:rs,2]-p[ce,2]>=.03)
   if not len(lifts):continue
   le=ce+int(lifts[0]);target=np.median(p[rs:re+1],axis=0)
   far=np.flatnonzero(np.linalg.norm(p[:cs]-np.median(p[cs:ce+1],axis=0),axis=1)>.12);gs=int(far[-1])+1 if len(far)else 0
   pre=np.flatnonzero((np.arange(len(p))>le)&(np.arange(len(p))<rs)&(np.linalg.norm(p[:,:2]-target[:2],axis=1)<=.03)&(p[:,2]-target[2]>=.03))
   e.append(dict(arm=arm,cs=cs,ce=ce,le=le,rs=rs,re=re,gs=gs,pre_start=int(pre[0])if len(pre)else None,pre_end=int(pre[-1])if len(pre)else None,n=len(items),release_z=float(target[2]),max_lift=float(p[ce:rs,2].max()-p[ce,2])))
 result.append(dict(episode=ep,events=e))
from collections import Counter
from experiment.data_selection.category_conversion import base_policy,from_targets,write_snapshot
selected=[];catalog=[];counts=Counter();arms=Counter()
for item in result:
 ep=item['episode'];assert len(item['events'])==1,(ep,item['events']);e=item['events'][0];assert e['pre_start']is not None and 0<=e['gs']<e['cs']<=e['ce']<e['le']<e['pre_start']<e['rs']<=e['re']<e['n']
 catalog.append(dict(episode_index=ep,length=e['n']));arm=e['arm'];arms[arm]+=1;j=0 if arm=='left'else 7;items=sorted(group[ep],key=lambda v:v['frame_index']);state=np.asarray([v['observation.state']for v in items]);p,_=forward_kinematics(state[:,j:j+6],spec[arm]);g=state[:,j+6];speed=np.linalg.norm(np.gradient(p,axis=0)*15,axis=1);grasp=np.median(p[e['cs']:e['ce']+1],axis=0);target=np.median(p[e['rs']:e['re']+1],axis=0)
 for f in range(e['n']):
  name=None
  if e['gs']<=f<e['cs']:
   distance=np.linalg.norm(p[f]-grasp);name='grasp/approach_12_to_9cm'if distance>.09 else 'grasp/approach_9_to_6cm'if distance>.06 else 'grasp/approach_6_to_1p5cm'if distance>.015 else 'grasp/open_settle'
  elif e['cs']<=f<=e['ce']:name='grasp/closing_first_half'if g[f]>.5 else 'grasp/closing_second_half'
  elif e['ce']<f<=e['le']:name='grasp/closed_hold_proxy'if p[f,2]-p[e['ce'],2]<.002 else 'grasp/lift_first_1p5cm'if p[f,2]-p[e['ce'],2]<.015 else 'grasp/lift_1p5_to_3cm'
  elif e['pre_start']<=f<e['rs']:
   height=p[f,2]-target[2]
   if height>.04:name='place/prealign_move'if speed[f]>.025 else 'place/prealign_settle'
   else:name='place/descent_4_to_1p5cm'if height>.015 else 'place/contact_pose_settle'
  elif e['rs']<=f<=e['re']:name='place/release_first_half'if g[f]<.5 else 'place/release_second_half'
  if name:selected.append(dict(episode_index=ep,frame_index=f,labels=[dict(arm=arm,category=name)]));counts[name]+=1
assert max(counts.values())/len(rows)<=.1,dict(counts)
high={'grasp/closed_hold_proxy','grasp/approach_6_to_1p5cm','grasp/closing_first_half','grasp/closing_second_half','grasp/lift_first_1p5cm'}
video={'place/prealign_move','place/prealign_settle','place/contact_pose_settle'}
release={'place/release_first_half','place/release_second_half'}
classes={'background':dict(fm=1,visual=1)}
assert high|video|release <=set(counts)
for name in counts:
 value=20 if name in high else 10 if name in release else 5
 classes[name]=dict(fm=value,visual=value)
 if name in video:classes[name]['visual_branches']=dict(current_video=20,future_video=20)
cfg=dict(task='move_pillbottle_pad',dataset_name='robotwin',dataset_root=str(root));policy=base_policy(classes);table,policy=from_targets(cfg,selected,catalog,policy);out=r/'refs/data_selection/move_pillbottle_pad_differential_v1';weights=write_snapshot(cfg,table,policy,out)
assert len(catalog)==50 and len(selected)==len({(v['episode_index'],v['frame_index'])for v in selected})
for ep,arms_labels in weights.category_labels.items():
 for arm,labels in arms_labels.items():
  for f,names in enumerate(labels):
   assert len(names)==1
   name=names[0]
   for branch in ['fm','current_depth','future_depth','current_video','future_video']:
    entry=classes[name];expected=entry.get('visual_branches',{}).get(branch,entry['fm']if branch=='fm'else entry['visual'])
    assert weights.frame_weights[branch][ep][arm][f]==expected
report=dict(episodes=len(catalog),frames=len(rows),selected_frames=len(selected),selected_fraction=len(selected)/len(rows),category_counts=dict(counts),category_fractions={k:v/len(rows)for k,v in counts.items()},arms=dict(arms),events=result,contact_verified=False,subsampled=False,model_initialized=False,optimizer_steps=0)
(out/'selection_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='events'},indent=2))

assert "torch" not in sys.modules,"annotation must remain CPU/NumPy only"

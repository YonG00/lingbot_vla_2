import os,sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.dataset as ds
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
from experiment.data_selection.grasp_sequence import forward_kinematics
from experiment.data_selection.category_conversion import base_policy,from_targets,write_snapshot
root=Path('/root/autodl-tmp/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30');fps=json.loads((root/'meta/info.json').read_text())['fps'];assert fps==15
m=json.loads((r/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())['tasks']['lift_pot'];ids=sorted(m['train_ids']+m['val_ids']);rows=ds.dataset(str(root/'data'),format='parquet').to_table(columns=['episode_index','frame_index','observation.state'],filter=ds.field('episode_index').isin(ids),use_threads=False).to_pylist();spec=json.loads((r/'configs/kinematics/robotwin_aloha_agilex_ee.json').read_text());group={}
for row in rows:group.setdefault(row['episode_index'],[]).append(row)
assert set(group)==set(ids)and len(ids)==50
selected={};events=[];catalog=[];counts={a:Counter()for a in ['left','right']}
for ep,items in sorted(group.items()):
 items.sort(key=lambda v:v['frame_index']);n=len(items);assert [v['frame_index']for v in items]==list(range(n));s=np.asarray([v['observation.state']for v in items]);catalog.append(dict(episode_index=ep,length=n));assert s.shape==(n,14)
 for arm,j in [('left',0),('right',7)]:
  p,_=forward_kinematics(s[:,j:j+6],spec[arm]);g=s[:,j+6];crossings=np.flatnonzero((g[:-1]>.02)&(g[1:]<=.02))+1;found=[]
  for ce in crossings:
   opens=np.flatnonzero(g[:ce]>=.49)
   if not len(opens):continue
   cs=int(opens[-1])+1;ce=int(ce)
   if np.max(np.linalg.norm(p[cs:ce+1]-np.median(p[cs:ce+1],axis=0),axis=1))>.015:continue
   lifts=np.flatnonzero((p[ce:,2]-p[ce,2]>=.03)&(g[ce:]<=.2))
   if not len(lifts):continue
   le=ce+int(lifts[0])
   if np.max(g[ce:le+1])>.5:continue
   target=np.median(p[cs:ce+1],axis=0);distance=np.linalg.norm(p-target,axis=1);far=np.flatnonzero(distance[:cs]>.06)
   if not len(far):continue
   start=int(far[-1])+1
   assert start<cs<=ce<le<n
   found.append(dict(episode_index=ep,arm=arm,start=start,close_start=cs,close_end=ce,lift_confirm=le,length=n,proxy_only=True))
  assert len(found)==1,(ep,arm,found)
  e=found[0];events.append(e)
  for f in range(e['start'],e['close_end']+1):
   if f<e['close_start']:
    distance=np.linalg.norm(p[f]-np.median(p[e['close_start']:e['close_end']+1],axis=0));name='ear/approach_6_to_3p5cm'if distance>.035 else 'ear/approach_3p5_to_1p5cm'if distance>.015 else 'ear/half_open_settle'
   else:name='ear/closing_first_half'if g[f]>.25 else 'ear/closing_second_half'
   key=(ep,f);selected.setdefault(key,dict(episode_index=ep,frame_index=f,labels=[]))['labels'].append(dict(arm=arm,category=name));counts[arm][name]+=1
report=dict(episodes=len(group),frames=len(rows),events=events,selected_frames=len(selected),selected_fraction=len(selected)/len(rows),arm_category_counts={a:dict(v)for a,v in counts.items()},contact_verified=False,subsampled=False,model_initialized=False,optimizer_steps=0)
print(json.dumps({k:v for k,v in report.items()if k!='events'},indent=2),flush=True)
classes={'background':dict(fm=1,visual=1)}
for a,c in counts.items():
 for k,v in c.items():
  assert v/len(rows)<=.1,(a,k,v,len(rows))
  classes[k]=dict(fm=20,visual=20)
cfg=dict(task='lift_pot',dataset_name='robotwin',dataset_root=str(root));table,policy=from_targets(cfg,list(selected.values()),catalog,base_policy(classes),fps=fps);out=r/'refs/data_selection/lift_pot_ear_approach_grasp20_v2';w=write_snapshot(cfg,table,policy,out)
for ep,arms in w.category_labels.items():
 for arm,labels in arms.items():
  for f,names in enumerate(labels):
   assert len(names)==1
   for branch in ['fm','current_depth','future_depth','current_video','future_video']:assert w.frame_weights[branch][ep][arm][f]==(1 if names[0]=='background'else 20)
(out/'selection_report.json').write_text(json.dumps(report,indent=2));assert 'torch'not in sys.modules

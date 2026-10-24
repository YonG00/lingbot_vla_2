import sys,json,hashlib,os
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.dataset as ds
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r));os.chdir(r)
from experiment.data_selection.grasp_sequence import forward_kinematics
from experiment.data_selection.category_conversion import from_targets,base_policy,write_snapshot
root=Path('/root/autodl-tmp/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30')
entry=json.loads((r/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())['tasks']['move_can_pot'];ids=sorted(entry['train_ids']+entry['val_ids'])
rows=ds.dataset(str(root/'data'),format='parquet').to_table(columns=['episode_index','frame_index','observation.state'],filter=ds.field('episode_index').isin(ids),use_threads=False).to_pylist()
spec=json.loads((r/'configs/kinematics/robotwin_aloha_agilex_ee.json').read_text());group={}
for row in rows:group.setdefault(row['episode_index'],[]).append(row)
events=[];selected=[];catalog=[];counts=Counter()
for ep,items in sorted(group.items()):
 items.sort(key=lambda x:x['frame_index']);n=len(items);assert [x['frame_index']for x in items]==list(range(n));catalog.append(dict(episode_index=ep,length=n));s=np.asarray([x['observation.state']for x in items]);found=[]
 for arm,j in [('left',0),('right',7)]:
  g=s[:,j+6];cross=np.flatnonzero((g[:-1]>.5)&(g[1:]<=.5))+1
  for c in cross:
   opens=np.flatnonzero(g[:c]>=.95)
   if not len(opens):continue
   cs=int(opens[-1])+1;closed=np.flatnonzero(g[cs:]<=.2)
   if not len(closed):continue
   ce=cs+int(closed[0]);p,_=forward_kinematics(s[:,j:j+6],spec[arm]);lift=np.flatnonzero((p[ce:,2]-p[ce,2]>=.03)&(g[ce:]<=.2))
   if not len(lift):continue
   le=ce+int(lift[0]);assert np.max(g[ce:le+1])<=.5
   movement=np.flatnonzero(np.linalg.norm(p[:cs]-p[0],axis=1)>.005);assert len(movement)
   start=max(0,int(movement[0])-1)
   found.append(dict(episode_index=ep,arm=arm,start=start,close_start=cs,close_end=ce,lift_confirm=le,end=le+1,length=n,proxy_only=True))
 assert len(found)==1,(ep,found)
 e=found[0];events.append(e);counts[e['arm']]+=1
 for f in range(e['start'],e['end']):selected.append(dict(episode_index=ep,frame_index=f,labels=[dict(arm=e['arm'],category='approach'if f<e['close_start']else 'grasp')]))
assert len(events)==50 and set(group)==set(ids)
cfg=dict(task='move_can_pot',dataset_name='robotwin',dataset_root=str(root));policy=base_policy({'background':dict(fm=1,visual=1),'approach':dict(fm=20,visual=20),'grasp':dict(fm=20,visual=20)})
table,policy=from_targets(cfg,selected,catalog,policy);out=r/'refs/data_selection/move_can_pot_approach_grasp20_v1';weight=write_snapshot(cfg,table,policy,out)
for e in events:
 for f in range(e['length']):
  for a in ['left','right']:
   expected=20 if a==e['arm']and e['start']<=f<e['end']else 1
   assert weight.frame_weights['fm'][e['episode_index']][a][f]==expected
report=dict(episodes=len(events),frames=len(rows),selected_frames=len(selected),selected_fraction=len(selected)/len(rows),arms=dict(counts),events=events,contact_verified=False,definition='active-arm first 5mm TCP movement minus one frame through closed-gripper initial 3cm lift; no transport/place; no sampling change',model_initialized=False,optimizer_steps=0)
(out/'selection_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='events'}),flush=True)

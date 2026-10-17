from pathlib import Path
from _paths import OUTPUT_ROOT, DATASET_ROOT
import h5py,numpy as np
rows=[];poses=[];labels=[];action_delta=[]
for task in ('place_object_stand','put_object_cabinet'):
 paths=sorted((DATASET_ROOT/task/'aloha_agilex/data').glob('*.hdf5'))
 for path in paths[::max(1,len(paths)//5)][:5]:
  with h5py.File(path) as f:
   if 'state/joint_states' not in f: continue
   values=f['state/joint_states']
   for idx in np.unique(np.linspace(0,len(values)-1,10,dtype=int)):
    v=np.asarray(values[idx]);rows.append(np.r_[v[:6],v[7:13]])
    poses.append([f['state/left_ee_poses'][idx],f['state/right_ee_poses'][idx]])
    labels.append(task+':'+path.name+':'+str(idx))
    action_delta.append(np.linalg.norm(v-np.asarray(f['action/joint_states'][idx])))
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
np.savez(OUTPUT_ROOT/'recorded_reference.npz',joints=np.asarray(rows),poses=np.asarray(poses),labels=np.asarray(labels))
print('raw_state_pose_samples',len(rows),'action_state_delta_mean',np.mean(action_delta),'max',max(action_delta),flush=True)
print('sample_state_ee',poses[0],flush=True)

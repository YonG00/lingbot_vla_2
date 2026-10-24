"""Offline targets indexed by original episode/frame, with explicit action source clock."""
import json
from pathlib import Path
import numpy as np
from .tcp_geometry_config import load_settings,checked,STAGES
class PoseCache:
    def __init__(self,path,dataset_name,dataset_root):
        self.config=load_settings(path)
        if dataset_name!=self.config['dataset_name']or str(Path(dataset_root).resolve())!=self.config['dataset_root']:raise ValueError('geometry dataset identity mismatch')
        meta_path=checked(self.config['cache_metadata'],self.config['cache_metadata_sha256']);self.meta=json.loads(meta_path.read_text())
        if self.meta.get('schema_version')!=1 or self.meta.get('relative_frame')!='left_tcp'or self.meta.get('rotation_format')!='matrix3x3'or self.meta['tcp_offset_m']!=self.config['tcp_offset_m']or self.meta['spec_sha256']!=self.config['spec_sha256']:raise ValueError('geometry cache convention mismatch')
        checked(self.config['spec_path'],self.config['spec_sha256'])
        self.arrays=dict(np.load(checked(meta_path.parent/self.meta['arrays'],self.meta['arrays_sha256']),allow_pickle=False))
        self.index={}
        for i,(ep,f)in enumerate(zip(self.arrays['episode_index'],self.arrays['frame_index'])):
            key=(int(ep),int(f))
            if key in self.index:raise ValueError('duplicate pose coordinate')
            self.index[key]=i
        self.episodes=set(ep for ep,f in self.index)
        for prefix in ['action','state']:
            if self.arrays[prefix+'_tcp'].shape!=(len(self.index),2,4,4)or self.arrays[prefix+'_rel'].shape!=(len(self.index),4,4):raise ValueError('pose cache shape mismatch')
            if not np.isfinite(self.arrays[prefix+'_tcp']).all()or not np.isfinite(self.arrays[prefix+'_rel']).all():raise ValueError('nonfinite pose cache')
        if self.arrays['stage'].shape!=(len(self.index),2)or self.arrays['valid'].shape!=(len(self.index),2):raise ValueError('pose stage/valid shape')
        if (self.arrays['stage']<0).any()or(self.arrays['stage']>=len(STAGES)).any():raise ValueError('invalid cached stage')
    def sample(self,episode,frame,horizon,padding,from_state=False):
        if len(padding)!=horizon:raise ValueError('geometry pad horizon')
        tcp=np.broadcast_to(np.eye(4,dtype=np.float32),(horizon,2,4,4)).copy();rel=np.broadcast_to(np.eye(4,dtype=np.float32),(horizon,4,4)).copy();valid=np.zeros((horizon,2),bool);gates=np.zeros((horizon,2,2),np.float32)
        prefix='state'if from_state else 'action';offset=int(bool(from_state))
        for k,pad in enumerate(padding):
            key=(int(episode),int(frame)+offset+k);idx=self.index.get(key)
            if pad:continue
            if idx is None:
                if int(episode)in self.episodes:raise ValueError('missing unpadded pose label: '+str(key))
                continue
            tcp[k]=self.arrays[prefix+'_tcp'][idx];rel[k]=self.arrays[prefix+'_rel'][idx];valid[k]=self.arrays['valid'][idx]
            for arm in [0,1]:
                stage=STAGES[int(self.arrays['stage'][idx,arm])]
                for branch,name in enumerate(['absolute','relative']):
                    section=self.config[name];selected=section.get('stages',[])
                    gates[k,arm,branch]=section.get('stage_weights',{}).get(stage,1.)if section['enabled']and(not selected or stage in selected)else 0.
        return dict(tcp_target_pose=tcp,tcp_target_relative=rel,tcp_target_valid=valid,tcp_geometry_gate=gates)

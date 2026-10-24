"""Offline absolute/relative TCP labels for both action and next-state action clocks."""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.grasp_sequence import _rpy,_T,forward_kinematics
STAGES=('idle_or_prepare','reach_aim','fine_align','pre_contact','gripper_close','grasp_lift','transport','target_align','target_hold','unknown')
def fk(q,spec,offset=.12):
    q=np.asarray(q,dtype=np.float64)
    n=len(q);T=np.broadcast_to(_T(_rpy(spec['root']['rpy']),spec['root']['translation']),(n,4,4)).copy()
    for i,j in enumerate(spec['joints']):
        axis=np.array(j['axis'],float);axis/=np.linalg.norm(axis);x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);c=np.cos(q[:,i])[:,None,None];s=np.sin(q[:,i])[:,None,None];R=c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*K
        T=T@_T(_rpy(j['origin_rpy']),j['origin_xyz'])@_T(R,np.zeros((n,3)))
    T=T@_T(_rpy(spec['tip_rpy']),spec['tip_xyz']);T[:,:3,3]+=T[:,:3,0]*offset
    pos,direction=forward_kinematics(q,spec,tcp_offset_m=offset);assert np.allclose(T[:,:3,3],pos,atol=1e-12)and np.allclose(T[:,:3,0],direction,atol=1e-12)
    return T

def relative(tcp):
    L,R=tcp[:,0],tcp[:,1];out=np.broadcast_to(np.eye(4),L.shape).copy();rot=L[:,:3,:3].transpose(0,2,1);out[:,:3,:3]=rot@R[:,:3,:3];out[:,:3,3]=(rot@(R[:,:3,3]-L[:,:3,3])[...,None])[...,0];return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():raise ValueError('refuse overwrite frozen geometry cache')
    import pyarrow.dataset as ds
    root=a.dataset_root.resolve();task='pick_dual_bottles';m=json.loads((ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())['tasks'][task];ids=sorted(m['train_ids']+m['val_ids']);rows=ds.dataset(str(root/'data'),format='parquet').to_table(columns=['episode_index','frame_index','action','observation.state'],filter=ds.field('episode_index').isin(ids),use_threads=False).to_pylist();rows.sort(key=lambda v:(v['episode_index'],v['frame_index']))
    source=ROOT/'refs/data_selection/pick_dual_bottles_full_stage_weight20_v1/segments.json';stages=json.loads(source.read_text());spec_path=ROOT/'configs/kinematics/robotwin_aloha_agilex_ee.json';spec=json.loads(spec_path.read_text());stage_map={};lengths={}
    for e in stages['episodes']:
        ep=e['episode_index'];lengths[ep]=e['length']
        for arm,segs in e['arms'].items():
            for seg in segs:
                for f in range(seg['start'],seg['end']):stage_map[ep,f,arm]=STAGES.index(seg['stage'])
    assert set(v['episode_index']for v in rows)==set(ids)and len(rows)==sum(lengths.values())
    arrays=dict(episode_index=np.array([v['episode_index']for v in rows]),frame_index=np.array([v['frame_index']for v in rows]),stage=np.array([[stage_map[v['episode_index'],v['frame_index'],arm]for arm in ['left','right']]for v in rows],dtype=np.int8))
    for ep,n in lengths.items():assert [v['frame_index']for v in rows if v['episode_index']==ep]==list(range(n))
    arrays['valid']=np.ones((len(rows),2),dtype=bool)
    errors={}
    for column,prefix in [('action','action'),('observation.state','state')]:
        q=np.array([v[column]for v in rows],dtype=np.float64);assert q.shape==(len(rows),14)and np.isfinite(q).all()
        tcp=np.stack([fk(q[:,:6],spec['left']),fk(q[:,7:13],spec['right'])],axis=1);rel=relative(tcp)
        err=float(np.max(np.abs(tcp[:,0]@rel-tcp[:,1])));assert err<1e-10
        for R in [tcp[...,:3,:3],rel[...,:3,:3]]:assert np.allclose(np.swapaxes(R,-1,-2)@R,np.eye(3),atol=1e-10)and np.allclose(np.linalg.det(R),1,atol=1e-10)
        W=np.eye(4);W[:3,:3]=_rpy(np.array([.2,.3,.4]));W[:3,3]=[1,2,3];assert np.allclose(relative(W@tcp),rel,atol=1e-10)
        arrays[prefix+'_tcp']=tcp.astype(np.float32);arrays[prefix+'_rel']=rel.astype(np.float32);arrays[prefix+'_joints']=q[:,[0,1,2,3,4,5,7,8,9,10,11,12]].astype(np.float32);errors[prefix]=err
    a.output_dir.mkdir(parents=True);np.savez_compressed(a.output_dir/'poses.npz',**arrays);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta=dict(schema_version=1,task=task,dataset_root=str(root),dataset_name='robotwin',fps=json.loads((root/'meta/info.json').read_text())['fps'],tcp_offset_m=.12,rotation_format='matrix3x3',relative_frame='left_tcp',source_columns=['action','observation.state'],arrays='poses.npz',arrays_sha256=sha(a.output_dir/'poses.npz'),spec_sha256=sha(spec_path),stage_source_sha256=sha(source),generator_sha256=sha(__file__),numeric_source_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),episodes=50,frames=len(rows),identity_max_error=errors)
    (a.output_dir/'cache_metadata.json').write_text(json.dumps(meta,indent=2))
    config=dict(version=1,dataset_name='robotwin',dataset_root=str(root),spec_path=str(spec_path),spec_sha256=sha(spec_path),cache_metadata='cache_metadata.json',cache_metadata_sha256=sha(a.output_dir/'cache_metadata.json'),tcp_offset_m=.12,huber_delta=.01,time_max=1.,absolute=dict(enabled=False,position_weight=1.,rotation_weight=.1,stages=[],stage_weights={}),relative=dict(enabled=False,position_weight=.1,rotation_weight=.01,stages=['fine_align','pre_contact','gripper_close','grasp_lift'],stage_weights={}))
    (a.output_dir/'geometry_config.disabled.json').write_text(json.dumps(config,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()

import os, json, tempfile, hashlib, ast, types
from pathlib import Path
from _paths import OUTPUT_ROOT, ROBOTWIN_ROOT, DATASET_ROOT
import xml.etree.ElementTree as ET
import numpy as np
import h5py, yaml, transforms3d as t3d
import sapien.core as sapien
os.environ['CUDA_VISIBLE_DEVICES']=''
asset=ROBOTWIN_ROOT/'assets/embodiments/aloha-agilex'
cfg=yaml.safe_load((asset/'config.yml').read_text())
urdf=asset/cfg['urdf_path']
original=ET.parse(urdf)
headless=ET.parse(urdf)
for link in headless.getroot().findall('link'):
 for tag in ('visual','collision'):
  for element in link.findall(tag): link.remove(element)
out=OUTPUT_ROOT
out.mkdir(parents=True,exist_ok=True)
headless.write(out/'headless.urdf')
scene=sapien.Scene([sapien.physx.PhysxCpuSystem()])
loader=scene.create_urdf_loader(); loader.fix_root_link=True
entity=loader.load(str(out/'headless.urdf'))
pose=cfg['robot_pose'][0]
entity.set_root_pose(sapien.Pose(pose[:3],pose[3:]))
entity.set_qpos(np.zeros(entity.dof))
active=[j.name for j in entity.get_active_joints()]
root_matrix=entity.get_root_pose().to_transformation_matrix().astype(float)
child_joints={j.find('child').get('link'):j for j in original.getroot().findall('joint')}
by_name={j.get('name'):j for j in original.getroot().findall('joint')}
def matrix(element):
 orig=element.find('origin')
 xyz=[float(x) for x in orig.get('xyz','0 0 0').split()] if orig is not None else [0,0,0]
 rpy=[float(x) for x in orig.get('rpy','0 0 0').split()] if orig is not None else [0,0,0]
 m=np.eye(4);m[:3,:3]=t3d.euler.euler2mat(*rpy,axes='sxyz');m[:3,3]=xyz
 return m
def components(m):return m[:3,3].tolist(), list(t3d.euler.mat2euler(m[:3,:3],axes='sxyz'))
spec={}
for side, names, ee_name in zip(('left','right'),cfg['arm_joints_name'],cfg['ee_joints']):
 path=[];joint=by_name[names[-1]]
 while joint is not None:
  path.append(joint)
  joint=child_joints.get(joint.find('parent').get('link'))
 path.reverse()
 joints=[];pending=np.eye(4)
 for j in path:
  pending=pending@matrix(j)
  if j.get('type')=='fixed':continue
  if j.get('name') not in names: raise RuntimeError('unaccounted movable joint '+j.get('name'))
  xyz,rpy=components(pending)
  joints.append(dict(name=j.get('name'),origin_xyz=xyz,origin_rpy=rpy,axis=[float(x) for x in j.find('axis').get('xyz').split()]))
  pending=np.eye(4)
 assert [j['name'] for j in joints]==names
 ee=entity.find_joint_by_name(ee_name)
 # RoboTwin reads the joint frame; include its fixed transform from child link.
 tip=ee.get_pose_in_child().to_transformation_matrix().astype(float)
 correction=np.eye(4); correction[:3,:3]=np.array(cfg['global_trans_matrix'])@np.array(cfg['delta_matrix'])
 shift=np.eye(4);shift[0,3]=cfg['gripper_bias']-.12
 tip=tip@correction@shift
 xyz,rpy=components(root_matrix);tx,tr=components(tip)
 spec[side]=dict(root=dict(translation=xyz,rpy=rpy),joints=joints,tip_xyz=tx,tip_rpy=tr)
 print(side,'names',names,'tip_xyz',tx,'tip_rpy',tr,flush=True)
spec['source']={'urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((asset/'config.yml').read_bytes()).hexdigest(),'frame':'RoboTwin get_*_ee_pose (joint global frame plus configured correction)','validation':'pending numerical parity'}
(out/'candidate_spec.json').write_text(json.dumps(spec,indent=2))
# Execute the exact RoboTwin accessors without importing planners/renderers.
source=(ROBOTWIN_ROOT/'envs/robot/robot.py').read_text()
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Robot')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_trans_endpose','get_left_ee_pose','get_right_ee_pose')]
ns=dict(np=np,t3d=t3d)
exec(compile(ast.Module(body=methods,type_ignores=[]),'<real_robot_accessors>','exec'),ns)
robot=types.SimpleNamespace()
for side,ee_name in zip(('left','right'),cfg['ee_joints']):
 setattr(robot,side+'_ee',entity.find_joint_by_name(ee_name))
 for key in ('gripper_bias','global_trans_matrix','delta_matrix'): setattr(robot,side+'_'+key,np.array(cfg[key]) if 'matrix' in key else cfg[key])
robot._trans_endpose=types.MethodType(ns['_trans_endpose'],robot)
indices=[[active.index(n) for n in names] for names in cfg['arm_joints_name']]
rng=np.random.default_rng(17)
samples=[];labels=[]
for i in range(100):samples.append(rng.uniform(-.8,.8,12));labels.append('random:'+str(i))
for task in ('place_object_stand','put_object_cabinet'):
 paths=sorted((DATASET_ROOT/task/'aloha_agilex/data').glob('*.hdf5'))
 for path in paths[::max(1,len(paths)//5)][:5]:
  with h5py.File(path) as f:
   values=f['action/joint_states'] if 'action/joint_states' in f else f['joint_action/vector']
   for idx in np.unique(np.linspace(0,len(values)-1,10,dtype=int)):
    v=np.asarray(values[idx]);samples.append(np.r_[v[:6],v[7:13]]);labels.append(task+':'+path.name+':'+str(idx))
expected=[]
for sample in samples:
 q=np.zeros(entity.dof)
 q[indices[0]]=sample[:6];q[indices[1]]=sample[6:]
 entity.set_qpos(q)
 poses=[]
 for side in ('left','right'):
  p=ns['get_'+side+'_ee_pose'](robot)
  m=np.eye(4);m[:3,:3]=t3d.quaternions.quat2mat(p[3:]);m[:3,3]=p[:3]
  poses.append(m)
 expected.append(poses)
np.savez(out/'sapien_reference.npz',joints=np.asarray(samples),expected=np.asarray(expected),labels=np.asarray(labels))
print('EXPORTED',len(samples),'samples',out,flush=True)

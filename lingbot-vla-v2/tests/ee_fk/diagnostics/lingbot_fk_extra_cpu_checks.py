import sys, os, json, ast
from pathlib import Path
from _paths import PACKAGE_ROOT
import torch
from lingbotvla.losses.ee_fk import DifferentiableEEFKSolver, compute_ee_fk_losses, _apply_norm
sys.path.insert(0,str(PACKAGE_ROOT/'tests/ee_fk'))
from test_ee_fk import _chain
layout={'left_arm':[0,6],'right_arm':[6,12],'gripper':[28,30]}
kwargs=dict(action_layout=layout,position_weight=1.,orientation_weight=.1,gripper_weight=.1)
for norm_type in ['identity','bounds_99_woclip','meanstd']:
 spec=[] if norm_type=='identity' else [dict(key='action.arm.position',start=0,end=12,norm_type=norm_type,mean=[0.1]*12,std=[0.5]*12,q01=[-1.]*12,q99=[1.]*12)]
 p=torch.randn(2,4,30,requires_grad=True)
 gt=torch.randn_like(p,requires_grad=True)
 solver=DifferentiableEEFKSolver({'left':_chain(),'right':_chain()})
 try:
  loss,m=compute_ee_fk_losses(solver,p,gt,action_norm_spec=spec,valid_time=torch.tensor([[True,False,True,False],[False]*4]),**kwargs)
  loss.backward()
  print('norm_gradient',norm_type,'finite',torch.isfinite(p.grad).all().item(),'gt_detached',gt.grad is None,'masked_zero',bool((p.grad[1]==0).all()),flush=True)
 except Exception as e: print('norm_gradient',norm_type,type(e).__name__,str(e),flush=True)
solver=DifferentiableEEFKSolver({'left':_chain(),'right':_chain()}).to(torch.bfloat16)
try:
 loss,m=compute_ee_fk_losses(solver,torch.randn(2,4,30),torch.randn(2,4,30),action_norm_spec=[],**kwargs)
 print('bf16_buffers_loss',float(loss),flush=True)
except Exception as e: print('bf16_buffers_loss',type(e).__name__,str(e),flush=True)
# Check the real implementation of Normalizer without importing image libraries.
source=(PACKAGE_ROOT/'lingbotvla/data/vla_data/transform.py').read_text()
tree=ast.parse(source)
body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dict_apply' or isinstance(n,ast.ClassDef) and n.name=='Normalizer']
import numpy as np
ns=dict(Dict=dict,np=np,torch=torch)
exec(compile(ast.Module(body=body,type_ignores=[]),'<normalizer>','exec'),ns)
stats=json.loads((PACKAGE_ROOT/'assets/norm_stats/robotwin.json').read_text())['norm_stats']
for key in ['action.arm.position','action.effector.position']:
 n=ns['Normalizer']({key:stats[key]},{key:'bounds_99_woclip'})
 value=torch.linspace(-1,1,steps=4*len(stats[key]['mean'][0])).reshape(4,-1) if isinstance(stats[key]['mean'][0],list) else torch.linspace(-1,1,steps=4*len(stats[key]['mean'])).reshape(4,-1)
 expected=n.unnormalize({key:value})[key]
 actual=_apply_norm(value[None],dict(stats[key],norm_type='bounds_99_woclip'))[0]
 print('actual_normalizer_parity',key,'max_abs_error',float(torch.max(torch.abs(actual-expected))),flush=True)

import json, sys
from pathlib import Path
import numpy as np
import torch

root=Path('/data/experiments/learn_skill_mix_task/lingbot-vla-v2');sys.path.insert(0,str(root))
from lingbotvla.losses.ee_fk import EEFKRuntime, compute_ee_fk_losses, _rotation_error
torch.backends.cuda.matmul.allow_tf32=False
spec=json.loads((root/'configs/kinematics/robotwin_aloha_agilex_ee.json').read_text())
ref=np.load(root/'tests/ee_fk/evidence/2026-10-17/sapien_reference.npz')
q=torch.tensor(ref['joints'],dtype=torch.float32,device='cuda')
expected=torch.tensor(ref['expected'],dtype=torch.float32,device='cuda')
runtime=EEFKRuntime(spec);solver=runtime.solver('cuda')
with torch.autocast('cuda',enabled=False):
 result=solver(q[:,:6],q[:,6:])
report={'samples':len(q),'device':torch.cuda.get_device_name(0),'arms':{}}
for i,side in enumerate(('left','right')):
 position,rotation=result[side]
 pe=float((position-expected[:,i,:3,3]).norm(dim=-1).max())
 re=float(_rotation_error(rotation,expected[:,i,:3,:3]).max())
 report['arms'][side]={'position_max_m':pe,'orientation_max_rad':re,'dtype':str(position.dtype)}
 assert pe<1e-5 and re<1e-4 and position.dtype==torch.float32
prediction=torch.randn(2,4,55,device='cuda',dtype=torch.bfloat16,requires_grad=True)
target=torch.randn(2,4,55,device='cuda',requires_grad=True)
with torch.autocast('cuda',dtype=torch.bfloat16):
 loss,_=compute_ee_fk_losses(solver,prediction,target,
  action_layout={'left_arm':[0,6],'right_arm':[6,12],'gripper':[28,30]},action_norm_spec=[],
  position_weight=1.,orientation_weight=.1,gripper_weight=.1,
  valid_time=torch.tensor([[True,False,True,False],[False]*4],device='cuda'))
loss.backward()
report.update(loss_dtype=str(loss.dtype),gradient_finite=bool(torch.isfinite(prediction.grad).all()),
 gradient_nonzero=bool((prediction.grad!=0).any()),gt_detached=target.grad is None,
 masked_gradient_zero=bool((prediction.grad[1]==0).all()))
assert report['gradient_finite'] and report['gradient_nonzero'] and report['gt_detached'] and report['masked_gradient_zero']
Path('/tmp/lingbot_ee_cuda_math_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)

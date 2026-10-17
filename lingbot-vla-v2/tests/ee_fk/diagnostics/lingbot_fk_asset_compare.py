import json
from pathlib import Path
from _paths import OUTPUT_ROOT
import numpy as np
import torch
from lingbotvla.losses.ee_fk import DifferentiableEEFKSolver, _rotation_error
root=OUTPUT_ROOT
spec=json.loads((root/'candidate_spec.json').read_text())
ref=np.load(root/'sapien_reference.npz')
solver=DifferentiableEEFKSolver(spec)
q=torch.tensor(ref['joints'],dtype=torch.float32)
expected=torch.tensor(ref['expected'],dtype=torch.float32)
actual=solver(q[:,:6],q[:,6:])
report={'samples':len(q),'arms':{}}
for i,side in enumerate(('left','right')):
 p,r=actual[side]
 pe=torch.linalg.vector_norm(p-expected[:,i,:3,3],dim=-1).numpy()
 re=_rotation_error(r,expected[:,i,:3,:3]).numpy()
 report['arms'][side]={'position_m':{'mean':float(pe.mean()),'p95':float(np.quantile(pe,.95)),'max':float(pe.max())},'angle_rad':{'mean':float(re.mean()),'p95':float(np.quantile(re,.95)),'max':float(re.max())}}
report['passed']=all(v['position_m']['max']<1e-5 and v['angle_rad']['max']<1e-4 for v in report['arms'].values())
# Real-chain gradient and padding check with trajectory joint targets.
pred=torch.zeros(2,4,30,requires_grad=True)
gt=torch.zeros_like(pred)
gt[:,:,:12]=q[100:108].reshape(2,4,12)
from lingbotvla.losses.ee_fk import compute_ee_fk_losses
loss,metrics=compute_ee_fk_losses(solver,pred,gt,action_layout={'left_arm':[0,6],'right_arm':[6,12],'gripper':[28,30]},action_norm_spec=[],position_weight=1,orientation_weight=.1,gripper_weight=.1,valid_time=torch.tensor([[True,True,False,False],[True,True,True,True]]))
loss.backward()
report['real_chain_gradient_finite']=bool(torch.isfinite(pred.grad).all())
report['real_chain_gradient_nonzero']=bool((pred.grad!=0).any())
report['masked_gradient_zero']=bool((pred.grad[0,2:]==0).all())
(root/'parity_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)

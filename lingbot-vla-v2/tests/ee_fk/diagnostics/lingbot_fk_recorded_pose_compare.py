import json,numpy as np,torch
from pathlib import Path
from _paths import OUTPUT_ROOT
from lingbotvla.losses.ee_fk import DifferentiableEEFKSolver,_rotation_error
root=OUTPUT_ROOT
spec=json.loads((root/'candidate_spec.json').read_text())
ref=np.load(root/'recorded_reference.npz')
q=torch.tensor(ref['joints'],dtype=torch.float32)
result=DifferentiableEEFKSolver(spec)(q[:,:6],q[:,6:])
report={'samples':len(q),'quaternion_convention':'wxyz','arms':{}}
# Quaternion->matrix independent of FK implementation.
for idx,side in enumerate(('left','right')):
 p,r=result[side];gt=torch.tensor(ref['poses'][:,idx],dtype=torch.float32)
 quat=gt[:,3:];quat=quat/quat.norm(dim=-1,keepdim=True)
 w,x,y,z=quat.unbind(-1)
 rot=torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
 pe=(p-gt[:,:3]).norm(dim=-1).numpy();re=_rotation_error(r,rot).numpy()
 report['arms'][side]={'position_m':{'mean':float(pe.mean()),'p95':float(np.quantile(pe,.95)),'max':float(pe.max())},'angle_rad':{'mean':float(re.mean()),'p95':float(np.quantile(re,.95)),'max':float(re.max())}}
report['passed']=all(v['position_m']['max']<1e-4 and v['angle_rad']['max']<1e-3 for v in report['arms'].values())
(root/'recorded_parity_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)

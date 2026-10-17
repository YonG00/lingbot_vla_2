import json,torch,numpy as np
from pathlib import Path
from _paths import OUTPUT_ROOT
from lingbotvla.losses.ee_fk import DifferentiableEEFKSolver,_rotation_error
root=OUTPUT_ROOT
spec=json.loads((root/'candidate_spec.json').read_text())
ref=np.load(root/'sapien_reference.npz');q=torch.tensor(ref['joints'],dtype=torch.float32)
a=DifferentiableEEFKSolver(spec)(q[:,:6],q[:,6:])
b=DifferentiableEEFKSolver(spec).bfloat16().float()(q[:,:6],q[:,6:])
report={side:{'position_roundtrip_max_m':float((a[side][0]-b[side][0]).norm(dim=-1).max()),'angle_roundtrip_max_rad':float(_rotation_error(a[side][1],b[side][1]).max())} for side in ('left','right')}
(root/'precision_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

import argparse, json, math, os, re, struct
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','reload','formal']);p.add_argument('--verify',action='store_true');a=p.parse_args()
job=json.loads(Path('/tmp/lingbot_ee_gpu_'+a.stage+'_job.json').read_text())
root=Path(job['output']);log=Path(job['log']).read_text(errors='replace')
print('RUN='+str(root),flush=True)
for line in log.replace('\r','\n').splitlines():
 if any(key in line for key in ('[ee_fk]','[target_change','Step:','Reached max_steps','Training completed','Traceback','Error:','HF 直出完成','hf_only')):
  print(line[:1300],flush=True)
events=root/'auto_learning_events.jsonl';latest={}
if events.exists():
 for line in events.read_text(errors='replace').splitlines():
  try:row=json.loads(line)
  except ValueError:continue
  if 'ee_' in str(row.get('name','')):latest[row['name']]=(row.get('step'),row.get('value'))
print('EE_EVENTS='+json.dumps(latest),flush=True)
checkpoints=sorted((root/'checkpoints').glob('global_step_*/hf_ckpt'))
print('HF_CHECKPOINTS='+str([str(x) for x in checkpoints]),flush=True)
if a.verify:
 if 'Traceback (most recent call last)' in log:raise RuntimeError('trainer traceback')
 expected_steps=3 if a.stage=='smoke' else (1 if a.stage=='reload' else 500)
 if not (f'Reached max_steps={expected_steps}' in log or 'Training completed' in log):
  raise RuntimeError('trainer has not reached expected completion')
 if not latest or not all(math.isfinite(float(pair[1])) for pair in latest.values()):
  raise RuntimeError('EE val events missing/nonfinite')
 if a.stage=='smoke':
  if not all((root/'checkpoints'/f'global_step_{step}'/'hf_ckpt/model.safetensors.index.json').is_file() for step in (2,3)):
   raise RuntimeError('periodic/final HF missing')
  for model in checkpoints:
   cfg=json.loads((model/'config.json').read_text())
   if not cfg.get('ee_fk_spec') or not cfg.get('ee_fk_loss_enabled'):raise RuntimeError('HF config lost FK metadata')
   index=json.loads((model/'model.safetensors.index.json').read_text())
   if any('ee_fk_runtime' in name for name in index['weight_map']):raise RuntimeError('runtime geometry leaked into weights')
   dtypes=set()
   for name in set(index['weight_map'].values()):
    with (model/name).open('rb') as stream:
     length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
    dtypes.update(row['dtype'] for key,row in header.items() if key!='__metadata__')
   if any(dtype.startswith('F') and dtype!='BF16' for dtype in dtypes):raise RuntimeError('unexpected floating point HF dtype')
   if any((model.parent/name).exists() for name in ('model','optimizer')):raise RuntimeError('unexpected DCP')
   print('HF_VERIFIED='+json.dumps(dict(path=str(model),dtypes=sorted(dtypes),tensor_count=len(index['weight_map']))),flush=True)
 # Pure event inspection after the training process has finished.
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 files=list(root.rglob('events.out.tfevents.*'))
 values={}
 for parent in {file.parent for file in files}:
  accumulator=EventAccumulator(str(parent)).Reload()
  for tag in accumulator.Tags().get('scalars',[]):
   if tag.startswith('ee_fk/'):
    points=accumulator.Scalars(tag)
    if not all(math.isfinite(point.value) for point in points):raise RuntimeError('nonfinite training EE loss')
    values[tag]=[(point.step,point.value) for point in points]
 if not values:raise RuntimeError('training EE scalar metrics missing')
 if not any(value>0 for _,value in values.get('ee_fk/weighted_loss',[])):
  raise RuntimeError('training EE loss never contributed a positive term')
 print('EE_TRAIN_TB='+json.dumps(values),flush=True)
 print('VERIFIED='+a.stage,flush=True)

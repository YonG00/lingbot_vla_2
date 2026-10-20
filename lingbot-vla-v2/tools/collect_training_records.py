import json,re
from pathlib import Path
base=Path('/data/outputs')
names=['official_adjust_bottle_9680_m28_add500_20261020','enhanced_adjust_bottle_9680_m28_add500_20261020','official_adjust_bottle_9680_bf16_m56_add200_20261020','three_task_from_adjust_bf16_200_m56_add400_20261020','place_container_plate_from_adjust_bf16_200_m56_add200_20261020','sequential_plate100_cup200_pot200_from_plate200_bf16_m56_20261020']
records=[]
names.append('sequential_switch_pillbottle_stapler_from_R6_bf16_m56_200each_20261020')
for name in names:
 p=base/name
 for out in [p]+list(p.glob('*remediation_round_[0-9]')):
  row=dict(run=str(out.resolve()),files={})
  for f in [out/'trainer_config.json',out/'recipe.json',out.with_name(out.name+'_pipeline')/'status.json']+list(out.glob('evaluation*/**/summary.json'))+list(out.glob('evaluation*/summary.json')):
   if f.exists() and str(f) not in row['files']:
    value=json.loads(f.read_text())
    if f.name=='trainer_config.json':
     cfg=value.get('trainer_config',value);value=dict(source=cfg['model'].get('model_path'),train={k:cfg['train'].get(k) for k in ['max_steps','micro_batch_size','global_batch_size','save_steps']})
    row['files'][str(f.relative_to(out)) if f.is_relative_to(out) else str(f)]=value
  log=out/'train.log'
  if log.exists():
   text=log.read_text(errors='replace');matches=re.findall(r'(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d).*?Step (\d+)/(\d+)',text)
   row['log_progress']=dict(first=matches[0] if matches else None,last=matches[-1] if matches else None)
  records.append(row)
print(json.dumps(records,indent=2))

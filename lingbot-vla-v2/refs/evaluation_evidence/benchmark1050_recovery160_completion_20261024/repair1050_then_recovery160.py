import fcntl,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/data/experiments/stack_bowls_sample_weighting/lingbot-vla-v2')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from experiment.official_control.model_benchmark import run,atomic_json
SOURCE=Path('/data/eval_results/20261024_1923_1050_benchmark_clean50_random50_parallel4_inline_novideo')
RECOVERY_MODEL='/data/outputs/20261022_1128_recovery_scout3_from_R17_replay10pct/20261022_1209_round_3_recovery/checkpoints/global_step_160/hf_ckpt'
plan=json.loads(Path('/tmp/repair1050_then_recovery160_plan.json').read_text())
recovery=Path(plan['recovery_destination']);chain=Path(plan['chain_directory'])
status=chain/'status.json';source_status=Path(str(SOURCE)+'_pipeline/status.json')
recovery_pipeline=Path(str(recovery)+'_pipeline');recovery_pipeline.mkdir(parents=True,exist_ok=True)
recovery_status=recovery_pipeline/'status.json'
def state(value,**kw):atomic_json(status,dict(state=value,source=str(SOURCE),recovery=str(recovery),**kw))
def verify_model(path):
 path=Path(path);index=json.loads((path/'model.safetensors.index.json').read_text());count=0
 for filename in set(index['weight_map'].values()):
  p=path/filename
  with p.open('rb') as f:
   length=int.from_bytes(f.read(8),'little');header=json.loads(f.read(length))
  count+=sum(k!='__metadata__' for k in header)
 if count!=len(index['weight_map']):raise ValueError('checkpoint shard tensor count mismatch')
 return count
try:
 with (Path(str(SOURCE)+'_pipeline')/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  original=json.loads((SOURCE/'manifest.json').read_text())['config']
  os.environ['LINGBOT_ROBOTWIN_ROOT']=original['robotwin']
  os.environ['LD_LIBRARY_PATH']='/data/experiments/stack_bowls_sample_weighting/.runtime/oidn-2.5.1/oidn-2.5.1.x86_64.linux/lib:'+os.environ.get('LD_LIBRARY_PATH','')
  verify_model(RECOVERY_MODEL)
  atomic_json(recovery_status,dict(state='queued_after_1050_repair',model=RECOVERY_MODEL,episodes=400,workers=4))
  for attempt in range(1,4):
   state('repairing_1050',pass_number=attempt)
   atomic_json(source_status,dict(state='repairing_errors',chain=str(chain),pass_number=attempt))
   if run(original,resume=True):break
  else:raise RuntimeError('1050 remains incomplete after 3 repair passes; recovery not started')
  s=json.loads((SOURCE/'benchmark_results.json').read_text())
  if s['status']!='complete' or s['overall']['episode_count']!=400:raise ValueError('source result not complete')
  atomic_json(source_status,dict(state='complete',report=str(SOURCE/'benchmark_results.json')))
  seeds=json.loads((SOURCE/'selected_scene_seeds.json').read_text())
  if len(seeds)!=100 or any(len(v)!=4 or len(set(v))!=4 for v in seeds.values()):raise ValueError('source scene seed table incomplete')
  target=dict(original,model=RECOVERY_MODEL,destination=str(recovery),seed_selection='fixed',scene_seeds=seeds)
  atomic_json(chain/'recovery_job.json',target)
  with (recovery_pipeline/'launch.lock').open('a') as other:
   fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for attempt in range(1,4):
    state('evaluating_recovery160',pass_number=attempt)
    atomic_json(recovery_status,dict(state='evaluating',pass_number=attempt,source_seed_table=str(SOURCE/'selected_scene_seeds.json')))
    if run(target,resume=attempt>1):break
   else:raise RuntimeError('recovery benchmark incomplete after 3 passes with unchanged seeds')
  result=json.loads((recovery/'benchmark_results.json').read_text())
  if result['status']!='complete' or result['overall']['episode_count']!=400:raise ValueError('recovery result incomplete')
  for row in result['tasks']:
   key=row['task_config']+'/'+row['task']
   if row['seeds']!=seeds[key]:raise ValueError('paired seed mismatch: '+key)
  atomic_json(recovery_status,dict(state='complete',report=str(recovery/'benchmark_results.json')))
  comparison=dict(source=s['overall'],recovery=result['overall'],identical_scene_seeds=True,
                  source_report=str(SOURCE/'benchmark_results.json'),recovery_report=str(recovery/'benchmark_results.json'))
  atomic_json(chain/'comparison.json',comparison)
  state('complete',comparison=str(chain/'comparison.json'))
except BaseException as e:
 state('failed',error=repr(e));traceback.print_exc();raise

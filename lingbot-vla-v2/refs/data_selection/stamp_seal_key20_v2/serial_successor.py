"""One-shot serial successor; never changes or stops the predecessor."""
import os,sys,json,time,subprocess,shutil,hashlib
from pathlib import Path
r=Path('/data/experiments/stack_bowls_sample_weighting/lingbot-vla-v2');os.chdir(r);sys.path.insert(0,str(r))
queue=Path('/data/experiments/stack_bowls_sample_weighting/stamp_seal_after_can0915_queue');queue.mkdir(exist_ok=True)
status=queue/'status.json'
def state(value,**extra):status.write_text(json.dumps(dict(state=value,**extra),indent=2))
try:
 predecessor=Path('/data/outputs/20261024_0915_can_pot_add200_from_0844_s200_bf16_m56_approach_grasp20_weightnorm_noreplay_parallel3_video_pipeline/status.json');deadline=time.monotonic()+86400
 state('waiting_for_predecessor',predecessor=str(predecessor))
 while True:
  info=json.loads(predecessor.read_text())if predecessor.exists()else {}
  if info.get('state')=='complete':break
  if info.get('state')in('failed','stopped','interrupted'):raise RuntimeError('predecessor not complete: '+json.dumps(info))
  if time.monotonic()>deadline:raise TimeoutError('predecessor wait exceeded 24h')
  time.sleep(30)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'GPU still busy after predecessor'
 assert shutil.disk_usage('/data/outputs').free>=16.06*1024**3,'disk below final-checkpoint safety budget 16.06GiB'
 from experiment.official_control.baseline_pipeline import verify_checkpoint
 recipe=json.loads((r/'configs/training_recipes/stamp_seal200_from_can0915_key20_noreplay.json').read_text());source=verify_checkpoint(Path(recipe['source']['checkpoint']));assert source['tensors']==1708
 state('expert_seed_validation',source=source)
 env=dict(os.environ,PYTHONPATH=str(r),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',LINGBOT_ROBOTWIN_ROOT='/data/experiments/official_adjust_bottle_control/RoboTwin-lingbot',LINGBOT_SAMPLE_WEIGHT_CONFIG=str(r/'refs/data_selection/stamp_seal_key20_v2/weight_config.json'))
 with(queue/'expert_seeds.log').open('x')as log:
  subprocess.run(['/data/miniconda3/envs/RoboTwin/bin/python','-u','/tmp/validate_stamp_seeds.py'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
 report=json.loads((queue/'expert_seeds.json').read_text());assert len(report['validated_seeds'])==3;recipe['evaluation']['single_task_episode_seeds']=report['validated_seeds']
 resolved=queue/'resolved_recipe.json';resolved.write_text(json.dumps(recipe,indent=2));state('launching_with_required_cpu_preflight',seeds=report['validated_seeds'],source=source)
 with(queue/'launch.log').open('x')as log:
  subprocess.run([sys.executable,'-u','experiment/task_recipes/run.py','--recipe',str(resolved),'--execute'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 state('successor_pipeline_launched',seeds=report['validated_seeds'],launch_log=str(queue/'launch.log'),source=source)
except BaseException as exc:
 state('blocked',error=repr(exc));raise

import importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
from types import SimpleNamespace as NS
from datetime import datetime

root=Path('/data/experiments/learn_skill_mix_task/lingbot-vla-v2')
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('ee_launcher',root/'experiment/robotwin/launch_skill_mix_task.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
out=Path('/data/outputs')/('ee_fk_gpu_smoke_'+stamp)
cfg=launcher.read_json('/data/al_cache/stand_target_change_v1/stand_al_config.json')
cfg.update(eval_interval_steps=3,min_steps_before_defer=9999,defer_retry_steps=99)
args=NS(model=Path('/data/outputs/al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt'),
 train_config=Path('/data/outputs/stand_targetchange_lnfm_9680_add500_20261017_r2/lingbotvla_cli.yaml'),
 dataset_root=Path('/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30'),
 qwen3vl=Path('/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct'),out=out,
 audit_report=Path('/data/al_cache/stand_target_change_v1/data_audit.json'),
 al_config=Path('/data/al_cache/stand_target_change_v1/stand_al_config.json'),al_config_data=cfg,
 manifest=root/'refs/al_artifacts/al/task_splits_50/manifest.json',learning_tasks=['place_object_stand'],
 replay_tasks=None,steps=3,micro=24,gas=1,new_ratio=1.,python=sys.executable,master_port=62531,
 ee_fk_loss=True,ee_fk_eval=True)
plan=launcher.build_plan(args)
free=shutil.disk_usage('/data/outputs').free
weight=sum(row['size'] for row in plan['checkpoint_inventory'])
if free < weight*4+3*1024**3:raise RuntimeError('insufficient disk for smoke and formal HFs')
out.mkdir(exist_ok=False)
for name,value in [('al_config.json',plan['config']),('trainer_config.json',plan['training_config']),
                   ('train_ids.json',plan['train_ids']),('provenance.json',plan)]:
 (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
(out/'datasets.txt').write_text('robotwin '+str(args.dataset_root)+'\n')
command=plan['command']+['--train.save_steps','2']
env=dict(os.environ,QWEN3VL_PATH=str(args.qwen3vl),PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='0',
 AL_POOL_STATE_NO_IMPORT='1',AL_POOL_STATE_MIRROR='',AL_SCAN_SKIP_HARDNESS='0',
 AL_HARDNESS_CACHE_FILE='',AL_HARDNESS_CACHE='',AL_SCOUT_CACHE_MODE='off',AL_SCOUT_CACHE_FILE='',
 AL_HARDNESS_SELFTEST_IDS='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
log=Path('/data/outputs/logs')/(out.name+'.log');log.parent.mkdir(exist_ok=True)
with log.open('xb') as stream:
 child=subprocess.Popen(['setsid','nohup',*command],cwd=root,env=env,stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT)
job=dict(output=str(out),log=str(log),pid=child.pid,command=command)
Path('/tmp/lingbot_ee_gpu_smoke_job.json').write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(job,indent=2),flush=True)

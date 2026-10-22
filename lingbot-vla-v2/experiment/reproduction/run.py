"""Three-stage curriculum. Preview by default; --execute launches a background controller."""
from copy import deepcopy
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.task_recipes.schema import validate, milestones
from experiment.task_recipes.recovery_cycle import historical_best, declines
from experiment.task_recipes.planning import digest


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def provenance(settings):
    import importlib.metadata
    paths=[ROOT/'configs/reproduction/three_stage_v1.json',ROOT/'configs/reproduction/trainer_bf16.json',
           ROOT/curriculum()['history_file'],ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json',
           Path(settings['norm_stats']),Path(settings['base_model'])/'config.json',
           Path(settings['base_model'])/'model.safetensors.index.json']
    versions={}
    for name in ['torch','transformers','safetensors','flash-attn']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='missing'
    return dict(python=sys.version,packages=versions,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
        small_file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})


def curriculum():
    c = load(ROOT / 'configs/reproduction/three_stage_v1.json')
    manifest = load(ROOT / 'refs/al_artifacts/al/task_splits_50/manifest.json')
    if len(c['task_order']) != 50 or set(c['task_order']) != set(manifest['tasks']):
        raise ValueError('curriculum must cover the exact 50 tasks once')
    return c


def recipe(train, source, template, out, settings, tests=()):
    c = dict(version=1, source=dict(trainer_config=str(template), checkpoint=str(source)),
             data=dict(dataset_root=settings['dataset_root']), output_dir=str(out), train=deepcopy(train),
             checkpoint=dict(mode='steps', every_steps=settings['save_every'], save_final=True, hf_dtype='bf16'),
             evaluation=dict(tasks=list(tests), checkpoints=['final'], episodes_per_task=3,
                             shared_model_session=True, max_parallel_tasks=settings['eval_workers'],
                             video_episodes=0, reuse_training_model=bool(tests)),
             remediation=dict(enabled=False), runtime=dict(shutdown_after_completion=False))
    c['train'].update(micro_batch_size=settings['micro'], gradient_accumulation_steps=1)
    return validate(c)


def sequential(tasks, steps, review=(), completed_only=False):
    return dict(mode='sequential', micro_batch_size=56, gradient_accumulation_steps=1, seed=3,
                episode_split='all', task_order=list(tasks),
                main_tasks={t:dict(steps=steps, weight=1) for t in tasks},
                replay=(dict(mode='round_robin', fraction=.1, tasks_per_step=1, switch_every_steps=5,
                             tasks={t:dict(weight=1) for t in review},
                             **({'completed_only':True} if completed_only else {})) if review else dict(fraction=0, tasks={})))


def plan(settings):
    c = curriculum(); source = Path(settings['output_dir']) / 'base_bf16'
    template = Path(settings['output_dir']) / 'trainer_template.json'
    root = Path(settings['output_dir']); jobs=[]
    jobs.append(('stage1_foundation', sequential(c['task_order'],100,c['task_order'],True), []))
    for row in c['stage2']:
        jobs.append(('stage2_'+row['label'],row['train'],[]))
    peaks = historical_best(load(ROOT/c['history_file']))
    names = sorted(peaks)
    if len(names)!=25:raise ValueError('frozen reference must contain 25 historically passed tasks')
    grouped=dict(mode='grouped',seed=3,episode_split='all',steps=c['stage3_group_steps'],tasks_per_group=3,
                 steps_per_group=3,main_tasks={t:dict(weight=1) for t in names},replay=dict(fraction=0,tasks={}))
    jobs.append(('stage3_grouped',grouped,[]))
    reports=[]
    for name,train,tests in jobs:
        r=recipe(train,source,template,root/name,settings,tests)
        reports.append(dict(name=name,recipe=r,steps=r['train']['steps'],saves=milestones(r),evaluation_tasks=tests))
        source=root/name/f"checkpoints/global_step_{r['train']['steps']}/hf_ckpt"
    steps=sum(j['steps'] for j in reports)
    return dict(jobs=reports,reference_tasks=names,scout_batches=[names[i:i+5] for i in range(0,25,5)],
                fixed_steps=steps,maximum_remediation_steps=25*c['stage3_remediation_steps'],
                fixed_hf_count=sum(len(j['saves']) for j in reports),
                note='checkpoint interval is per training job, with an additional final save; no automatic deletion')


def resolve_template(settings):
    text=(ROOT/'configs/reproduction/trainer_bf16.json').read_text()
    values=dict(BASE=str(Path(settings['output_dir'])/'base_bf16'),QWEN=settings['qwen_model'],
                ASSETS=settings['assets_root'],NORM=settings['norm_stats'],PACKAGE=str(ROOT),OUTPUT=settings['output_dir'])
    def replace(v):
        if isinstance(v,str):
            for key,value in values.items():v=v.replace('${'+key+'}',value)
            if '${' in v:raise ValueError('unresolved resource path '+v)
            return v
        if isinstance(v,list):return [replace(x) for x in v]
        if isinstance(v,dict):return {k:replace(x)for k,x in v.items()}
        return v
    return replace(json.loads(text))


def preflight(settings):
    from experiment.official_control.prepare_adjust_bottle import validate_identity
    validate_identity()
    base=Path(settings['base_model']); index=load(base/'model.safetensors.index.json')
    required=[base/'config.json',Path(settings['dataset_root'])/'meta',Path(settings['qwen_model']),Path(settings['norm_stats']),
              Path(settings['assets_root'])/'depth/moge2-vitb-normal.pt',Path(settings['assets_root'])/'depth/model.pt',
              Path(settings['assets_root'])/'dino_video/teacher_step_10000.pth',Path(settings['assets_root'])/'dino_video/config.yaml',
              ROOT.parent/'RoboTwin-lingbot/script/eval_policy_client_lingbotvla.py']
    for name in set(index['weight_map'].values()):
        p=base/name
        if p.resolve().parent!=base.resolve():raise ValueError('unsafe shard path')
        required.append(p)
    missing=[str(p)for p in required if not p.exists()]
    if missing:raise ValueError('missing resources: '+json.dumps(missing))
    output=Path(settings['output_dir'])
    if output==base or base in output.parents:raise ValueError('output must be outside the base-model directory')
    parent=output
    while not parent.exists():parent=parent.parent
    report=plan(settings)
    # Keep every authorized save, including the five possible final remediation saves.
    count=report['fixed_hf_count']+1+5
    required_bytes=int(sum((base/s).stat().st_size for s in set(index['weight_map'].values()))/2*count*1.1)+3*2**30
    if shutil.disk_usage(parent).free<required_bytes:
        raise ValueError(f'output needs worst-case safety budget {required_bytes/2**30:.2f} GiB; no automatic pruning')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
        raise RuntimeError('GPU busy; no concurrent training/evaluation allowed')
    subprocess.run([settings['sim_python'],'-c','import sapien,torch,numpy'],check=True)
    from tools.task_split import load_episodes
    load_episodes(settings['dataset_root'])
    # Validate the real dataset IDs and deployment dependencies before spending GPU time.
    from experiment.task_recipes.resident_evaluation import validate_preprocessing
    validate_preprocessing(resolve_template(settings))


def convert(base, destination):
    """Convert one CPU shard at a time; never load a full FP32 model or overwrite input."""
    import torch
    from safetensors.torch import load_file, save_file
    from experiment.official_control.baseline_pipeline import verify_checkpoint
    base=Path(base);destination=Path(destination)
    if destination.exists():
        report=load(destination/'conversion_complete.json')
        if report['source']!=str(base):raise ValueError('conversion source differs')
        verify_checkpoint(destination);return
    index=load(base/'model.safetensors.index.json')
    source_bytes=sum((base/s).stat().st_size for s in set(index['weight_map'].values()))
    if shutil.disk_usage(destination.parent).free<source_bytes//2+3*2**30:raise ValueError('insufficient conversion storage')
    partial=destination.with_name(destination.name+'.partial');partial.mkdir(exist_ok=False)
    total=0;seen={};shards=[]
    for name in sorted(set(index['weight_map'].values())):
        weights=load_file(str(base/name),device='cpu')
        floating={v.dtype for v in weights.values()if v.is_floating_point()}
        if floating and floating!={torch.float32}:raise ValueError('base floating weights must all be FP32')
        weights={k:v.to(torch.bfloat16)if v.is_floating_point()else v for k,v in weights.items()}
        total+=sum(v.numel()*v.element_size()for v in weights.values())
        for key in weights:
            if key in seen:raise ValueError('duplicate tensor')
            seen[key]=name
        save_file(weights,str(partial/name),metadata={'format':'pt'})
        saved=load_file(str(partial/name),device='cpu')
        if set(saved)!=set(weights) or any(not torch.equal(saved[k],weights[k]) for k in weights):
            raise ValueError('converted shard failed exact verification: '+name)
        shards.append(dict(name=name,source_bytes=(base/name).stat().st_size))
        del saved,weights
    if seen!=index['weight_map']:raise ValueError('base tensor index mismatch')
    index.setdefault('metadata',{})['total_size']=total
    dump(partial/'model.safetensors.index.json',index)
    for p in base.iterdir():
        if p.is_file() and p.name!='model.safetensors.index.json' and p.suffix in ['.json','.txt','.model','.jinja','.py']:
            shutil.copy2(p,partial/p.name)
    config=load(partial/'config.json');config['torch_dtype']='bfloat16';dump(partial/'config.json',config)
    verify_checkpoint(partial)
    dump(partial/'conversion_complete.json',dict(source=str(base),shards=shards,tensors=len(seen),dtype='BF16'))
    partial.rename(destination)


def execute(settings):
    from experiment.task_recipes.run import worker
    from experiment.official_control.baseline_pipeline import verify_checkpoint
    from experiment.official_control.parallel_evaluation import run_jobs
    root=Path(settings['output_dir']); root.mkdir(parents=True,exist_ok=True)
    lock=root/'controller.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        config_path=root/'settings.json'
        if config_path.exists() and load(config_path)!=settings:raise ValueError('resume settings differ; use the original paths/options')
        dump(config_path,settings)
        report=plan(settings);dump(root/'execution_plan.json',report)
        preflight(settings)
        dump(root/'environment.json',provenance(settings))
        os.environ['LINGBOT_SIM_PYTHON']=settings['sim_python']
        os.environ['LINGBOT_TENSORBOARD_RUN_PREFIX']=root.name
        convert(settings['base_model'],root/'base_bf16')
        template=root/'trainer_template.json'; dump(template,resolve_template(settings))
        source=root/'base_bf16';evaluations=[];completed=[]
        def state(value,**extra):dump(root/'status.json',dict(state=value,completed=completed,**extra))
        def train_job(name,train,tests):
            nonlocal source
            out=root/name;launch=out.with_name(out.name+'_pipeline')
            r=recipe(train,source,template,out,settings,tests)
            if (launch/'status.json').exists() and load(launch/'status.json').get('state')=='complete':
                if load(launch/'recipe.json')!=r:raise ValueError('completed recipe changed '+name)
                done=load(launch/'status.json')
            else:
                # Partial jobs are preserved; no optimizer state is silently reconstructed from HF.
                if launch.exists() or out.exists():raise RuntimeError('incomplete job '+name+'; preserve it and restart with a fresh output directory')
                launch.mkdir();dump(launch/'recipe.json',r);state('training',job=name)
                if worker(r,launch)!=0:raise RuntimeError('training job failed '+name)
                done=load(launch/'status.json')
            source=out/f"checkpoints/global_step_{r['train']['steps']}/hf_ckpt";verify_checkpoint(source)
            evaluations.extend(done['evaluation']);completed.append(name)
            dump(root/'training_ledger.json',dict(completed=completed,evaluations=evaluations,latest_model=str(source)))
            return done
        for item in report['jobs']:
            train_job(item['name'],item['recipe']['train'],item['evaluation_tasks'])
            if item['name']=='stage2_R16':
                dump(root/'stage2_recipe_complete.json',dict(training_complete=True,
                     evaluation_performed=False,reference_tasks=report['reference_tasks'],
                     reference_file=curriculum()['history_file']))
        # Frozen 25-task coverage: remedied tasks never remove another task from the scan.
        c=curriculum();peaks=historical_best(load(ROOT/c['history_file']));rounds=[]
        for number,tasks in enumerate(report['scout_batches'],1):
            folder=root/f'stage3_scan_{number}';done_path=root/f'stage3_scan_{number}_completed.json'
            if done_path.exists():
                round_report=load(done_path)
                if round_report['tasks']!=tasks or round_report['source']!=str(source):raise ValueError('scan source changed')
                results=round_report['results']
            else:
                if folder.exists():raise RuntimeError('incomplete scan preserved: '+str(folder))
                folder.mkdir();state('evaluating',batch=number,tasks=tasks)
                jobs=[dict(task=t,label=f'baseline_scan_{number}',model=str(source),destination=str(folder/t),episodes=3,
                           video_episodes=0,require_baseline=False,robotwin=str(ROOT.parent/'RoboTwin-lingbot'),
                           qwen=settings['qwen_model'],sim_python=settings['sim_python'])for t in tasks]
                results=run_jobs(jobs,folder,min(5,settings['eval_workers']))
                round_report=dict(tasks=tasks,source=str(source),results=results)
                dump(done_path,round_report)
            lessons=declines(results,peaks,c['stage3_remediation_steps'])
            round_report['lessons']=lessons
            if lessons:
                train_job(f'stage3_recovery_{number}',sequential([x['task']for x in lessons],c['stage3_remediation_steps'],report['reference_tasks']),[])
            rounds.append(round_report);dump(root/'recovery_ledger.json',rounds)
        dump(root/'final_model.json',dict(path=str(source),verified=verify_checkpoint(source),scan_tasks=report['reference_tasks'],
             note='25 tasks scanned across successive models; repaired tasks not retested; not a final-model 25-task success rate'))
        state('complete',latest_model=str(source),rounds=rounds)
    except BaseException as error:
        dump(root/'status.json',dict(state='failed',error=repr(error)));raise
    finally:
        lock.unlink(missing_ok=True)


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',type=Path,required=True,help='FP32 base HF checkpoint; converted automatically to a BF16 copy')
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--qwen-model',type=Path,help='default BASE/qwen; local Qwen3-VL processor assets')
    p.add_argument('--assets-root',type=Path,help='default BASE; contains depth/ and dino_video/')
    p.add_argument('--norm-stats',type=Path,default=ROOT/'assets/norm_stats/robotwin_competition_clean.json')
    p.add_argument('--sim-python',type=Path,default=Path(sys.executable),help='Python for prepared RoboTwin environment')
    p.add_argument('--save-every',type=int,default=2000)
    p.add_argument('--micro',type=int,default=56)
    p.add_argument('--eval-workers',type=int,choices=range(1,9),default=5)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--check-only',action='store_true',help='check resources/environment without conversion or training; GPU must be idle')
    p.add_argument('--resume',action='store_true',help='skip completed jobs; no exact mid-job optimizer resume')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.save_every<1 or a.micro<3:p.error('save interval must be positive and micro >=3')
    settings=dict(base_model=str(a.base_model.expanduser().resolve()),dataset_root=str(a.dataset_root.expanduser().resolve()),
                  output_dir=str(a.output_dir.expanduser().resolve()),qwen_model=str((a.qwen_model or a.base_model/'qwen').expanduser().resolve()),
                  assets_root=str((a.assets_root or a.base_model).expanduser().resolve()),norm_stats=str(a.norm_stats.expanduser().resolve()),
                  sim_python=str(a.sim_python.expanduser().absolute()),save_every=a.save_every,micro=a.micro,eval_workers=a.eval_workers)
    settings['definition_sha256']=digest(dict(curriculum=curriculum(),
        trainer=load(ROOT/'configs/reproduction/trainer_bf16.json'),
        reference=load(ROOT/curriculum()['history_file']),
        manifest=load(ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json')))
    return a,settings


def main(argv=None):
    a,s=arguments(argv)
    if a.worker:execute(load(a.worker));return
    report=plan(s)
    if a.check_only:
        preflight(s);print('REPRODUCTION_PREFLIGHT_PASS');return
    if not a.execute:
        print(json.dumps({k:v for k,v in report.items()if k!='jobs'},ensure_ascii=False,indent=2))
        for job in report['jobs']:print(job['name'],job['steps'],'saves',job['saves'])
        return
    root=Path(s['output_dir'])
    if (root/'controller.lock').exists():raise ValueError('controller lock exists; verify old PID has exited before manually removing a stale lock')
    if root.exists() and not a.resume:raise ValueError('output exists: use --resume for completed job boundaries')
    root.mkdir(parents=True,exist_ok=True)
    path=root/'requested_settings.json'
    if path.exists() and load(path)!=s:raise ValueError('resume settings differ')
    dump(path,s)
    # CLI returns promptly; all paid long work stays in a dedicated process group.
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker',str(path)]
    log=root/(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_controller.log')
    with log.open('x')as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,log=str(log),output_dir=str(root)),ensure_ascii=False))

if __name__=='__main__':main()

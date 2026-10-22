"""统一配方入口：默认预览；--execute明确后台运行，不改既有实验入口。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiment.task_recipes.prepare import preview,prepare
from experiment.task_recipes.runtime import install,load_run
from experiment.task_recipes.planning import confirmed_progress,probe_steps
from experiment.task_recipes.naming import dated_run_path


def worker(raw,launch):
    c,p,groups,frames=preview(raw);out=Path(c['output_dir'])
    def state(value,**data):
        (launch/'status.json').write_text(json.dumps(dict(state=value,**data),indent=2));print('RECIPE_STATE',value,data,flush=True)
    def run(command,log):
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            AL_DYNAMO_CACHE_SIZE_LIMIT='0',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='29556',PYTHONUNBUFFERED='1')
        if (out/'sample_weight_registry.json').exists():env['LINGBOT_SAMPLE_WEIGHT_CONFIG']=str(out/'sample_weight_registry.json')
        if (out/'trainer_config.json').exists():env['QWEN3VL_PATH']=json.loads((out/'trainer_config.json').read_text())['model']['tokenizer_path']
        for k in ['LINGBOT_PRECISION_AUDIT','LINGBOTVLA_TRAINING_CONFIG','AL_VLM_ATTENTION']:env.pop(k,None)
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            print('CHILD',child.pid,str(log),flush=True);code=child.wait()
        if code:raise RuntimeError((str(log),code))
    try:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'GPU busy; refusing concurrent run'
        state('prepare');prepare(c)
        logs=Path('/root/tf-logs')
        if logs.is_dir():
            name=out.name+'_'+out.parent.name if 'remediation_round_' in out.name else out.name
            link=logs/name
            assert not link.exists() and not link.is_symlink(),'TensorBoard link already exists'
            link.symlink_to(out/'runs',target_is_directory=True)
        here=str(Path(__file__).resolve())
        state('cpu_preflight');run([sys.executable,here,'--stage','cpu','--out',str(out)],out/'cpu.log')
        state('gpu_probe');run([sys.executable,here,'--stage','smoke','--out',str(out)],out/'probe.log')
        probe=json.loads((out/'probe/summary.json').read_text())
        assert probe['state'] in ['three_step_probe_complete','probe_complete'] and probe.get('optimizer_steps',3)==probe_steps(c)
        state('training',steps=c['train']['steps']);run(['bash',str(out/'train.sh')],out/'train.log')
        from experiment.official_control.baseline_pipeline import verify_checkpoint
        report=json.loads((out/'plan.json').read_text())
        verified=[verify_checkpoint(out/f'checkpoints/global_step_{step}/hf_ckpt') for step in report['hf_steps']]
        (out/'verified_checkpoints.json').write_text(json.dumps(verified,indent=2))
        from experiment.data_selection.inheritance import freeze_registry
        for step in report['hf_steps']:
            freeze_registry(out,out/f'checkpoints/global_step_{step}/hf_ckpt')
        results=[]
        for step in report['evaluation_steps']:
            if not c['evaluation']['tasks']:continue
            state('evaluation',step=step,completed=results)
            directory=out/f'evaluation_{step}'
            if c['evaluation'].get('reuse_training_model'):
                from experiment.task_recipes.resident_evaluation import read_completed
                results.extend(read_completed(directory,c,step))
                continue
            directory.mkdir(exist_ok=False)
            cfg=json.loads((out/'trainer_config.json').read_text())
            from experiment.task_recipes.resident_evaluation import jobs_for
            from experiment.official_control.episode_parallel import workers_for
            jobs=jobs_for(c,out,step,cfg)
            workers=workers_for(c['evaluation'])
            if c['evaluation']['shared_model_session']:
                if workers>1 or any('episode_index' in job for job in jobs):
                    from experiment.official_control.parallel_evaluation import run_jobs
                    run_jobs(jobs,directory,workers,scheduling=c['evaluation'].get('scheduling','batches'))
                else:
                    from experiment.official_control.shared_evaluation import run_jobs
                    run_jobs(jobs,directory)
                results.extend(json.loads((directory/'summary.json').read_text()))
            else:
                for job in jobs:
                    dest=Path(job['destination']);dest.mkdir(exist_ok=False)
                    path=dest/'job.json';path.write_text(json.dumps(job,indent=2))
                    run([sys.executable,str(ROOT/'experiment/official_control/evaluate.py'),'--job',str(path)],dest/'pipeline.log')
                    results.append(json.loads((dest/'summary.json').read_text()))
        from experiment.task_recipes.remediation import run_rounds
        def record_round(number,lessons,child):
            state('remediation',round=number,curriculum=lessons,output_dir=child['output_dir'])
            (out/f'remediation_round_{number}_plan.json').write_text(json.dumps(dict(curriculum=lessons,recipe=child),indent=2))
        def execute_round(child):
            child_launch=Path(child['output_dir']+'_pipeline');child_launch.mkdir(exist_ok=False)
            (child_launch/'recipe.json').write_text(json.dumps(child,indent=2))
            if worker(child,child_launch)!=0:raise RuntimeError('remediation round failed; see '+str(child_launch))
            return json.loads((child_launch/'status.json').read_text())['evaluation']
        remediation=run_rounds(c,results,execute_round,record_round)
        (out/'remediation_summary.json').write_text(json.dumps(remediation,indent=2))
        state('complete',checkpoints=verified,evaluation=results,remediation=remediation,
            progress=confirmed_progress(p,c['train']['steps']),shutdown=False)
        return 0
    except BaseException as error:
        state('failed',error=repr(error));traceback.print_exc();return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--stage',choices=['prepare','cpu','smoke','train'])
    a=parser.parse_args()
    if a.worker:return worker(json.loads(a.worker.read_text()),a.worker.parent)
    if a.stage in ['cpu','smoke','train']:
        if a.out is None:parser.error('--out required')
        if a.stage=='cpu':
            from experiment.task_recipes.cpu import check
            return check(a.out)
        c,p,_=load_run(a.out);install(a.out)
        cfg=json.loads((a.out/'trainer_config.json').read_text())
        os.environ['QWEN3VL_PATH']=cfg['model']['tokenizer_path']
        for k in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE']:os.environ[k]='1'
        if a.stage=='smoke':
            from experiment.official_control.bf16_trial import smoke
            return smoke(a.out,steps=probe_steps(c))
        import runpy
        assert json.loads((a.out/'cpu_report.json').read_text())['state']=='cpu_pass'
        os.chdir(ROOT);sys.argv=['tasks/vla/train_lingbotvla.py',str(a.out/'trainer_config.json')]
        initial={}
        if c['evaluation'].get('reuse_training_model'):
            from experiment.task_recipes.resident_evaluation import evaluate_resident
            initial['_POST_TRAINING_CALLBACK']=lambda model,step:evaluate_resident(model,step,a.out)
        runpy.run_path(sys.argv[0],run_name='__main__',init_globals=initial);return
    if a.recipe is None:parser.error('--recipe required')
    raw=json.loads(a.recipe.read_text())
    if a.out:raw['output_dir']=str(a.out)
    raw['output_dir']=dated_run_path(raw['output_dir'])
    c,p,groups,frames=preview(raw)
    if a.stage=='prepare':prepare(c);return
    from experiment.task_recipes.schema import milestones
    print(json.dumps(dict(source=c['source'],output_dir=c['output_dir'],tasks={k:len(v) for k,v in groups.items()},frames=frames,
        optimizer_steps=c['train']['steps'],global_batch_size=p['global_batch_size'],hf_steps=milestones(c),
        projected_progress=confirmed_progress(p,c['train']['steps']),evaluation=c['evaluation'],remediation=c['remediation'],training_started=False),indent=2))
    if a.execute:
        out=Path(c['output_dir']);assert not out.exists(),'refuse to overwrite run'
        launch=out.with_name(out.name+'_pipeline');launch.mkdir(parents=True,exist_ok=False)
        path=launch/'recipe.json';path.write_text(json.dumps(c,indent=2))
        with (launch/'pipeline.log').open('x') as log:
            child=subprocess.Popen(['setsid','nohup',sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(path)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        print('RECIPE_STARTED',child.pid,str(launch))

if __name__=='__main__':raise SystemExit(main())

"""Finite recovery巡检: compare with historical best; train declines and test fresh tasks."""
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def historical_best(snapshot):
    result={}
    for task in snapshot['tasks']:
        rows=task['observations']
        if not rows:raise ValueError('missing observations')
        for row in rows:
            s,n=row['successes'],row['episodes']
            if type(s)is not int or type(n)is not int or n<1 or not 0<=s<=n:raise ValueError('invalid historical counts')
        best=max(enumerate(rows),key=lambda pair:(Fraction(pair[1]['successes'],pair[1]['episodes']),pair[0]))[1]
        if best['successes']>0:result[task['task']]=dict(best)
    return result


def declines(results,peaks,steps=80):
    seen=set();lessons=[]
    for row in results:
        task=row['task'];s=row.get('successes')
        if task in seen or task not in peaks or row.get('state')!='complete' or row.get('episodes')!=3 or type(s)is not int or not 0<=s<=3:
            raise ValueError('invalid evaluation, never interpret errors as failures')
        seen.add(task);best=peaks[task]
        if Fraction(s,3)<Fraction(best['successes'],best['episodes']):
            lessons.append(dict(task=task,steps=steps,current_successes=s,current_episodes=3,historical_best=best))
    return lessons


def choose_tasks(peaks,treated,seen,seed,round_index,count=5):
    candidates=sorted(set(peaks)-set(treated)-set(seen))
    if len(candidates)<count:raise ValueError('not enough unseen untreated tasks')
    return random.Random(f'{seed}:{round_index}').sample(candidates,count)


def make_recipe(template,source_out,source_step,lessons,tests,out,review_tasks=None,review_fraction=0,review_switch=5):
    c=deepcopy(template)
    c['source']=dict(trainer_config=str(Path(source_out)/'trainer_config.json'),checkpoint=str(Path(source_out)/f'checkpoints/global_step_{source_step}/hf_ckpt'))
    c['output_dir']=str(out);t=c['train']
    for key in ['tasks_per_group','steps_per_group']:t.pop(key,None)
    t.update(mode='sequential',steps=sum(x['steps']for x in lessons),main_tasks={x['task']:dict(steps=x['steps'],weight=1)for x in lessons},task_order=[x['task']for x in lessons],replay=dict(fraction=0,tasks={}))
    if review_fraction:
        t['replay']=dict(mode='round_robin',fraction=review_fraction,tasks_per_step=1,switch_every_steps=review_switch,
            tasks={task:dict(weight=1)for task in review_tasks})
    c['checkpoint'].update(mode='steps',every_steps=500,save_final=True,hf_dtype='bf16');c['checkpoint'].pop('every_tasks',None)
    c['evaluation'].update(tasks=tests,checkpoints=['final'],episodes_per_task=3,shared_model_session=True,max_parallel_tasks=5,video_episodes=0,reuse_training_model=True)
    c.setdefault('remediation',{})['enabled']=False
    c.setdefault('runtime',{})['shutdown_after_completion']=False
    return c


def run(config):
    from experiment.task_recipes.run import worker
    from experiment.task_recipes.naming import dated_run_path
    from experiment.official_control.parallel_evaluation import run_jobs
    from experiment.official_control.baseline_pipeline import verify_checkpoint
    rounds=config['max_rounds'];budget=config['steps_per_task']
    if type(rounds)is not int or not 1<=rounds<=3 or type(budget)is not int or budget<1:raise ValueError('invalid round budget')
    root=Path(config['output_dir']);root.mkdir(parents=True,exist_ok=False)
    (root/'job.json').write_text(json.dumps(config,indent=2));history=[]
    def state(value,**kw):
        (root/'status.json').write_text(json.dumps(dict(state=value,rounds=history,**kw),indent=2));print(value,kw,flush=True)
    try:
        peaks=historical_best(json.loads((ROOT/config['history_file']).read_text()))
        initial=json.loads((ROOT/config['initial_report']).read_text())['first5_results']
        lessons=declines(initial,peaks,budget)
        if [x['task']for x in lessons]!=config['initial_lessons']:raise ValueError('initial lesson evidence differs')
        treated=set();seen={r['task']for r in initial}
        source=Path(config['source_run']);source_step=config['source_step']
        template=json.loads((ROOT/config['template']).read_text());verify_checkpoint(source/f'checkpoints/global_step_{source_step}/hf_ckpt')
        for number in range(1,rounds+1):
            pending_treated=treated|{l['task']for l in lessons}
            tests=choose_tasks(peaks,pending_treated,seen,config['seed'],number)
            child=Path(dated_run_path(root/f'round_{number}_recovery'))
            entry=dict(round=number,source_model=str(source/f'checkpoints/global_step_{source_step}/hf_ckpt'),lessons=lessons,evaluation_tasks=tests,output_dir=str(child))
            (root/f'round_{number}_plan.json').write_text(json.dumps(entry,indent=2))
            if lessons:
                recipe=make_recipe(template,source,source_step,lessons,tests,child,sorted(peaks),
                    config.get('review_fraction',0),config.get('review_switch_every_steps',5))
                files=list((source/f'checkpoints/global_step_{source_step}/hf_ckpt').glob('*.safetensors'))
                required=int(sum(p.stat().st_size for p in files)*1.1)+3*2**30
                deadline=time.monotonic()+config.get('storage_wait_seconds',86400)
                while shutil.disk_usage(root).free<required:
                    if time.monotonic()>deadline:raise TimeoutError('storage remains insufficient; no checkpoint policy changed')
                    state('waiting_for_storage',round=number,required_gib=required/2**30,free_gib=shutil.disk_usage(root).free/2**30,planned=entry)
                    time.sleep(10)
                launch=child.with_name(child.name+'_pipeline');launch.mkdir(exist_ok=False)
                (launch/'recipe.json').write_text(json.dumps(recipe,indent=2))
                state('training_and_evaluation',round=number,planned=entry)
                if worker(recipe,launch)!=0:raise RuntimeError('round failed: '+str(launch))
                done=json.loads((launch/'status.json').read_text());results=done['evaluation']
                source=child;source_step=recipe['train']['steps'];treated=pending_treated
                entry['checkpoints']=done['checkpoints'];entry['training_steps']=source_step
            else:
                child.mkdir(exist_ok=False);cfg=json.loads((source/'trainer_config.json').read_text());jobs=[]
                for task in tests:
                    jobs.append(dict(task=task,label=f'recovery_scout_{number}',model=str(source/f'checkpoints/global_step_{source_step}/hf_ckpt'),destination=str(child/task),episodes=3,video_episodes=0,require_baseline=False,robotwin=str(ROOT.parent/'RoboTwin-lingbot'),qwen=cfg['model']['tokenizer_path'],sim_python='/data/miniconda3/envs/RoboTwin/bin/python'))
                state('evaluation_only',round=number,planned=entry)
                results=run_jobs(jobs,child,5);entry['training_steps']=0
            next_lessons=declines(results,peaks,budget)
            seen.update(tests);entry.update(state='complete',evaluation=results,next_lessons=next_lessons)
            history.append(entry);(root/f'round_{number}_completed.json').write_text(json.dumps(entry,indent=2))
            # Only update reference best upwards; do not lower the historical target.
            for r in results:
                if Fraction(r['successes'],3)>Fraction(peaks[r['task']]['successes'],peaks[r['task']]['episodes']):
                    peaks[r['task']]=dict(successes=r['successes'],episodes=3,source=str(child))
            lessons=next_lessons
        state('complete',latest_model=str(source/f'checkpoints/global_step_{source_step}/hf_ckpt'),treated_tasks=sorted(treated),pending_after_final=lessons,
            note='trained tasks are excluded from re-evaluation; recovery effectiveness unverified')
    except BaseException as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args();run(json.loads(a.job.read_text()))

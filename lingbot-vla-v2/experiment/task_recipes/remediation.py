"""闭环结果驱动的有限补课；异常结果不转换为失败样本。仅依赖标准库。"""
from copy import deepcopy
from pathlib import Path
from experiment.task_recipes.naming import dated_run_path


def curriculum(tasks, results, model):
    selected=[r for r in results if r.get('model')==str(model)]
    if len(selected)!=len(tasks) or {r.get('task') for r in selected}!=set(tasks):
        raise ValueError('missing or duplicate final evaluation results')
    by_task={r['task']:r for r in selected}
    lessons=[]
    for task in tasks:
        r=by_task[task];successes=r.get('successes')
        if r.get('state')!='complete' or r.get('episodes')!=3 or type(successes) is not int or not 0<=successes<=3:
            raise ValueError('invalid closed-loop evaluation: '+task)
        if successes in (1,2):lessons.append(dict(task=task,successes=successes,steps=100))
    # Prefer 2/3, then 1/3; equal scores preserve evaluation task order.
    return sorted(lessons,key=lambda x:-x['successes'])


def next_recipe(parent, previous_out, lessons, round_index):
    c=deepcopy(parent);previous_out=Path(previous_out)
    c['source']=dict(trainer_config=str(previous_out/'trainer_config.json'),
        checkpoint=str(previous_out/f"checkpoints/global_step_{parent['train']['steps']}/hf_ckpt"))
    c['output_dir']=dated_run_path(Path(c['output_dir'])/f'remediation_round_{round_index}')
    c['train'].update(mode='sequential',main_tasks={x['task']:dict(steps=x['steps'],weight=1) for x in lessons},
        task_order=[x['task'] for x in lessons],steps=sum(x['steps'] for x in lessons),replay=dict(fraction=0,tasks={}))
    c['evaluation']['checkpoints']=['final']
    c['evaluation']['tasks']=[x['task'] for x in lessons]
    # User-approved 300-step course exception: avoid a redundant save only 50 steps before final.
    if c['train']['steps']==300 and c['checkpoint'].get('mode','steps')=='steps' and c['checkpoint']['every_steps']==250:
        c['checkpoint']['every_steps']=300
    c['remediation']['enabled']=False  # The outer controller owns the round budget.
    return c


def run_rounds(c, results, execute, record):
    if not c['remediation']['enabled']:return []
    current=c;history=[]
    for number in range(1,c['remediation']['max_rounds']+1):
        out=Path(current['output_dir'])
        model=out/f"checkpoints/global_step_{current['train']['steps']}/hf_ckpt"
        lessons=curriculum(current['evaluation']['tasks'],results,model)
        if not lessons:break
        child=next_recipe(current,out,lessons,number)
        # Keep round directories beside one another, never nested inside the last round.
        child['output_dir']=dated_run_path(Path(c['output_dir'])/f'remediation_round_{number}')
        record(number,lessons,child)
        results=execute(child)
        child_model=Path(child['output_dir'])/f"checkpoints/global_step_{child['train']['steps']}/hf_ckpt"
        curriculum(child['evaluation']['tasks'],results,child_model)  # Validate even the last round's results.
        history.append(dict(round=number,curriculum=lessons,output_dir=child['output_dir'],evaluation=results))
        current=child
    return history

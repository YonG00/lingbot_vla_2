"""严格配方契约：仅标准库；未识别字段直接报错。"""
from copy import deepcopy
import math


def keys(value, allowed, label):
    if not isinstance(value,dict) or set(value)-set(allowed):
        raise ValueError(f'{label}: unknown fields or invalid object')


def integer(value,label):
    if type(value) is not int or value<=0:raise ValueError(f'{label}: positive integer required')


def weight(value,label):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError(f'{label}: finite positive weight required')


def validate(raw):
    try:return _validate(raw)
    except (KeyError,TypeError) as error:raise ValueError('missing or malformed recipe field: '+str(error)) from error


def _validate(raw):
    c=deepcopy(raw)
    keys(c,['version','source','data','output_dir','train','checkpoint','evaluation','runtime','remediation'],'recipe')
    if type(c.get('version')) is not int or c['version']!=1:raise ValueError('recipe version must be 1')
    keys(c['source'],['trainer_config','checkpoint'],'source')
    if not all(isinstance(c['source'].get(k),str) and c['source'][k] for k in ['trainer_config','checkpoint']):raise ValueError('explicit source paths required')
    if not isinstance(c.get('output_dir'),str) or not c['output_dir']:raise ValueError('output_dir required')
    keys(c['data'],['dataset_root'],'data')
    if not isinstance(c['data'].get('dataset_root'),str) or not c['data']['dataset_root']:raise ValueError('dataset_root required')
    t=c['train'];keys(t,['mode','steps','micro_batch_size','gradient_accumulation_steps','seed','main_tasks','replay','episode_split','task_order'],'train')
    if t['mode'] not in ['mixed','task_units','sequential']:raise ValueError('mode must be mixed, task_units or sequential')
    for name in ['micro_batch_size','gradient_accumulation_steps']:integer(t[name],name)
    if type(t['seed']) is not int or t['seed']<0:raise ValueError('seed must be nonnegative integer')
    if t.get('episode_split','all') not in ['all','train']:raise ValueError('episode_split must be all or train')
    t.setdefault('episode_split','all')
    main=t['main_tasks']
    if not isinstance(main,dict) or not main:raise ValueError('main_tasks cannot be empty')
    for name,entry in main.items():
        keys(entry,['weight','steps'],'main task')
        weight(entry.get('weight',1),name);entry.setdefault('weight',1)
        if t['mode'] in ['task_units','sequential']:integer(entry['steps'],name+'.steps')
        elif 'steps' in entry:raise ValueError('mixed mode uses sample ratios; per-task optimizer steps require task_units')
    if t['mode'] in ['task_units','sequential']:
        total=sum(x['steps'] for x in main.values())
        if 'steps' in t and t['steps']!=total:raise ValueError('global steps differ from sum of main budgets')
        t['steps']=total
    if t['mode']=='sequential':
        order=t.get('task_order')
        if not isinstance(order,list) or len(order)!=len(set(order)) or set(order)!=set(main):raise ValueError('sequential task_order must cover every main task exactly once')
        if any(v['weight']!=1 for v in main.values()):raise ValueError('sequential mode uses task_order and steps, not task weights')
    elif 'task_order' in t:raise ValueError('task_order only applies to sequential mode')
    integer(t['steps'],'steps')
    r=t.setdefault('replay',dict(fraction=0,tasks={}))
    keys(r,['fraction','tasks','mode','switch_every_steps','pool_file','pool_snapshot'],'replay')
    r.setdefault('mode','sample')
    if r['mode'] not in ['sample','round_robin']:raise ValueError('invalid replay mode')
    if r['mode']=='round_robin':
        r.setdefault('switch_every_steps',10);integer(r['switch_every_steps'],'switch_every_steps')
        if t['mode']=='mixed':raise ValueError('round_robin replay requires one main task per update')
    elif 'switch_every_steps' in r:raise ValueError('switch_every_steps requires round_robin replay')
    f=r['fraction']
    if type(f) not in (int,float) or not math.isfinite(f) or not 0<=f<1:raise ValueError('replay fraction must be in [0,1)')
    if t['micro_batch_size']*t['gradient_accumulation_steps']*(1-f)<1:raise ValueError('every update must contain at least one main sample')
    if not isinstance(r['tasks'],dict) or bool(r['tasks'])!=(f>0):raise ValueError('replay tasks and fraction must agree')
    if r['mode']=='sample' and set(main)&set(r['tasks']):raise ValueError('main and replay tasks must not overlap')
    for name,entry in r['tasks'].items():keys(entry,['weight'],'replay task');weight(entry['weight'],name)
    if r['mode']=='round_robin' and f>0:
        if any(x['weight']!=1 for x in r['tasks'].values()):raise ValueError('round_robin replay uses equal task weights')
        if any(not(set(r['tasks'])-{task}) for task in main):raise ValueError('no replay task remains after excluding current main task')
    import re
    for name in list(main)+list(r['tasks']):
        if not re.fullmatch('[a-z][a-z0-9_]*',name):raise ValueError('invalid task name')
    cp=c['checkpoint'];keys(cp,['mode','every_steps','every_tasks','save_final','hf_dtype'],'checkpoint')
    cp.setdefault('mode','steps')
    if cp['mode']=='steps':
        integer(cp['every_steps'],'every_steps')
        if 'every_tasks' in cp:raise ValueError('every_tasks requires task checkpoint mode')
    elif cp['mode']=='task':
        if t['mode']!='sequential':raise ValueError('task checkpoint mode requires sequential training')
        if 'every_steps' in cp:raise ValueError('task checkpoint mode uses boundaries, not every_steps')
        cp.setdefault('every_tasks',1);integer(cp['every_tasks'],'every_tasks')
    else:raise ValueError('checkpoint mode must be steps or task')
    if cp['save_final'] is not True or cp['hf_dtype']!='bf16':raise ValueError('v1 requires final BF16 HF; no silent precision/save change')
    e=c['evaluation'];keys(e,['tasks','episodes_per_task','checkpoints','shared_model_session'],'evaluation')
    e.setdefault('tasks',list(t.get('task_order',main)))
    if not isinstance(e['tasks'],list) or len(e['tasks'])!=len(set(e['tasks'])):raise ValueError('evaluation tasks must be unique list')
    for name in e['tasks']:
        if not isinstance(name,str) or not re.fullmatch('[a-z][a-z0-9_]*',name):raise ValueError('invalid evaluation task')
    integer(e['episodes_per_task'],'episodes_per_task')
    if type(e['shared_model_session']) is not bool:raise ValueError('shared_model_session must be boolean')
    saves=milestones(c)
    if not isinstance(e['checkpoints'],list) or not e['checkpoints']:raise ValueError('evaluation checkpoint list required')
    selected=[]
    for step in e['checkpoints']:
        n=t['steps'] if step=='final' else step
        if type(n) is not int or n not in saves:raise ValueError('evaluation checkpoint is not saved')
        selected.append(n)
    if len(selected)!=len(set(selected)):raise ValueError('duplicate evaluation checkpoints')
    remedial=c.setdefault('remediation',{})
    keys(remedial,['enabled','max_rounds'],'remediation')
    remedial.setdefault('enabled',False);remedial.setdefault('max_rounds',1)
    if type(remedial['enabled']) is not bool:raise ValueError('remediation enabled must be boolean')
    integer(remedial['max_rounds'],'remediation.max_rounds')
    if remedial['enabled']:
        if not e['tasks'] or e['episodes_per_task']!=3 or t['steps'] not in selected:
            raise ValueError('remediation requires final checkpoint evaluation with three episodes per task')
        if t['replay']['fraction']!=0:raise ValueError('remediation currently requires replay disabled')
    runtime=c.setdefault('runtime',{})
    keys(runtime,['shutdown_after_completion'],'runtime')
    if runtime.get('shutdown_after_completion',False) is not False:raise ValueError('v1 does not perform shutdown; request it separately')
    runtime.setdefault('shutdown_after_completion',False)
    return c


def milestones(c):
    if c['checkpoint'].get('mode','steps')=='task':
        total=0;result=[]
        order=c['train']['task_order'];interval=c['checkpoint'].get('every_tasks',1)
        for index,task in enumerate(order,1):
            total+=c['train']['main_tasks'][task]['steps']
            if index%interval==0 or index==len(order):result.append(total)
        return result
    n=c['train']['steps'];interval=c['checkpoint']['every_steps']
    return sorted(set(list(range(interval,n+1,interval))+[n]))


def save_interval(c):
    return 0 if c['checkpoint'].get('mode','steps')=='task' else c['checkpoint']['every_steps']

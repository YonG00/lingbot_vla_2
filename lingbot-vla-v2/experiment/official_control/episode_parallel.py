"""Single-task episode shards, explicit seeds and canonical task aggregation."""
import ast
import json
from pathlib import Path
import shutil
from .recording import video_count

def workers_for(e):
    if len(e['tasks'])==1 and e.get('single_task_episode_workers',1)>1:
        return min(e['single_task_episode_workers'],e['episodes_per_task'])
    return e['max_parallel_tasks']

def split_jobs(jobs,e):
    if len(jobs)!=1 or e.get('single_task_episode_workers',1)==1:return jobs
    job=jobs[0];n=job['episodes']
    if n==1 and 'single_task_episode_seeds' not in e:return jobs
    seeds=e.get('single_task_episode_seeds',list(range(100000,100000+n)))
    if len(seeds)!=n or len(set(seeds))!=n or any(type(s)is not int or s<0 for s in seeds):raise ValueError('unique explicit scene seeds required')
    videos=video_count(n,job.get('video_episodes'));parent=Path(job['destination'])
    return [dict(job,job_id=f"{job['task']}:episode:{i}",episode_index=i,episode_total=n,
        scene_seed=seed,expected_seeds=[seed],description_episode_count=n,episodes=1,
        video_episodes=int(i<videos),destination=str(parent/'episodes'/f'episode_{i:03d}'),
        aggregate_destination=str(parent)) for i,seed in enumerate(seeds)]

def validate_jobs(jobs):
    if not jobs or len({j.get('job_id',j['task']) for j in jobs})!=len(jobs):raise ValueError('unique nonempty evaluation job identities required')
    if len({str(Path(j['destination']).resolve()) for j in jobs})!=len(jobs):raise ValueError('duplicate evaluation destinations')
    if len({j['task'] for j in jobs})!=len(jobs):
        if len({j['task'] for j in jobs})!=1 or any('episode_index'not in j for j in jobs):raise ValueError('duplicate tasks require single-task episode shards')
        if sorted(j['episode_index'] for j in jobs)!=list(range(len(jobs))):raise ValueError('episode shard indices must be complete')
        if len({j['scene_seed'] for j in jobs})!=len(jobs):raise ValueError('duplicate episode seeds')

def aggregate(results,jobs):
    if not any('episode_index' in j for j in jobs):return results
    ordered=sorted(jobs,key=lambda j:j['episode_index'])
    if len(results)!=len(ordered):raise ValueError('incomplete episode results')
    by_id={result['job_id']:result for result in results}
    if len(by_id)!=len(ordered):raise ValueError('duplicate episode result identity')
    rows=[];video_paths=[];root=Path(ordered[0]['aggregate_destination'])
    for job in ordered:
        result=by_id[job['job_id']]
        if (result['state']!='complete' or result['task']!=job['task'] or result['seeds']!=[job['scene_seed']]
            or result['episodes']!=1 or result['successes']not in (0,1) or result.get('oidn_errors')!=0
            or result['recorded_episodes']!=job['video_episodes']):raise ValueError('episode result mismatch')
        if result['model']!=job['model']:raise ValueError('episode model mismatch')
        rows.append(result)
        if job['video_episodes']:
            source=Path(job['destination'])/'results'/job['task']/'episode0_global.mp4'
            if not source.is_file() or not source.stat().st_size:raise ValueError('missing episode video')
            video_paths.append((source,root/'results'/job['task']/f"episode{job['episode_index']}_global.mp4"))
    for key in ['server_pid','shared_server','dtype','action_chunk','client_sha256','model']:
        if len({row[key] for row in rows})!=1:raise ValueError('episode session/precision mismatch: '+key)
    if not rows[0]['shared_server']:raise ValueError('episode shards must share inference')
    summary={k:v for k,v in rows[0].items() if k not in ['job_id','episode_index','scene_seed']}
    summary.update(episodes=len(rows),successes=sum(row['successes'] for row in rows),
        success_rate=sum(row['successes'] for row in rows)/len(rows),seeds=[job['scene_seed'] for job in ordered],
        recorded_episodes=len(video_paths),episode_results=rows,episode_parallel=True)
    for source,target in video_paths:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('refusing video overwrite')
        shutil.copyfile(source,target)
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    return [summary]

def retryable_expert_error(error):
    """Only known scene-dependent expert planning failures; never policy errors."""
    import traceback
    frames=traceback.extract_tb(error.__traceback__)
    if isinstance(error,AssertionError) and str(error)=='target_pose cannot be None for move action.':
        return any(Path(f.filename).name=='_base_task.py' and f.name=='grasp_actor' for f in frames)
    if isinstance(error,IndexError) and str(error)=='list index out of range':
        return any(Path(f.filename).name=='put_bottles_dustbin.py' and f.name=='play_once' for f in frames)
    return False


def compile_episode_client(source,filename,seed,description_count,episode_index=0, seed_search=None):
    """Patch only runtime AST; retain checked simulator source files unchanged."""
    tree=ast.parse(source,filename);seed_nodes=[];description_nodes=[];index_nodes=[];loops=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='st_seed' for t in node.targets):
            if ast.dump(node.value)!=ast.dump(ast.parse('100000 * (1 + seed)',mode='eval').body):raise ValueError('unsupported client seed expression')
            seed_nodes.append(node)
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='now_id' for t in node.targets):
            if not isinstance(node.value,ast.Constant) or node.value.value!=0:raise ValueError('unsupported episode index contract')
            index_nodes.append(node)
        if isinstance(node,ast.While) and ast.dump(node.test)==ast.dump(ast.parse('succ_seed < test_num',mode='eval').body):loops.append(node)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='generate_episode_descriptions':
            if len(node.args)!=3 or not isinstance(node.args[2],ast.Name) or node.args[2].id!='test_num':raise ValueError('unsupported description contract')
            description_nodes.append(node)
    if len(seed_nodes)!=1 or len(description_nodes)!=1 or len(index_nodes)!=1 or len(loops)!=1:raise ValueError('client seed/description adapter contract changed')
    index_nodes[0].value=ast.Constant(episode_index)
    if seed_search is None:
        guard=ast.parse(f'if now_seed != {seed}: raise RuntimeError("fixed episode seed rejected by expert validation; provide validated seeds")').body[0]
        loops[0].body.insert(0,guard)
    else:
        stride=seed_search['stride'];limit=seed_search['max_candidates']
        if type(stride) is not int or stride<1 or type(limit) is not int or limit<1:
            raise ValueError('invalid inline seed search')
        increments=[n for n in ast.walk(loops[0]) if isinstance(n,ast.AugAssign)
            and isinstance(n.target,ast.Name) and n.target.id=='now_seed']
        if not increments or any(not isinstance(n.op,ast.Add) or not isinstance(n.value,ast.Constant) or n.value.value!=1 for n in increments):
            raise ValueError('unsupported seed increment contract')
        for node in increments:node.value=ast.Constant(stride)
        # Only expected unstable scenes and expert-unsuccessful cases may retry.
        for node in ast.walk(loops[0]):
            if isinstance(node,ast.ExceptHandler) and isinstance(node.type,ast.Name) and node.type.id=='Exception':
                error_name=node.name or '_expert_error'
                node.name=error_name
                node.body=ast.parse(f"""
from experiment.official_control.episode_parallel import retryable_expert_error
if not retryable_expert_error({error_name}):
    raise
TASK_ENV.close_env()
print('INLINE_EXPERT_REJECTED seed=' + str(now_seed) + ' reason=' + str({error_name}), flush=True)
now_seed += {stride}
args['render_freq'] = render_freq
continue
""").body
        # The original client reconstructs the chosen scene before policy inference.
        # An unstable reconstruction is not a policy failure: no action was requested.
        setups=[n for n in loops[0].body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
            and n.value.func.value.id=='TASK_ENV' and n.value.func.attr=='setup_demo']
        if len(setups)>1:raise ValueError('ambiguous policy scene setup')
        if setups:
            setup=setups[0];index=loops[0].body.index(setup)
            handler=ast.parse(f"""
try:
    pass
except UnStableError:
    TASK_ENV.close_env()
    succ_seed -= 1
    suc_test_seed_list.pop()
    print('INLINE_SCENE_REBUILD_REJECTED seed=' + str(now_seed), flush=True)
    now_seed += {stride}
    continue
""").body[0]
            handler.body=[setup]
            loops[0].body[index]=handler
        guards=ast.parse(f"""
if now_seed < {seed} or (now_seed-{seed}) % {stride} != 0 or (now_seed-{seed}) // {stride} >= {limit}:
    raise RuntimeError('inline expert seed candidate limit exhausted')
print('INLINE_EXPERT_CANDIDATE seed=' + str(now_seed), flush=True)
""").body
        loops[0].body[:0]=guards
    seed_nodes[0].value=ast.Constant(seed);description_nodes[0].args[2]=ast.Constant(description_count)
    return compile(ast.fix_missing_locations(tree),filename,'exec')

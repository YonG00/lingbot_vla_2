"""Select expert-valid scenes before launching parallel policy episodes."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def compile_expert_client(source, filename):
    """Reuse actual client setup; replace only policy execution with expert selection."""
    tree = ast.parse(source, filename)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_policy']
    constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id == 'WebsocketClientPolicy']
    if len(functions) != 1 or len(constructors) != 1:
        raise ValueError('expert client setup contract changed')
    function = functions[0]
    # Refuse to bypass an unfamiliar implementation's success criterion.
    required = ['TASK_ENV.play_once()', 'TASK_ENV.plan_success and TASK_ENV.check_success()']
    normalized = ast.unparse(function)
    if any(s not in normalized for s in required):
        raise ValueError('expert success predicate changed')
    constructors[0].value = ast.Constant(None)
    function.body = ast.parse('''
from experiment.official_control.expert_seed_selection import scan_expert
return scan_expert(TASK_ENV, args, st_seed, test_num)
''').body
    return compile(ast.fix_missing_locations(tree), filename, 'exec')


def scan_expert(task, args, start, count):
    from envs.utils.create_actor import UnStableError
    from experiment.official_control.model_benchmark import atomic_json
    import numpy as np
    destination = Path(os.environ['LINGBOT_SEED_SCAN_OUTPUT'])
    audit = json.loads(os.environ['LINGBOT_SEED_SCAN_IDENTITY'])
    limit = int(os.environ['LINGBOT_SEED_SCAN_MAX_CANDIDATES'])
    accepted = []
    candidates = []
    args['eval_mode'] = True
    args['render_freq'] = 0
    args['eval_video_log'] = False
    args['eval_global_view_log'] = False
    task.suc = 0
    task.test_num = 0
    # Each future policy episode starts from an independent client RNG state.
    py_state, np_state = random.getstate(), np.random.get_state()
    for seed in range(start, start + limit):
        random.setstate(py_state); np.random.set_state(np_state)
        try:
            task.setup_demo(now_ep_num=len(accepted), seed=seed, is_test=True, **args)
            task.play_once()
            task.close_env()
            valid = bool(task.plan_success and task.check_success())
            reason = 'expert_success' if valid else 'expert_unsuccessful'
        except UnStableError:
            task.close_env()
            valid, reason = False, 'unstable_scene'
        except BaseException:
            # Renderer/asset/program errors must not be hidden as rejected scenes.
            task.close_env()
            raise
        candidates.append(dict(seed=seed, valid=valid, reason=reason))
        if valid:
            accepted.append(seed)
        audit.update(state='complete' if len(accepted)==count else 'selecting',
                     seeds=accepted, candidates=candidates)
        atomic_json(destination, audit)
        print('EXPERT_SEED_CANDIDATE '+json.dumps(candidates[-1]), flush=True)
        if len(accepted) == count:
            return seed + 1, 0
    raise RuntimeError('expert seed candidate limit exhausted; accepted '+str(len(accepted))+'/'+str(count))


def identity_for(config, item):
    root = Path(config['robotwin'])
    paths = [root/'script/eval_policy_client_lingbotvla.py',
             root/'task_config'/ (item['task_config']+'.yml'),
             root/'envs'/(item['task']+'.py'), Path(__file__)]
    return dict(task=item['task'], task_config=item['task_config'], episodes=item['episodes'],
        start_seed=config.get('seed_search_start',100000),
        max_candidates=config.get('seed_search_max_candidates',200),
        file_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def select_seeds(config, item, directory):
    from experiment.official_control.model_benchmark import atomic_json
    if config.get('seed_selection','inline') in ('inline','fixed') or item['key'] in config.get('scene_seeds',{}):
        return list(item['seeds'])
    identity = identity_for(config,item)
    output = Path(directory)/'valid_seeds'/item['task_config']/(item['task']+'.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous['identity'] != identity:
            raise ValueError('expert seed cache identity changed: '+item['key'])
        if previous['state']=='complete':
            seeds=previous['seeds']
            if len(seeds)!=item['episodes'] or len(set(seeds))!=len(seeds):
                raise ValueError('invalid expert seed cache')
            return seeds
    root = Path(config['robotwin'])
    log = output.with_suffix('.log')
    result_dir=output.parent/(item['task']+'_expert_only')
    env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT),str(root),str(root/'script')]),
        PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='0', LINGBOT_SEED_SCAN_OUTPUT=str(output),
        LINGBOT_SEED_SCAN_IDENTITY=json.dumps(dict(identity=identity)),
        LINGBOT_SEED_SCAN_MAX_CANDIDATES=str(identity['max_candidates']))
    command=[config['sim_python'],'-u',str(Path(__file__).resolve()),'--client',str(root),
        '--config','policy/ACT/deploy_policy.yml','--overrides','--task_name',item['task'],
        '--task_config',item['task_config'],'--train_config_name','expert_seed_selection',
        '--seed','0','--policy_name','ACT','--port','1','--robo_name','robotwin',
        '--test_num',str(item['episodes']),'--eval_video_log','False','--eval_global_view_log','False',
        '--output_dir',str(result_dir)]
    atomic_json(output.with_suffix('.command.json'),dict(command=command,identity=identity))
    from experiment.official_control.evaluate import stop
    with log.open('a') as stream:
        process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            if process.wait(timeout=config.get('seed_selection_timeout_seconds',1800)) != 0:
                raise RuntimeError('expert seed selection failed: '+str(log))
        finally:
            stop(process)
    selected=json.loads(output.read_text())
    if selected['identity']!=identity or selected['state']!='complete' or len(selected['seeds'])!=item['episodes']:
        raise ValueError('incomplete expert seed selection')
    return selected['seeds']


def main():
    # Remaining arguments are the unchanged real client's own CLI.
    root=Path(sys.argv[sys.argv.index('--client')+1])
    rest=sys.argv[sys.argv.index('--client')+2:]
    os.chdir(root);sys.path[:0]=[str(root),str(root/'script')]
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    task=rest[rest.index('--task_name')+1]
    import importlib
    importlib.import_module('envs.'+task)
    client=root/'script/eval_policy_client_lingbotvla.py'
    code=compile_expert_client(client.read_text(),str(client))
    # Start candidates are independent of the client's seed-group CLI encoding.
    namespace=dict(__name__='__main__',__file__=str(client))
    original=scan_expert
    def with_start(task,args,start,count):
        identity=json.loads(os.environ['LINGBOT_SEED_SCAN_IDENTITY'])['identity']
        return original(task,args,identity['start_seed'],count)
    # exec imports this module by package name; install the same callable there.
    import experiment.official_control.expert_seed_selection as module
    module.scan_expert=with_start
    sys.argv=[str(client),*rest]
    exec(code,namespace)


if __name__=='__main__':main()

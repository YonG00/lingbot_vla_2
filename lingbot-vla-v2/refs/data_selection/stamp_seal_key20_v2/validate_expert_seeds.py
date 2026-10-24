import ast,os,sys,json,random
from pathlib import Path
robo=Path('/data/experiments/official_adjust_bottle_control/RoboTwin-lingbot');os.chdir(robo);sys.path[:0]=[str(robo),str(robo/'script')]
os.environ['LD_LIBRARY_PATH']='/data/experiments/stack_bowls_sample_weighting/.runtime/oidn-2.5.1/oidn-2.5.1.x86_64.linux/lib:'+os.environ.get('LD_LIBRARY_PATH','')
random.seed(0)
import numpy as np
np.random.seed(0)
source=robo/'script/eval_policy_client_lingbotvla.py';tree=ast.parse(source.read_text(),str(source));replaced=0
body='''
args['eval_mode']=True
args['render_freq']=0
rows=[]
valid_seeds=[]
for candidate in range(100000,100020):
    error=None
    try:
        TASK_ENV.setup_demo(now_ep_num=len(valid_seeds),seed=candidate,is_test=True,**args)
        TASK_ENV.play_once()
        TASK_ENV.close_env()
        valid=bool(TASK_ENV.plan_success and TASK_ENV.check_success())
    except Exception as exc:
        error=repr(exc)
        valid=False
        try:TASK_ENV.close_env()
        except Exception:pass
    rows.append(dict(seed=candidate,expert_valid=valid,error=error,episode_index=len(valid_seeds)))
    print('EXPERT_SEED_RESULT',rows[-1],flush=True)
    if valid:valid_seeds.append(candidate)
    import json
    from pathlib import Path
    Path('/data/experiments/stack_bowls_sample_weighting/stamp_seal_after_can0915_queue/expert_seeds.json').write_text(json.dumps(dict(task='stamp_seal',validated_seeds=valid_seeds,checks=rows,policy_inference=False),indent=2))
    if len(valid_seeds)==3:return candidate,0
raise RuntimeError('fewer than three expert-valid seeds within bounded candidates')
'''
for node in ast.walk(tree):
 if isinstance(node,ast.FunctionDef)and node.name=='eval_policy':node.body=ast.parse(body).body;replaced+=1
 if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='model'for t in node.targets)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='WebsocketClientPolicy':node.value=ast.Constant(None);replaced+=1
assert replaced==2
sys.argv=[str(source),'--config','policy/ACT/deploy_policy.yml','--overrides','--task_name','stamp_seal','--task_config','demo_clean','--train_config_name','official_control','--seed','0','--policy_name','ACT','--port','1','--robo_name','robotwin','--eval_video_log','False','--eval_global_view_log','False','--output_dir','/data/experiments/stack_bowls_sample_weighting/stamp_seal_after_can0915_queue/expert_output','--test_num','1']
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),dict(__name__='__main__',__file__=str(source)))

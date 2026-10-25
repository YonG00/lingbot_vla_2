import json,os,sys,tempfile,unittest
import numpy as np
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from experiment.official_control.expert_seed_selection import compile_expert_client,scan_expert,select_seeds,identity_for

class UnStableError(Exception):pass

CLIENT='''
class WebsocketClientPolicy:
    def __init__(self, **kwargs): raise AssertionError('must not connect')
model=WebsocketClientPolicy(port=1)
def eval_policy(task_name,TASK_ENV,args,model,st_seed,test_num=3,**kwargs):
    TASK_ENV.play_once()
    if TASK_ENV.plan_success and TASK_ENV.check_success(): pass
'''

class SeedSelectionTests(unittest.TestCase):
    def fake_task(self, accepted=(), unstable=(), failure=None):
        class Task:
            def setup_demo(self,**kwargs):self.seed=kwargs['seed']
            def play_once(self):
                if failure:raise failure
                if self.seed in unstable:raise UnStableError()
                self.plan_success=self.seed in accepted
            def check_success(self):return self.seed in accepted
            def close_env(self):pass
        return Task()
    def scan(self,task,start=10,count=2,limit=8):
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/'seeds.json'
            mod=ModuleType('envs.utils.create_actor');mod.UnStableError=UnStableError
            env=dict(LINGBOT_SEED_SCAN_OUTPUT=str(output),LINGBOT_SEED_SCAN_IDENTITY=json.dumps(dict(identity={'task':'x'})),LINGBOT_SEED_SCAN_MAX_CANDIDATES=str(limit))
            with patch.dict(sys.modules,{'envs.utils.create_actor':mod}),patch.dict(os.environ,env):
                scan_expert(task,{},start,count)
            return json.loads(output.read_text())
    def test_rejections_replaced_before_exact_valid_count(self):
        r=self.scan(self.fake_task(accepted=[11,14],unstable=[12]))
        self.assertEqual(r['seeds'],[11,14]);self.assertEqual(len(r['candidates']),5)
        self.assertEqual(r['state'],'complete')
        self.assertEqual(r['candidates'][2]['reason'],'unstable_scene')
    def test_infrastructure_errors_not_silently_retried(self):
        with self.assertRaisesRegex(RuntimeError,'renderer failed'):
            self.scan(self.fake_task(failure=RuntimeError('renderer failed')))
    def test_candidate_limit(self):
        with self.assertRaisesRegex(RuntimeError,'candidate limit exhausted'):
            self.scan(self.fake_task(),limit=3)
    def test_expert_client_never_connects_policy_server(self):
        namespace={};exec(compile_expert_client(CLIENT,'client.py'),namespace)
        self.assertIsNone(namespace['model'])
        with patch('experiment.official_control.expert_seed_selection.scan_expert',return_value=(17,0)) as scan:
            self.assertEqual(namespace['eval_policy']('x',object(),{},None,10,test_num=2),(17,0))
            self.assertEqual(scan.call_args.args[-2:],(10,2))
    def test_contract_drift_fails(self):
        for source in [CLIENT.replace('check_success()','other()'),CLIENT.replace('WebsocketClientPolicy(port=1)','object()')]:
            with self.assertRaises(ValueError):compile_expert_client(source,'client.py')
    def test_cached_seed_identity_and_fixed_override(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t)
            for name in ['script/eval_policy_client_lingbotvla.py','task_config/demo_clean.yml','envs/x.py']:
                p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test')
            c=dict(robotwin=str(r),seed_selection='expert');item=dict(task='x',task_config='demo_clean',episodes=2,key='demo_clean/x',seeds=[1,2])
            identity=identity_for(c,item)
            cache=r/'valid_seeds/demo_clean/x.json';cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps(dict(identity=identity,state='complete',seeds=[12,16])))
            self.assertEqual(select_seeds(c,item,r),[12,16])
            (r/'envs/x.py').write_text('changed')
            with self.assertRaises(ValueError):select_seeds(c,item,r)
            c['scene_seeds']={'demo_clean/x':[1,2]}
            self.assertEqual(select_seeds(c,item,r),[1,2])

if __name__=='__main__':unittest.main()

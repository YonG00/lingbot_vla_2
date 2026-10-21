import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiment.task_recipes.resident_evaluation import unwrap,validate_handoff,jobs_for,serve_and_evaluate,read_completed
from experiment.task_recipes.test_contracts import fixture
from experiment.task_recipes.schema import validate

class ResidentTests(unittest.TestCase):
    def test_training_entry_injects_callback_only_when_requested(self):
        import sys
        from experiment.task_recipes import run
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)
            (out/'trainer_config.json').write_text(json.dumps(dict(model=dict(tokenizer_path='/tmp/qwen'))))
            (out/'cpu_report.json').write_text(json.dumps(dict(state='cpu_pass')))
            for enabled in [False,True]:
                c=self.config();c['evaluation']['reuse_training_model']=enabled
                with patch.object(sys,'argv',['run','--stage','train','--out',name]),patch.object(run,'load_run',return_value=(c,{},{})),patch.object(run,'install'),patch.object(run.os,'chdir'),patch('runpy.run_path') as entry:
                    run.main()
                initial=entry.call_args.kwargs['init_globals']
                self.assertEqual('_POST_TRAINING_CALLBACK' in initial,enabled)
    def test_training_json_joints_match_deployment_literal_parser(self):
        import ast
        from experiment.task_recipes.resident_evaluation import deployment_data_config
        source=dict(joints=[{'arm.position':14},"{'gripper.position': 2}"],norm_type=[{'arm.position':'bounds_99_woclip'}],cameras=['head'])
        c=deployment_data_config(source)
        self.assertEqual([ast.literal_eval(x) for x in c.joints],[{'arm.position':14},{'gripper.position':2}])
        self.assertIsInstance(source['joints'][0],dict)
        self.assertEqual(ast.literal_eval(c.norm_type[0]),{'arm.position':'bounds_99_woclip'})
    def test_prepare_view_keeps_weight_modules_and_clears_training_cache(self):
        from experiment.task_recipes.resident_evaluation import prepare_view
        class Fake:
            def __init__(self):
                self.model=SimpleNamespace()
                self.config=SimpleNamespace(attention_implementation='flex_cached',precompute_grid_thw=True,use_cache=False)
                self.pos_embeds=object();self.disabled=False
            def parameters(self):return []
            def modules(self):return [self]
            def gradient_checkpointing_disable(self):self.disabled=True
            def get_attention_interface(self):return self.config.attention_implementation
            def float(self):return self
            def eval(self):return self
        class Inference(Fake):pass
        model=Fake();view=prepare_view(model,Inference)
        self.assertIs(view.model,model.model);self.assertTrue(model.disabled)
        self.assertIsNone(view.pos_embeds);self.assertEqual(view.attention_interface,'eager')
        self.assertFalse(view.config.precompute_grid_thw);self.assertTrue(view.config.use_cache)
    def config(self):
        c=fixture('place_container_plate_200')
        c['evaluation'].update(reuse_training_model=True,shared_model_session=True,checkpoints=['final'])
        return validate(c)
    def test_schema(self):
        self.assertNotIn('reuse_training_model',validate(fixture('place_container_plate_200'))['evaluation'])
        for change in [dict(reuse_training_model=1),dict(shared_model_session=False),dict(tasks=[])]:
            c=self.config();c['evaluation'].update(change)
            with self.assertRaises(ValueError):validate(c)
    def test_identity_and_guards(self):
        core=object();self.assertIs(unwrap(SimpleNamespace(_orig_mod=core)),core)
        c=self.config();validate_handoff(c,200)
        with self.assertRaises(ValueError):validate_handoff(c,100)
        with patch.dict(os.environ,WORLD_SIZE='2'):
            with self.assertRaises(ValueError):validate_handoff(c,200)
    def test_jobs(self):
        c=self.config();c['evaluation'].update(max_parallel_tasks=8,scheduling='rolling',video_episodes=2)
        jobs=jobs_for(c,Path('/tmp/run'),200,dict(model=dict(tokenizer_path='/tmp/qwen')))
        self.assertEqual([j['task'] for j in jobs],c['evaluation']['tasks'])
        self.assertTrue(all(j['episodes']==3 and j['video_episodes']==2 for j in jobs))
    def test_lifetime(self):
        events=[]
        class Server:
            async def run(self):
                events.append('serve')
                try:await asyncio.Event().wait()
                finally:events.append('stop')
        def work():
            self.assertEqual(events,['serve']);events.append('simulators');return 7
        self.assertEqual(asyncio.run(serve_and_evaluate(Server(),work)),7)
        self.assertEqual(events,['serve','simulators','stop'])
    def test_failure(self):
        class Server:
            async def run(self):await asyncio.Event().wait()
        def work():raise RuntimeError('simulator failed')
        with self.assertRaisesRegex(RuntimeError,'simulator failed'):asyncio.run(serve_and_evaluate(Server(),work))
    def test_read_completed(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as name:
            d=Path(name)/'evaluation_200';d.mkdir()
            status=dict(state='complete',reused_training_model=True,model_loads=0)
            handoff=dict(training_pid=42,server_pid=42,weight_loads=0)
            results=[dict(task=t,server_pid=42,episodes=3,model=str(d.parent/'checkpoints/global_step_200/hf_ckpt')) for t in c['evaluation']['tasks']]
            for n,v in [('status',status),('handoff',handoff),('summary',results)]:
                (d/(n+'.json')).write_text(json.dumps(v))
            self.assertEqual(read_completed(d,c,200),results)
            handoff['server_pid']=43;(d/'handoff.json').write_text(json.dumps(handoff))
            with self.assertRaises(ValueError):read_completed(d,c,200)
if __name__=='__main__':unittest.main()

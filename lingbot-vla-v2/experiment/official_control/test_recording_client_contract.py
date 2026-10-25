"""Exercise the actual RoboTwin client recording scope without GPU/sapien imports."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest

CLIENT=Path(__file__).resolve().parents[3]/'RoboTwin-lingbot/script/eval_policy_client_lingbotvla.py'

class ClientRecordingContract(unittest.TestCase):
    def run_scope(self, enabled, fail=False):
        tree=ast.parse(CLIENT.read_text())
        scopes=[n for n in ast.walk(tree) if isinstance(n,ast.With) and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='_start_eval_global_video' for c in ast.walk(n))]
        self.assertEqual(len(scopes),1,'actual client must own the recording lifecycle')
        code='def run_scope():\n'+''.join('    '+line+'\n' for line in ast.unparse(scopes[0]).splitlines())
        events=[]
        class Task:
            eval_video_path=None
            test_num=0
            take_action_cnt=0
            step_lim=1
            eval_success=False
            def _start_eval_global_video(self,*args,**kw):events.append('start')
            def _stop_eval_global_video(self,result='interrupted'):events.append('stop:'+result)
            def get_obs(self):return dict(observation={k:dict(rgb=None) for k in ('head_camera','left_camera','right_camera')},joint_action=dict(vector=None))
            def get_instruction(self):return 'test'
            def take_action(self,action):self.take_action_cnt+=1;self.eval_success=True
        class Model:
            def infer(self,observation):
                events.append('reset' if observation.get('reset') else 'infer')
                if fail:raise RuntimeError('inference failure')
                return dict(action=SimpleNamespace(shape=(55,)),server_timing={})
        env=dict(ExitStack=contextlib.ExitStack,Path=Path,TASK_ENV=Task(),model=Model(),usr_args=dict(robo_name='robotwin'),args=dict(eval_global_view_log=enabled,eval_global_view_save_dir='/tmp/video',global_view_fps=2))
        exec(compile(code,'actual_client_recording_scope','exec'),env)
        if fail:
            with self.assertRaisesRegex(RuntimeError,'inference failure'):env['run_scope']()
        else:env['run_scope']()
        return events
    def test_global_video_starts_before_reset_and_finishes(self):
        events=self.run_scope(True)
        self.assertEqual(events[:3],['start','reset','infer']);self.assertIn('stop:success',events)
    def test_inference_exception_closes_encoder(self):
        self.assertEqual(self.run_scope(True,fail=True),['start','reset','stop:interrupted'])
    def test_no_video_does_not_create_encoder(self):
        self.assertEqual(self.run_scope(False),['reset','infer'])
    def test_cli_recording_options_reach_task_args(self):
        tree=ast.parse(CLIENT.read_text());assigned={}
        for n in ast.walk(tree):
            if isinstance(n,ast.Assign):
                for t in n.targets:
                    if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='args' and isinstance(t.slice,ast.Constant):assigned[t.slice.value]=ast.unparse(n.value)
        for key in ('eval_global_view_log','global_view_fps','eval_global_view_save_dir'):self.assertIn(key,assigned)
        self.assertIn('usr_args',assigned['eval_global_view_log'])

if __name__=='__main__':unittest.main()

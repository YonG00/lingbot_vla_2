import atexit
import json
from pathlib import Path
import random
import tempfile
import unittest
from experiment.official_control.render_sync import install_render_sync,install_policy_trace


def task_class():
    class Task:
        def _init_task_env_(self):
            self.frame=self.syncs=0;self.captures=[];self.eval_mode=True
            self.render_freq=0;self.crazy_random_light=False;self.eval_video_path=None
            self.failure=False
        def _start_eval_global_video(self):pass
        def _update_render(self):
            self.syncs+=1
            if self.crazy_random_light:random.random()
        def get_obs(self):
            self._update_render();return dict(frame=self.frame)
        def take_action(self,n):
            self._update_render()
            for _ in range(n):
                self.frame+=1;self._update_render()
                if self.frame%5==0:self.captures.append(self.frame)
                if self.failure:raise RuntimeError('action failed')
                if self.frame==17:return self.get_obs()
            self._update_render()
    return Task


class RenderSyncTests(unittest.TestCase):
    def run_mode(self,mode,**settings):
        with tempfile.TemporaryDirectory() as d:
            cls=task_class();stats,flush=install_render_sync(cls,mode,Path(d)/'audit.json')
            try:
                task=cls();task._init_task_env_()
                for k,v in settings.items():setattr(task,k,v)
                task._start_eval_global_video();initial=task.get_obs();final=task.take_action(30)
                flush();self.assertEqual(json.loads((Path(d)/'audit.json').read_text()),stats)
                return task,stats,initial,final,random.getstate()
            finally:atexit.unregister(flush)

    def test_deferred_sync_keeps_physics_observation_success_and_capture_steps(self):
        eager,a,first,last,_=self.run_mode('eager')
        deferred,b,other_first,other_last,_=self.run_mode('on_demand')
        self.assertEqual((first,last),(other_first,other_last))
        self.assertEqual(eager.frame,deferred.frame)
        self.assertEqual(eager.captures,deferred.captures)
        self.assertGreater(a['actual_sync_calls'],b['actual_sync_calls'])
        self.assertGreater(b['skipped_sync_calls'],0)
        self.assertEqual(b['actual_sync_calls'],2)

    def test_random_light_viewer_and_head_video_keep_original_sync(self):
        for settings in [dict(crazy_random_light=True),dict(render_freq=1),dict(eval_video_path='/video'),dict(eval_mode=False)]:
            random.seed(31);a,sa,_,_,rng_a=self.run_mode('eager',**settings)
            random.seed(31);b,sb,_,_,rng_b=self.run_mode('on_demand',**settings)
            self.assertEqual(a.syncs,b.syncs);self.assertEqual(rng_a,rng_b)
            self.assertEqual(sb['skipped_sync_calls'],0)

    def test_expert_phase_and_exception_do_not_leak_deferral(self):
        with tempfile.TemporaryDirectory() as d:
            cls=task_class();stats,flush=install_render_sync(cls,'on_demand',Path(d)/'audit.json')
            try:
                task=cls();task._init_task_env_();task.take_action(2)
                self.assertEqual(stats['skipped_sync_calls'],0)
                task._start_eval_global_video();task.failure=True
                with self.assertRaises(RuntimeError):task.take_action(1)
                self.assertEqual(task._render_action_depth,0)
                before=task.syncs;task._update_render();self.assertEqual(task.syncs,before+1)
                task._init_task_env_();self.assertFalse(task._render_policy_active)
            finally:atexit.unregister(flush)

    def test_trace_excludes_latency_and_preserves_payload_and_return(self):
        class Packer:
            def pack(self,value):return json.dumps(value,sort_keys=True).encode()
        class Client:
            _packer=Packer();calls=0
            def infer(self,obs):
                self.calls+=1;return dict(action=[obs['state']],server_timing=self.calls)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.jsonl';install_policy_trace(Client,p)
            client=Client();obs=dict(state=3)
            a=client.infer(obs);b=client.infer(obs)
            rows=[json.loads(x) for x in p.read_text().splitlines()]
            self.assertEqual(rows[0],rows[1]);self.assertNotEqual(a['server_timing'],b['server_timing'])
            self.assertEqual(obs,dict(state=3));self.assertEqual(client.calls,2)


if __name__=='__main__':unittest.main()

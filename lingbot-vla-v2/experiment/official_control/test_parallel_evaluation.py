from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from experiment.official_control.session_policy import SessionPool
from experiment.official_control.parallel_evaluation import run_jobs,resource_failure,compare_results,probe_jobs


class FakePolicy:
    chunk_ret=True
    def __init__(self):
        self.vla=SimpleNamespace(feature_transform=None);self.global_step=0
        self.last_action_chunk=None;self.last_normalized_action_chunk=None
    def infer(self,obs):
        if obs.get('reset'):
            self.global_step=0;self.robot_config=obs['robot'];self.action_key=obs['robot']
            self.vla.feature_transform=obs['robot'];return dict(action=None)
        self.global_step+=1;self.last_action_chunk=obs['value']
        if obs.get('fail'):raise RuntimeError('inference failure')
        return dict(robot=self.vla.feature_transform,step=self.global_step,value=self.last_action_chunk)


class ParallelTests(unittest.TestCase):
    def test_sessions_keep_resets_caches_and_robot_separate(self):
        policy=FakePolicy();pool=SessionPool(policy);a,b=pool.connect(),pool.connect()
        a.infer(dict(reset=True,robot='a'));b.infer(dict(reset=True,robot='b'))
        self.assertEqual(a.infer(dict(value=1)),dict(robot='a',step=1,value=1))
        self.assertEqual(b.infer(dict(value=2)),dict(robot='b',step=1,value=2))
        a.infer(dict(reset=True,robot='a'))
        self.assertEqual(b.infer(dict(value=3)),dict(robot='b',step=2,value=3))
        self.assertIs(a.pool.policy.vla,b.pool.policy.vla)
        self.assertEqual(policy.global_step,0);self.assertIsNone(policy.vla.feature_transform)
        self.assertFalse(hasattr(policy,'robot_config'))

    def test_new_session_and_failure_contract(self):
        policy=FakePolicy();s=SessionPool(policy).connect()
        with self.assertRaises(ValueError):s.infer(dict(value=1))
        with self.assertRaises(ValueError):s.infer(dict(reset=True,path_to_pi_model='/new'))
        s.infer(dict(reset=True,robot='a'))
        with self.assertRaises(RuntimeError):s.infer(dict(value=1,fail=True))
        self.assertEqual(policy.global_step,0);self.assertIsNone(policy.vla.feature_transform)

    def test_three_simulators_share_one_server_and_really_overlap(self):
        self.check_overlap(3)

    def test_six_simulators_share_one_server_and_really_overlap(self):
        self.check_overlap(6)

    def test_eight_simulators_share_one_server_and_really_overlap(self):
        self.check_overlap(8)

    def test_rolling_replaces_finished_slot_before_slow_tasks_finish(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);barrier=threading.Barrier(8);replaced=threading.Event();lock=threading.Lock()
            finished=set();opened=[];stopped=[];active=0;peak=0
            def execute(job,shared):
                nonlocal active,peak
                i=int(job['task'])
                with lock:active+=1;peak=max(peak,active)
                if i<8:
                    barrier.wait(timeout=5)
                    if i: self.assertTrue(replaced.wait(timeout=5))
                elif i==8:
                    with lock:self.assertNotIn(1,finished)
                    replaced.set()
                Path(job['destination'],'summary.json').write_text(json.dumps(dict(state='complete',task=job['task'],server_pid=42,shared_server=True)))
                with lock:active-=1;finished.add(i)
                return 0
            def open_(*a):opened.append(1);return dict(process=SimpleNamespace(pid=42))
            jobs=[dict(task=str(i),model='/m',destination=str(root/str(i))) for i in range(10)]
            results=run_jobs(jobs,root,8,open_,execute,lambda _:stopped.append(1),scheduling='rolling')
            self.assertEqual(peak,8);self.assertEqual(len(results),10)
            self.assertEqual(opened,[1]);self.assertEqual(stopped,[1])

    def check_overlap(self,workers):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);barrier=threading.Barrier(workers);events=[];lock=threading.Lock()
            shared=dict(process=SimpleNamespace(pid=42));active=0;peak=0
            def execute(job,shared):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                barrier.wait(timeout=5)
                dest=Path(job['destination'])
                dest.joinpath('summary.json').write_text(json.dumps(dict(state='complete',task=job['task'],server_pid=42,shared_server=True)))
                with lock:active-=1
                return 0
            def open_(*args):events.append('open');return shared
            jobs=[dict(task=str(t),model='/model',destination=str(root/str(t))) for t in range(workers)]
            run_jobs(jobs,root,workers,open_,execute,lambda _:events.append('stop'))
            self.assertEqual(peak,workers);self.assertEqual(events,['open','stop'])
            self.assertEqual(json.loads((root/'status.json').read_text())['model_loads'],1)

    def test_failures_stop_server_and_oom_detection_is_specific(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);events=[]
            def fail(*a,**kw):raise RuntimeError('client failed')
            with self.assertRaises(RuntimeError):run_jobs([dict(task='a',model='/m',destination=str(root/'a'))],root,3,
                lambda *a:dict(process=SimpleNamespace(pid=42)),fail,lambda _:events.append('stop'))
            self.assertEqual(events,['stop'])
            (root/'server.log').write_text('some unrelated failure')
            self.assertFalse(resource_failure(root))
            (root/'server.log').write_text('CUDA out of memory')
            self.assertTrue(resource_failure(root))

    def test_ten_tasks_use_six_then_four_without_reloading(self):
        self.check_batches(6)

    def test_ten_tasks_use_five_then_five_without_reloading(self):
        self.check_batches(5)

    def check_batches(self,workers):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);barriers=[threading.Barrier(workers),threading.Barrier(10-workers)]
            lock=threading.Lock();finished=set();opened=[];stopped=[]
            def execute(job,shared):
                index=int(job['task']);batch=0 if index<workers else 1
                if batch:
                    with lock:self.assertTrue(set(range(workers))<=finished)
                barriers[batch].wait(timeout=5)
                Path(job['destination'],'summary.json').write_text(json.dumps(dict(state='complete',task=job['task'],server_pid=42,shared_server=True)))
                with lock:finished.add(index)
                return 0
            def open_(*a):opened.append(1);return dict(process=SimpleNamespace(pid=42))
            jobs=[dict(task=str(i),model='/m',destination=str(root/str(i))) for i in range(10)]
            results=run_jobs(jobs,root,workers,open_,execute,lambda _:stopped.append(1))
            self.assertEqual(len(results),10);self.assertEqual(opened,[1]);self.assertEqual(stopped,[1])

    def test_single_episode_resource_probe_drops_downstream_reference(self):
        original=dict(task='a',episodes=1,model='/m',reference_summary='/three_episode_report',expected_seeds=[100000])
        raw=dict(jobs=[original],compare_reference=False)
        actual=probe_jobs(raw,Path('/attempt'))[0]
        self.assertNotIn('reference_summary',actual)
        self.assertFalse(actual['require_baseline'])
        self.assertEqual(actual['expected_seeds'],[100000])
        self.assertIn('reference_summary',original)
        raw['compare_reference']=True
        self.assertEqual(probe_jobs(raw,Path('/attempt'))[0]['reference_summary'],'/three_episode_report')

    def test_comparison_detects_seed_or_success_or_model_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'serial.json';baseline=dict(task='a',successes=1,seeds=[1,2,3],model='/model')
            p.write_text(json.dumps(baseline));jobs=[dict(task='a',reference_summary=str(p))]
            self.assertTrue(compare_results([baseline],jobs)[0]['matched'])
            for changed in [dict(successes=0),dict(seeds=[3,2,1]),dict(model='/wrong')]:
                self.assertFalse(compare_results([dict(baseline,**changed)],jobs)[0]['matched'])

if __name__=='__main__':unittest.main()

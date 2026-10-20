import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from experiment.official_control.shared_evaluation import run_jobs

class SharedTests(unittest.TestCase):
    def test_two_tasks_one_open_and_close_and_same_session(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);events=[];process=SimpleNamespace(pid=42)
            session=dict(process=process,model='/model')
            def open_(job,directory):events.append('open');return session
            def execute(job,shared):
                self.assertIs(shared,session)
                events.append(job['task'])
                (Path(job['destination'])/'summary.json').write_text(json.dumps(dict(task=job['task'],server_pid=42,shared_server=True)))
                return 0
            jobs=[dict(task=t,model='/model',destination=str(root/t)) for t in ['adjust_bottle','click_bell']]
            run_jobs(jobs,root,open_,execute,lambda p:events.append('stop'))
            self.assertEqual(events,['open','adjust_bottle','click_bell','stop'])
            self.assertEqual(json.loads((root/'status.json').read_text())['model_loads'],1)
    def test_failure_always_stops_shared_server_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);events=[]
            def execute(job,shared):raise RuntimeError('simulation failed')
            with self.assertRaises(RuntimeError):
                run_jobs([dict(task='adjust_bottle',model='/model',destination=str(root/'task'))],root,
                    lambda *a:dict(process=SimpleNamespace(pid=42)),execute,lambda p:events.append('stop'))
            self.assertEqual(events,['stop'])
            self.assertEqual(json.loads((root/'status.json').read_text())['state'],'failed')
    def test_different_models_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):
                run_jobs([dict(task='a',model='/a'),dict(task='b',model='/b')],Path(d),lambda *a:self.fail('loaded wrong models'))

if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from experiment.official_control.model_benchmark import plan, jobs_for, report, run, atomic_json, configure, reference_results


class BenchmarkTests(unittest.TestCase):
    def config(self, root):
        model = root/'model';model.mkdir()
        (model/'config.json').write_text('{}')
        (model/'model.safetensors').write_bytes(b'test')
        return dict(tasks=['task_a','task_b'],expected_task_count=2,model=str(model),
                    destination=str(root/'results'),robotwin='/sim',qwen='/qwen',sim_python='/python')

    def test_single_mode_and_parallel_limits(self):
        c=configure(dict(destination='/tmp/out'),mode='single',task='stack_bowls_two',episodes=7,workers=2)
        items=plan(c)
        self.assertEqual(len(items),1)
        self.assertEqual((items[0]['episodes'],items[0]['workers'],items[0]['video_episodes']),(7,2,0))
        c=configure(c,mode='single',workers=8)
        self.assertEqual(plan(c)[0]['workers'],7)
        c=configure(c,mode='single',workers=1,task_config='both')
        self.assertEqual(len(plan(c)),2)
        self.assertTrue(all(i['workers']==1 for i in plan(c)))
        with self.assertRaises(ValueError):configure(dict(),mode='single')
        with self.assertRaises(ValueError):configure(dict(tasks=['a']),mode='benchmark',task='a')

    def test_reference_collect_results_arithmetic_and_percentage(self):
        items=plan(dict(tasks=['a','b'],expected_task_count=2))
        rows=[dict(key=i['key'],task_config=i['task_config'],state='complete',
                   episodes=3,successes=k,elapsed_seconds=2.0) for k,i in enumerate(items)]
        result=reference_results(items,rows,9.5)
        self.assertEqual(result['overall']['successes'],6)
        self.assertEqual(result['overall']['episode_count'],12)
        self.assertEqual(result['overall']['success_rate_percent'],50.0)
        self.assertEqual(result['task_configs']['demo_clean']['success_rate_percent'],100/6)
        self.assertEqual(result['tasks'][3]['success_rate_percent'],100.0)
        self.assertEqual(result['status'],'complete')
        rows[0]=dict(key=items[0]['key'],state='error',error='renderer failed')
        result=reference_results(items,rows,10)
        self.assertEqual(result['overall']['episode_count'],9)
        self.assertIsNone(result['tasks'][0]['successes'])
        self.assertEqual(result['rate_scope'],'completed_tasks_only')
        self.assertEqual(result['overall']['error_task_count'],1)

    def test_defaults_100_items_300_episodes(self):
        items=plan(dict(tasks=['task_'+str(i) for i in range(50)]))
        self.assertEqual(len(items),100)
        self.assertEqual(sum(i['episodes'] for i in items),300)
        self.assertTrue(all(i['video_episodes']==0 and i['workers']==3 for i in items))
        self.assertEqual(len({i['key'] for i in items}),100)

    def test_bad_inputs(self):
        for change in [dict(episodes_per_task=True),dict(episode_workers=9),dict(video_episodes=4),
                       dict(tasks=['a','a']),dict(task_configs=['demo_random']),dict(scene_seeds={'bad':[]} )]:
            c=dict(tasks=['a'],expected_task_count=1);c.update(change)
            with self.assertRaises(ValueError):plan(c)

    def test_per_config_seeds_and_shards(self):
        c=dict(tasks=['a'],expected_task_count=1,scene_seeds={'demo_randomized/a':[10,12,17]})
        i=plan(c)[1]
        c.update(model='/m',robotwin='/r',qwen='/q',sim_python='/p')
        shards=jobs_for(c,i,Path('/tmp/output'))
        self.assertEqual([j['scene_seed'] for j in shards],[10,12,17])
        self.assertTrue(all(j['task_config']=='demo_randomized' and j['video_episodes']==0 for j in shards))
        self.assertEqual(len({j['destination'] for j in shards}),3)

    def test_one_episode_explicit_shard(self):
        c=dict(tasks=['a'],expected_task_count=1,episodes_per_task=1,
               episode_workers=1,model='/m',robotwin='/r',qwen='/q',sim_python='/p')
        self.assertEqual(jobs_for(c,plan(c)[0],Path('/tmp/out'))[0]['scene_seed'],100000)

    def fake_runner(self,jobs,output,workers,**kwargs):
        self.assertEqual(workers,3)
        self.assertEqual(kwargs['scheduling'],'rolling')
        row=dict(state='complete',task=jobs[0]['task'],task_config=jobs[0]['task_config'],
                 model=jobs[0]['model'],episodes=3,successes=1,success_rate=1/3,
                 seeds=[j['scene_seed'] for j in jobs],episode_results=[])
        dest=Path(jobs[0]['aggregate_destination']);dest.mkdir(parents=True)
        atomic_json(dest/'summary.json',row)
        return [row]

    def test_one_model_all_items_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            c=self.config(Path(t));opens=[];stops=[];calls=[]
            process=SimpleNamespace(poll=lambda:None)
            def opener(*a):opens.append(1);return dict(process=process)
            def runner(*a,**kw):calls.append(1);return self.fake_runner(*a,**kw)
            self.assertTrue(run(c,opener=opener,runner=runner,stopper=lambda p:stops.append(p)))
            self.assertEqual((len(opens),len(stops),len(calls)),(1,1,4))
            result=json.loads((Path(c['destination'])/'benchmark_audit.json').read_text())
            self.assertEqual(result['groups']['overall']['evaluated_episodes'],12)
            public=json.loads((Path(c['destination'])/'benchmark_results.json').read_text())
            self.assertEqual(public['overall']['episode_count'],12)
            self.assertAlmostEqual(public['overall']['success_rate_percent'],100/3)
            self.assertNotIn('fingerprint',public)
            self.assertEqual(len(public['tasks']),4)
            self.assertTrue(run(c,resume=True,opener=opener,runner=runner,stopper=lambda _:None))
            self.assertEqual(len(opens),1)
            c['episodes_per_task']=4
            with self.assertRaises(ValueError):run(c,resume=True)

    def test_errors_not_counted_as_model_failures(self):
        with tempfile.TemporaryDirectory() as t:
            c=self.config(Path(t));process=SimpleNamespace(poll=lambda:None)
            def runner(jobs,*a,**kw):
                if jobs[0]['task']=='task_a':raise RuntimeError('expert seed rejected')
                return self.fake_runner(jobs,*a,**kw)
            self.assertFalse(run(c,opener=lambda *_:dict(process=process),runner=runner,stopper=lambda _:None))
            result=json.loads((Path(c['destination'])/'benchmark_audit.json').read_text())
            g=result['groups']['overall']
            self.assertEqual((g['completed_items'],g['error_items'],g['evaluated_episodes']),(2,2,6))
            self.assertEqual(g['success_rate'],1/3)
            self.assertEqual(result['state'],'incomplete')

    def test_resume_rejects_missing_or_altered_summary(self):
        with tempfile.TemporaryDirectory() as t:
            c=self.config(Path(t));p=SimpleNamespace(poll=lambda:None)
            run(c,opener=lambda *_:dict(process=p),runner=self.fake_runner,stopper=lambda _:None)
            result=json.loads((Path(c['destination'])/'benchmark_audit.json').read_text())
            path=Path(result['items'][0]['summary_path']);row=json.loads(path.read_text());row['task_config']='demo_randomized';atomic_json(path,row)
            with self.assertRaises(ValueError):run(c,resume=True,opener=lambda *_:dict(process=p))

    def test_random_and_clean_aggregate_separately(self):
        items=plan(dict(tasks=['a'],expected_task_count=1))
        rows=[dict(key=i['key'],task_config=i['task_config'],state='complete',episodes=3,successes=k) for k,i in enumerate(items)]
        r=report(items,rows)
        self.assertEqual(r['groups']['demo_clean']['success_rate'],0)
        self.assertEqual(r['groups']['demo_randomized']['success_rate'],1/3)
        self.assertEqual(r['groups']['overall']['success_rate'],1/6)

if __name__=='__main__':unittest.main()

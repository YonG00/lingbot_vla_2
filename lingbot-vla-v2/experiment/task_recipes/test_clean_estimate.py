import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from experiment.task_recipes.clean_estimate import estimate,choose_retention
class EstimateTests(unittest.TestCase):
    def test_task_mean_not_episode_weighting_or_best_history(self):
        old=[dict(task='a',successes=4,episodes=4),dict(task='b',successes=1,episodes=3)]
        r=estimate(['a','b'],old,[dict(task='a',successes=0,episodes=3)])
        self.assertAlmostEqual(r['macro_success_rate'],1/6);self.assertEqual(r['current_model_tasks'],1)
        self.assertTrue(r['not_full_current_model_benchmark'])
    def test_missing_never_assumed_failure(self):
        r=estimate(['a','b'],[dict(task='a',successes=1,episodes=1)],[])
        self.assertEqual(r['macro_success_rate'],1);self.assertEqual(r['missing_tasks'],['b'])
    def test_any_success_qualifies(self):
        p=dict(tasks=[dict(task='old',observations=[dict(successes=1,episodes=4)]),dict(task='new',observations=[dict(successes=3,episodes=3)]),dict(task='fail',observations=[dict(successes=0,episodes=3)])])
        self.assertEqual(choose_retention(p,{'new'},7,1),['old'])
    def test_duplicate_current_rejected(self):
        r=dict(task='a',successes=1,episodes=3)
        with self.assertRaises(ValueError):estimate(['a'],[],[r,r])
    def test_post_queue_waits_for_success_and_replaces_only_retention_tasks(self):
        from experiment.task_recipes import post_evaluation
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);model=str(root/'model');old=['a','b','c']
            (root/'history.json').write_text(json.dumps(dict(tasks=old+['new'],historical_latest=[dict(task=t,successes=1,episodes=3,source='history')for t in old])))
            pipeline=root/'previous.json'
            primary=[dict(task='new',successes=0,episodes=3,model=model)]
            pipeline.write_text(json.dumps(dict(state='complete',evaluation=primary)))
            c=dict(destination=str(root/'report'),historical_file='history.json',wait_for_pipeline=str(pipeline),primary_tasks=['new'],model=model,robotwin='robotwin',qwen='qwen',sim_python='python',selection=dict(chosen_tasks=old))
            results=[dict(task=t,successes=0,episodes=3,seeds=[1,2,3],model=model)for t in old]
            with patch.object(post_evaluation,'ROOT',root),patch.object(post_evaluation.subprocess,'check_output',return_value=b''),patch('experiment.official_control.baseline_pipeline.verify_checkpoint'),patch('experiment.official_control.parallel_evaluation.run_jobs',return_value=results) as runner:
                post_evaluation.execute(c)
            self.assertEqual(runner.call_args.args[2],3)
            report=json.loads((root/'report/report.json').read_text())
            self.assertEqual(report['before']['macro_success_rate'],.25)
            self.assertEqual(report['after']['macro_success_rate'],0)
            self.assertEqual(report['after']['current_model_tasks'],4)
            c['destination']=str(root/'failed');pipeline.write_text(json.dumps(dict(state='failed')))
            with patch.object(post_evaluation,'ROOT',root),patch.object(post_evaluation.subprocess,'check_output') as gpu:
                with self.assertRaises(RuntimeError):post_evaluation.execute(c)
                gpu.assert_not_called()
if __name__=='__main__':unittest.main()

import unittest
from experiment.task_recipes.schema import validate
from experiment.task_recipes.test_contracts import fixture
from experiment.task_recipes.remediation import next_recipe


class EvaluationTasks(unittest.TestCase):
    def test_default_follows_curriculum_order(self):
        raw=fixture('sequential_plate100_cup200_pot200')
        del raw['evaluation']['tasks']
        raw['train']['task_order'].reverse()
        c=validate(raw)
        self.assertEqual(c['evaluation']['tasks'],c['train']['task_order'])

    def test_explicit_retention_tasks_remain_supported(self):
        raw=fixture('place_container_plate_200')
        raw['evaluation']['tasks']=['adjust_bottle']
        self.assertEqual(validate(raw)['evaluation']['tasks'],['adjust_bottle'])

    def test_parallel_tasks_parameter_is_explicit_and_bounded(self):
        raw=fixture('place_container_plate_200')
        self.assertEqual(validate(raw)['evaluation']['max_parallel_tasks'],1)
        raw['evaluation'].update(max_parallel_tasks=6,shared_model_session=True)
        self.assertEqual(validate(raw)['evaluation']['max_parallel_tasks'],6)
        for bad in [0,9,True,1.5]:
            raw['evaluation']['max_parallel_tasks']=bad
            with self.assertRaises(ValueError):validate(raw)
        raw['evaluation'].update(max_parallel_tasks=6,shared_model_session=False)
        with self.assertRaises(ValueError):validate(raw)

    def test_eight_rolling_first_two_videos_keeps_three_episodes(self):
        raw=fixture('place_container_plate_200')
        raw['evaluation'].update(max_parallel_tasks=8,shared_model_session=True,scheduling='rolling',video_episodes=2)
        c=validate(raw)
        self.assertEqual(c['evaluation']['episodes_per_task'],3)
        self.assertEqual(c['evaluation']['video_episodes'],2)
        for change in [dict(scheduling='unknown'),dict(video_episodes=-1),dict(video_episodes=True),dict(max_parallel_tasks=1)]:
            modified={**raw,'evaluation':{**raw['evaluation'],**change}}
            with self.assertRaises(ValueError):validate(modified)

    def test_remediation_does_not_inherit_untrained_tasks(self):
        c=validate(fixture('sequential_plate100_cup200_pot200'))
        lesson=[dict(task='place_empty_cup',steps=100)]
        child=validate(next_recipe(c,c['output_dir'],lesson,1))
        self.assertEqual(child['evaluation']['tasks'],['place_empty_cup'])

if __name__=='__main__':unittest.main()

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

    def test_remediation_does_not_inherit_untrained_tasks(self):
        c=validate(fixture('sequential_plate100_cup200_pot200'))
        lesson=[dict(task='place_empty_cup',steps=100)]
        child=validate(next_recipe(c,c['output_dir'],lesson,1))
        self.assertEqual(child['evaluation']['tasks'],['place_empty_cup'])

if __name__=='__main__':unittest.main()

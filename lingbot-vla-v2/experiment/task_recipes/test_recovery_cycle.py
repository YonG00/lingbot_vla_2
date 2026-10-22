import unittest
from experiment.task_recipes.recovery_cycle import historical_best,declines,choose_tasks,make_recipe
from experiment.task_recipes.test_contracts import fixture
from experiment.task_recipes.schema import validate,milestones
from experiment.task_recipes.planning import build_plan

class RecoveryTests(unittest.TestCase):
    def peaks(self):return {n:dict(successes=s,episodes=k,source='history')for n,s,k in [('a',4,4),('b',2,3),('c',2,4),('d',1,3)]}
    def result(self,t,s):return dict(task=t,state='complete',episodes=3,successes=s)
    def test_relative_drop_includes_one_and_two_successes(self):
        lessons=declines([self.result('a',2),self.result('b',1),self.result('c',1),self.result('d',1)],self.peaks())
        self.assertEqual([r['task']for r in lessons],['a','b','c']);self.assertTrue(all(r['steps']==80 for r in lessons))
        self.assertEqual(declines([self.result('a',3),self.result('b',3)],self.peaks()),[])
    def test_errors_and_duplicate_results_not_treated_as_regression(self):
        bad=self.result('a',0);bad['state']='failed'
        with self.assertRaises(ValueError):declines([bad],self.peaks())
        with self.assertRaises(ValueError):declines([self.result('a',0)]*2,self.peaks())
    def test_historical_best_not_latest_and_any_success_qualifies(self):
        p=dict(tasks=[dict(task='a',observations=[dict(successes=4,episodes=4),dict(successes=0,episodes=3)]),dict(task='b',observations=[dict(successes=1,episodes=4)])])
        peaks=historical_best(p);self.assertEqual(peaks['a']['successes'],4);self.assertIn('b',peaks)
    def test_three_rounds_fresh_and_exclude_treated(self):
        peaks={f'task{i}':dict(successes=1,episodes=3)for i in range(25)}
        seen=set(list(peaks)[:5]);treated=set(list(peaks)[:2]);all_new=[]
        for number in range(1,4):
            new=choose_tasks(peaks,treated,seen,7,number)
            self.assertEqual(len(new),5);self.assertFalse(set(new)&seen);self.assertFalse(set(new)&treated)
            seen.update(new);treated.add(new[0]);all_new+=new
        self.assertEqual(len(set(all_new)),15)
    def test_recipe_no_video_10percent_review_and_save_final(self):
        parent=fixture('place_container_plate_200');parent['train'].update(tasks_per_group=3,steps_per_group=3)
        lessons=[dict(task='a',steps=80),dict(task='b',steps=80)]
        c=validate(make_recipe(parent,'/tmp/source',1000,lessons,['c','d'],'/tmp/round',list(self.peaks()),.1,5))
        self.assertEqual(c['train']['steps'],160);self.assertEqual(milestones(c),[160]);self.assertEqual(c['evaluation']['video_episodes'],0)
        self.assertTrue(c['evaluation']['reuse_training_model']);self.assertEqual(c['train']['replay']['fraction'],.1)
        p=build_plan(c)
        self.assertEqual(sum(r['replay_samples']for r in p['steps']),896)
        self.assertTrue(all(r['replay_task']!=r['main_task']for r in p['steps']))
        self.assertNotIn('tasks_per_group',c['train'])
if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import unittest
from experiment.task_recipes.test_contracts import fixture
from experiment.task_recipes.schema import validate
from experiment.task_recipes.prepare import preview
from experiment.task_recipes.planning import build_plan,confirmed_progress
from experiment.task_recipes.replay_pool import eligible_tasks,resolve_pool,merge_results

ROOT=Path(__file__).resolve().parents[2]


class ReplayRotation(unittest.TestCase):
    def test_cumulative_results_and_idempotence(self):
        empty=dict(version=1,tasks=[])
        passed=dict(state='complete',task='a',successes=1,episodes=3)
        pool=merge_results(empty,passed,'run1/summary.json')
        self.assertEqual(merge_results(pool,passed,'run1/summary.json'),pool)
        failed=dict(state='complete',task='a',successes=0,episodes=3)
        self.assertEqual(eligible_tasks(merge_results(pool,failed,'run2/summary.json')),['a'])
        with self.assertRaises(ValueError):merge_results(pool,failed,'run1/summary.json')
        with self.assertRaises(ValueError):merge_results(pool,dict(state='failed'),'broken')

    def config(self,n=10):
        c=fixture('sequential_plate100_cup200_pot200')
        c['train'].update(main_tasks={'z':dict(steps=60,weight=1)},task_order=['z'])
        c['train']['replay']=dict(mode='round_robin',fraction=0.1,switch_every_steps=n,
            tasks={k:dict(weight=1) for k in ['a','b','c']})
        c['evaluation']['tasks']=['z']
        return validate(c)

    def test_one_task_per_update_and_equal_blocks(self):
        p=build_plan(self.config());rows=p['steps']
        self.assertEqual([r['replay_task'] for r in rows],['a']*10+['b']*10+['c']*10+['a']*10+['b']*10+['c']*10)
        self.assertEqual(set(r['replay_samples'] for r in rows),{5,6})
        self.assertTrue(all(sum(r['samples_by_task'].values())==56 for r in rows))
        progress=confirmed_progress(p,60)
        self.assertEqual(progress['replay_samples'],336)
        self.assertEqual([progress['samples_by_task'][k] for k in ['a','b','c']],[112]*3)

    def test_n1_and_main_exclusion_at_boundary(self):
        c=self.config(1)
        self.assertEqual([x['replay_task'] for x in build_plan(c)['steps'][:6]],['a','b','c']*2)
        c=self.config(10);c['train'].update(main_tasks={'a':dict(steps=5,weight=1),'b':dict(steps=55,weight=1)},task_order=['a','b'])
        rows=build_plan(validate(c))['steps']
        self.assertTrue(all(r['main_task']!=r['replay_task'] for r in rows))
        self.assertEqual(rows[0]['replay_task'],'b')
        self.assertNotEqual(rows[5]['replay_task'],'b')

    def test_invalid_or_ambiguous_config(self):
        for mutate in [lambda c:c['train']['replay'].update(switch_every_steps=0),
                       lambda c:c['train']['replay']['tasks']['a'].update(weight=2),
                       lambda c:c['train']['replay'].update(tasks={'z':dict(weight=1)})]:
            c=self.config();mutate(c)
            with self.assertRaises(ValueError):validate(c)

    def test_qualification_and_frozen_snapshot(self):
        raw=json.loads((ROOT/'configs/training_recipes/sequential_four_tasks_replay10pct_rr10_example.json').read_text())
        c=resolve_pool(raw,ROOT);names=list(c['train']['replay']['tasks'])
        self.assertEqual(len(names),14)
        self.assertIn('move_playingcard_away',names)
        self.assertIn('rotate_qrcode',names)
        self.assertNotIn('place_phone_stand',names)
        self.assertNotIn('place_object_scale',names)
        self.assertNotIn('place_a2b_right',names)
        self.assertIn('place_container_plate',names)
        self.assertEqual(resolve_pool(c,ROOT),c)
        c['train']['replay']['tasks'].pop(names[0])
        with self.assertRaises(ValueError):resolve_pool(c,ROOT)
        with self.assertRaises(ValueError):eligible_tasks(dict(version=1,tasks=[dict(task='x',observations=[dict(successes=4,episodes=3,source='bad')])]))

    def test_real_manifest_main_pool_overlap_and_preview(self):
        raw=json.loads((ROOT/'configs/training_recipes/sequential_four_tasks_replay10pct_rr10_example.json').read_text())
        c,p,g,_=preview(raw)
        self.assertEqual(len(g),16)
        self.assertEqual(confirmed_progress(p,800)['replay_samples'],4480)
        self.assertEqual(len(c['train']['replay']['pool_snapshot']['tasks']),17)
        # Main task can also belong to historical replay pool; data groups remain unique.
        raw['train']['main_tasks']={'adjust_bottle':dict(steps=200,weight=1)}
        raw['train']['task_order']=['adjust_bottle']
        _,p,g,_=preview(raw)
        self.assertEqual(len(g),14)
        self.assertTrue(all(r['replay_task']!='adjust_bottle' for r in p['steps']))

if __name__=='__main__':unittest.main()

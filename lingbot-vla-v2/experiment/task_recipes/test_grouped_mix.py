from collections import Counter
import unittest
from experiment.task_recipes.test_contracts import fixture
from experiment.task_recipes.schema import validate
from experiment.task_recipes.planning import build_plan,PlannedSampler,confirmed_progress,probe_steps

class GroupedTests(unittest.TestCase):
    def config(self):
        c=fixture('place_container_plate_200');t=c['train']
        t.pop('task_order',None)
        t.update(mode='grouped',steps=1000,main_tasks={f'task_{i}':dict(weight=1)for i in range(25)},tasks_per_group=3,steps_per_group=3)
        t['replay']=dict(fraction=0,tasks={});c['evaluation']['tasks']=list(t['main_tasks'])
        c['evaluation']['checkpoints']=['final'];c['checkpoint']['every_steps']=500
        return validate(c)
    def test_three_tasks_stay_three_steps_and_rotate_uneven_remainder(self):
        plan=build_plan(self.config());rows=plan['steps']
        for start in range(0,999,3):
            chunk=rows[start:start+3];group=set(chunk[0]['main_group'])
            self.assertEqual(len(group),3)
            self.assertTrue(all(set(r['samples_by_task'])==group and sum(r['samples_by_task'].values())==56 for r in chunk))
            sums=Counter()
            for r in chunk:sums.update(r['samples_by_task'])
            self.assertEqual(set(sums.values()),{56})
            if start:self.assertNotEqual(group,set(rows[start-1]['main_group']))
        self.assertEqual(len(rows),1000)
    def test_every_75_updates_balanced_and_tail_bounded(self):
        p=build_plan(self.config())
        for stop in range(75,1000,75):
            progress=confirmed_progress(p,stop)
            self.assertEqual(set(progress['samples_by_task'].values()),{stop*56//25})
        final=confirmed_progress(p,1000);values=final['samples_by_task'].values()
        self.assertEqual(sum(values),56000);self.assertLessEqual(max(values)-min(values),57)
        self.assertEqual(sum(final['group_participation_steps'].values()),3000)
    def test_sampler_effective_batch_matches_group_and_resume(self):
        p=build_plan(self.config());names=list(self.config()['train']['main_tasks']);pools={n:[i]for i,n in enumerate(names)}
        sampler=PlannedSampler(p,pools,3);iterator=iter(sampler)
        for row in p['steps'][:6]:
            batch=[names[next(iterator)]for _ in range(56)]
            self.assertEqual(Counter(batch),Counter(row['samples_by_task']))
        saved=sampler.state_dict();other=PlannedSampler(p,pools,3);other.load_state_dict(saved)
        self.assertEqual(list(iterator),list(other))
    def test_deterministic_and_invalid_settings(self):
        c=self.config();self.assertEqual(build_plan(c),build_plan(c))
        for changes in [dict(tasks_per_group=26),dict(steps_per_group=0),dict(tasks_per_group=True)]:
            bad=self.config();bad['train'].update(changes)
            with self.assertRaises(ValueError):validate(bad)
        bad=self.config();bad['train']['main_tasks']['task_0']['weight']=2
        with self.assertRaises(ValueError):validate(bad)
        bad=self.config();bad['train']['mode']='mixed'
        with self.assertRaises(ValueError):validate(bad)
        bad=self.config();bad['remediation']['enabled']=True
        with self.assertRaises(ValueError):validate(bad)
    def test_probe_spans_two_groups_and_legacy_stays_three(self):
        c=self.config();self.assertEqual(probe_steps(c),6)
        rows=build_plan(c)['steps'][:probe_steps(c)]
        self.assertEqual({r['group_index']for r in rows},{0,1})
        self.assertEqual(probe_steps(validate(fixture('place_container_plate_200'))),3)
if __name__=='__main__':unittest.main()

"""无需torch的配方/采样/恢复/虚拟epoch契约。"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from experiment.task_recipes.schema import validate,milestones
from experiment.task_recipes.planning import build_plan,confirmed_progress,PlannedSampler,index_pools
from experiment.task_recipes.runtime import StreamDataset,load_run
from experiment.task_recipes.prepare import preview

ROOT=Path(__file__).resolve().parents[2]

def fixture(name='three_main_with_replay_400'):
    return json.loads((ROOT/f'configs/training_recipes/{name}.json').read_text())

class RecipeContracts(unittest.TestCase):
    def test_sequential_boundaries_and_checkpoint_steps(self):
        c=validate(fixture('sequential_plate100_cup200_pot200'));p=build_plan(c)
        self.assertEqual([r['main_task'] for r in p['steps']],['place_container_plate']*100+['place_empty_cup']*200+['move_can_pot']*200)
        self.assertEqual(milestones(c),[250,500])
        self.assertEqual(p['steps'][249]['main_task'],'place_empty_cup')
        self.assertTrue(all(r['replay_samples']==0 and len(r['samples_by_task'])==1 for r in p['steps']))
        raw=fixture('sequential_plate100_cup200_pot200')
        raw['train']['main_tasks']['place_container_plate']['weight']=2
        with self.assertRaises(ValueError):validate(raw)

    def test_main_budget_and_replay_ratio_are_separate(self):
        c=validate(fixture());p=build_plan(c);x=confirmed_progress(p,400)
        self.assertEqual(x['main_steps'],dict(place_container_plate=150,place_empty_cup=150,move_can_pot=100))
        self.assertEqual(x['replay_samples'],4480)
        self.assertEqual(x['samples_by_task']['adjust_bottle'],3360)
        self.assertEqual(x['samples_by_task']['click_bell'],1120)
        self.assertTrue(all(sum(r['samples_by_task'].values())==56 for r in p['steps']))
        self.assertEqual(milestones(c),[200,400])
    def test_mixed_mode_equal_task_samples_not_equal_frame_length(self):
        raw=fixture('place_container_plate_200');raw['train'].update(steps=100,main_tasks={'a':dict(weight=1),'b':dict(weight=1),'c':dict(weight=1)})
        p=build_plan(validate(raw));values=list(confirmed_progress(p,100)['samples_by_task'].values())
        self.assertLessEqual(max(values)-min(values),1)
    def test_weights_control_mixed_samples(self):
        raw=fixture('place_container_plate_200');raw['train'].update(steps=100,main_tasks={'a':dict(weight=3),'b':dict(weight=1)})
        result=confirmed_progress(build_plan(validate(raw)),100)
        self.assertEqual(result['samples_by_task'],dict(a=4200,b=1400))
    def test_invalid_ambiguous_or_unsafe_recipe_rejected(self):
        mutations=[lambda c:c.update(typo=1),lambda c:c['train'].update(steps=99),
            lambda c:c['train']['replay'].update(fraction=1),
            lambda c:c['train']['replay']['tasks'].update(place_container_plate=dict(weight=1)),
            lambda c:c['checkpoint'].update(save_final=False),lambda c:c['evaluation'].update(checkpoints=[350]),
            lambda c:c['runtime'].update(shutdown_after_completion=True),lambda c:c['train'].update(seed=True),
            lambda c:c['train']['main_tasks']['place_container_plate'].update(weight=float('nan'))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                c=fixture();mutate(c)
                with self.assertRaises(ValueError):validate(c)
    def test_mixed_rejects_per_task_optimizer_step_claims(self):
        c=fixture();c['train']['mode']='mixed';c['train']['steps']=400
        with self.assertRaises(ValueError):validate(c)
    def test_replay_cannot_leave_an_empty_main_batch(self):
        c=fixture();c['train']['replay']['fraction']=.999
        with self.assertRaises(ValueError):validate(c)
    def test_sample_cursor_restores_exact_sequence_and_checks_signature(self):
        c=validate(fixture('place_container_plate_200'));c['train']['steps']=4;p=build_plan(c)
        pools=dict(place_container_plate=[0,2,10,100])
        sampler=PlannedSampler(p,pools,3);it=iter(sampler);first=[next(it) for _ in range(61)];state=sampler.state_dict();rest=list(it)
        restored=PlannedSampler(p,pools,3);restored.load_state_dict(state)
        self.assertEqual(list(restored),rest)
        self.assertEqual(first+rest,list(PlannedSampler(p,pools,3)))
        with self.assertRaises(ValueError):PlannedSampler(p,pools,4).load_state_dict(state)
        with self.assertRaises(ValueError):restored.set_epoch(1)
    def test_raw_index_space_and_virtual_epoch_are_independent(self):
        class Real:
            def __len__(self):return 1000
            def __getitem__(self,i):return i
        virtual=StreamDataset(Real(),56)
        self.assertEqual(len(virtual),56)
        self.assertEqual(virtual[999],999)
        pools=index_pools({'a':[50],'b':[60]},[60,50,50])
        self.assertEqual(pools,dict(a=[1,2],b=[0]))
        with self.assertRaises(ValueError):index_pools({'a':[50]},[50,51])
    def test_prefetch_cursor_does_not_inflate_confirmed_progress(self):
        c=validate(fixture());p=build_plan(c)
        progress=confirmed_progress(p,3)
        self.assertEqual(sum(progress['samples_by_task'].values()),168)
        self.assertEqual(sum(progress['main_steps'].values()),3)
    def test_preview_resolves_groups_and_freezes_exact_budget(self):
        c,p,g,frames=preview(fixture())
        self.assertEqual(len(g),5)
        self.assertEqual(sum(map(len,g.values())),250)
        self.assertEqual(frames,24119+7156+3855)
        self.assertEqual(len(p['steps']),400)
    def test_plan_tampering_rejected(self):
        c=validate(fixture());p=build_plan(c);p['steps'][0]['main_task']='wrong'
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            (out/'recipe.json').write_text(json.dumps(c));(out/'step_plan.json').write_text(json.dumps(p));(out/'task_groups.json').write_text('{}')
            with self.assertRaises(ValueError):load_run(out)
    def test_default_cli_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'absent'
            result=subprocess.run([sys.executable,str(ROOT/'experiment/task_recipes/run.py'),'--recipe',str(ROOT/'configs/training_recipes/place_container_plate_200.json'),'--out',str(out)],capture_output=True,text=True,check=True)
            self.assertFalse(json.loads(result.stdout)['training_started'])
            self.assertFalse(out.exists())
            self.assertFalse(out.with_name(out.name+'_pipeline').exists())

if __name__=='__main__':unittest.main()

"""Nonuniform task boundaries, legacy interval regression, and remediation combinations."""
import unittest
from experiment.task_recipes.test_contracts import fixture
from experiment.task_recipes.schema import validate,milestones,save_interval
from experiment.task_recipes.remediation import next_recipe
from lingbotvla.utils.checkpoint_schedule import checkpoint_due


class CheckpointModes(unittest.TestCase):
    def test_every_three_tasks_and_final_partial_group(self):
        c=fixture('sequential_plate100_cup200_pot200')
        names=['a','b','c','d','e','f','g']
        c['train'].update(main_tasks={name:dict(steps=100 if i<2 else 200,weight=1) for i,name in enumerate(names)},task_order=names)
        c['checkpoint']=dict(mode='task',every_tasks=3,save_final=True,hf_dtype='bf16')
        c=validate(c)
        self.assertEqual(milestones(c),[400,1000,1200])
        self.assertEqual([s for s in range(1,1201) if checkpoint_due(s,0,milestones(c))],[400,1000,1200])
        c['checkpoint']['every_tasks']=0
        with self.assertRaises(ValueError):validate(c)

    def test_legacy_and_smoke(self):
        for interval in [0,200,250]:
            for step in range(1,701):
                self.assertEqual(checkpoint_due(step,interval),bool(interval and step%interval==0))
                self.assertFalse(checkpoint_due(step,interval,[100,300],True))

    def test_nonuniform_task_boundaries(self):
        c=fixture('sequential_plate100_cup200_pot200')
        c['checkpoint']=dict(mode='task',save_final=True,hf_dtype='bf16')
        c=validate(c);points=milestones(c)
        self.assertEqual(points,[100,300,500]);self.assertEqual(save_interval(c),0)
        self.assertEqual([s for s in range(1,501) if checkpoint_due(s,0,points)],points)
        c['checkpoint']['every_steps']=250
        with self.assertRaises(ValueError):validate(c)

    def test_task_mode_rejects_mixed(self):
        c=fixture('place_container_plate_200')
        c['checkpoint']=dict(mode='task',save_final=True,hf_dtype='bf16')
        with self.assertRaises(ValueError):validate(c)

    def test_save_modes_times_remediation(self):
        for mode in ['steps','task']:
            for enabled in [False,True]:
                c=fixture('sequential_plate100_cup200_pot200')
                c['checkpoint']=dict(mode=mode,save_final=True,hf_dtype='bf16')
                if mode=='steps':c['checkpoint']['every_steps']=250
                c['remediation']=dict(enabled=enabled)
                c=validate(c)
                self.assertEqual(milestones(c),[250,500] if mode=='steps' else [100,300,500])
                lessons=[dict(task='place_empty_cup',steps=100),dict(task='move_can_pot',steps=200)]
                child=validate(next_recipe(c,c['output_dir'],lessons,1))
                self.assertEqual(milestones(child),[300] if mode=='steps' else [100,300])

if __name__=='__main__':unittest.main()

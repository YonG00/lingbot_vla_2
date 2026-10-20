import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from experiment.official_control.three_task import uniform_weights,TRAIN_TASKS,EVAL_TASKS

class ThreeTaskTests(unittest.TestCase):
    def test_task_mass_is_equal_despite_unequal_frames(self):
        weights,summary=uniform_weights({'a':[0],'b':[5],'c':[9]},[0]+[5]*3+[9]*6)
        self.assertAlmostEqual(weights[0],1)
        for value in summary['task_probability'].values():self.assertAlmostEqual(value,1/3)
        self.assertEqual(summary['frames'],{'a':1,'b':3,'c':6})
    def test_missing_overlapping_and_extra_episodes_rejected(self):
        for groups,eps in [({'a':[0],'b':[0]},[0]),({'a':[0,1]},[0]),({'a':[0]},[0,1])]:
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):uniform_weights(groups,eps)
    def test_task_scope_and_manifest_are_exact(self):
        self.assertEqual(len(EVAL_TASKS),len(set(EVAL_TASKS)))
        self.assertEqual(len(EVAL_TASKS),8)
        self.assertNotIn('adjust_bottle',TRAIN_TASKS)
        root=Path(__file__).resolve().parents[2]
        manifest=json.loads((root/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())['tasks']
        self.assertEqual(sum(manifest[t]['n_total'] for t in TRAIN_TASKS),150)
        self.assertEqual(sum(manifest[t]['train_frames']+manifest[t]['val_frames'] for t in TRAIN_TASKS),24119)
    def test_plan_only_does_not_start_or_write(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'new'
            script=Path(__file__).with_name('three_task_pipeline.py')
            p=subprocess.run([sys.executable,str(script),'--out',str(out)],capture_output=True,text=True,check=True)
            plan=json.loads(p.stdout)
            self.assertEqual((plan['steps'],plan['hf_steps'],plan['evaluate_step']),(400,[200,400],400))
            self.assertFalse(out.exists())
            self.assertFalse(out.with_name(out.name+'_pipeline').exists())

if __name__=='__main__':unittest.main()

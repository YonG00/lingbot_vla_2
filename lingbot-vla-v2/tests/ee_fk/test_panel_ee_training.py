"""Pure CPU panel contracts; no torch/TensorBoard import or GPU collection."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('panel_ee_training', ROOT/'tools/auto_learning_pipeline/al_panel.py')
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


class PanelEETrainingTests(unittest.TestCase):
    def points(self):
        return {f'ee_fk/{name}_loss': [NS(step=100, value=value), NS(step=120, value=9.)]
                for name, value in [('position', .001), ('orientation', .03), ('gripper', .002)]}

    def test_exact_step_missing_partial_zero_and_nonfinite(self):
        source = {key: [(point.step, point.value) for point in points]
                  for key, points in self.points().items()}
        series = {'_training_ee': source}
        text = panel.ee_training_loss_text(series, 100)
        self.assertIn('位置 0.00100 姿态 0.03000 夹爪 0.00200', text)
        self.assertIn('未加权/该步', text)
        self.assertIsNone(panel.ee_training_loss_text(series, 99))
        self.assertIsNone(panel.ee_training_loss_text({}, 100))
        source['ee_fk/gripper_loss'] = [(100, float('nan'))]
        self.assertIsNone(panel.ee_training_loss_text(series, 100))
        source['ee_fk/gripper_loss'] = [(100, 0.)]
        self.assertIn('夹爪 0.00000', panel.ee_training_loss_text(series, 100))

    def test_real_reader_and_both_views_bind_ee_to_history_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'al_panel_ee';run.mkdir()
            al = run/'al_config.json'
            al.write_text(json.dumps(dict(max_global_steps=500,eval_interval_steps=100,
                pass_transition_enabled=False,task_scan_enabled=False,hardness_scan_enabled=True)))
            (run/'lingbotvla_cli.yaml').write_text(f'train:\n  auto_learning: {al}\n  max_steps: 500\n')
            rows = [dict(action='select', task='stand', step=0)]
            for step in (0, 100):
                rows += [dict(name=f'task/stand/{name}',step=step,value=value) for name,value in
                         [('active_val_gmean_mse',.006),('pass_threshold_gmean_mse',.001),
                          ('val_to_pass_threshold',6.)]]
            rows.append(dict(name='task/stand/unit_loss',step=100,value=.06))
            (run/'auto_learning_events.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
            (run/'runs').mkdir();(run/'runs/events.out.tfevents.test').touch()
            points = self.points()
            points['training/loss'] = [NS(step=120,value=.05)]
            accumulator = NS(Reload=lambda:None,Tags=lambda:{'scalars':list(points)},Scalars=points.get)
            factory = Mock(return_value=accumulator)
            module = ModuleType('tensorboard.backend.event_processing.event_accumulator')
            module.EventAccumulator = factory
            with patch.dict(sys.modules,{module.__name__:module}), \
                 patch.object(panel,'ROOT',root), patch.object(panel,'discover_trainers',return_value=[]):
                for flags in ([], ['--wide']):
                    with self.subTest(flags=flags), patch('sys.argv',['al_panel.py','al_panel_ee',*flags]), \
                         contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(panel.main(),0)
                    text = output.getvalue()
                    self.assertIn('EE训练@100',text)
                    self.assertIn('夹爪 0.00200',text)
                    self.assertNotIn('EE训练@0',text)
                    self.assertNotIn('EE训练@120',text)
            self.assertEqual(factory.call_count,2)  # one TB load per refresh, no second read


if __name__ == '__main__':
    unittest.main()

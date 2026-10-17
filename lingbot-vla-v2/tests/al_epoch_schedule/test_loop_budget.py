"""Execute the real trainer's range policy with CPU hook/sampler contracts."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def loop_range(*, al=True, maximum=500, current=0, start=0, epoch_steps=231, save_epochs=0):
    tree = ast.parse((ROOT/'tasks/vla/train_lingbotvla.py').read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target,ast.Name) and target.id=='epoch_train_steps' for target in node.targets))
    guard = next(node for node in ast.walk(tree) if isinstance(node,ast.If)
                 and any(isinstance(child,ast.Assign) and any(isinstance(target,ast.Name)
                     and target.id=='epoch_train_steps' for target in child.targets) for child in node.body))
    loop = next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                and isinstance(node.target,ast.Name) and node.target.id=='epoch_step')
    namespace=dict(args=NS(train=NS(train_steps=epoch_steps,max_steps=maximum,save_epochs=save_epochs)),
                   _al_hook=object() if al else None,start_step=start,global_step=current)
    exec(compile(ast.Module(body=[assignment,guard],type_ignores=[]),'actual_range','exec'),namespace)
    return list(eval(compile(ast.Expression(loop.iter),'actual_range','eval'),namespace))


class ALOptimizerStepLoopTests(unittest.TestCase):
    def test_range_ignores_dataset_epoch_only_for_fixed_step_al_without_epoch_saves(self):
        self.assertEqual(len(loop_range()),500)
        self.assertEqual(len(loop_range(al=False)),231)
        self.assertEqual(len(loop_range(maximum=None)),231)
        self.assertEqual(len(loop_range(save_epochs=1)),231)
        self.assertEqual(loop_range(current=500),[])
        self.assertEqual(len(loop_range(current=250,start=19)),250)

    def test_real_hook_sampler_cross_231_without_partial_eval_and_keep_final_flush(self):
        # Force the existing no-torch sampler fallback. No model, CUDA or pytest.
        block_torch=patch.dict(sys.modules,{'torch':None})
        block_torch.start()
        self.addCleanup(block_torch.stop)  # include delayed imports during scheduler evaluation
        from lingbotvla.auto_learning.config import AutoLearningConfig
        from lingbotvla.auto_learning.real.hook import AutoLearnLoopHook
        from lingbotvla.auto_learning.real.sampler import AutoLearnSampler
        from lingbotvla.auto_learning.sampling.sampler import BatchSampler
        from lingbotvla.auto_learning.orchestration.scheduler import Scheduler
        from lingbotvla.auto_learning.testing import fake_tasks as ft
        from lingbotvla.auto_learning.testing.sim import build_backend
        al = AutoLearningConfig(enabled=True,batch_size=2,new_slots=2,replay_slots=0,
            eval_interval_steps=100,min_steps_before_defer=10000,defer_retry_steps=100,
            pass_transition_enabled=False,task_scan_enabled=False,hardness_scan_enabled=False,
            review_after_task_transitions=0,max_global_steps=500)
        cfg=ft.make_cfg([ft._spec('stand',curve=[.9,.8])],al=al)
        backend=build_backend(cfg)
        scheduler=Scheduler(backend,al,seed=3)
        sampler=AutoLearnSampler(BatchSampler(al,backend.resolver,backend.catalog,scheduler.rng),batch_size=2)
        hook=AutoLearnLoopHook(scheduler=scheduler,sampler=sampler,cfg=al)
        iterator=None;boundaries=[]
        # Stop early at 255 to also test an intentional partial final unit.
        for index in loop_range(maximum=255):
            step=index+1
            directive=hook.on_step_begin(step-1)
            self.assertFalse(directive.finished)
            if directive.rebuild_iterator:iterator=iter(sampler)
            [next(iterator) for _ in range(2)]
            hook.on_step_end(step,.1)
            if hook.at_safe_checkpoint_boundary:boundaries.append(step)
            if step==231:
                self.assertEqual(hook.step_in_unit,31)
                self.assertEqual(scheduler.state.global_step,200)
                self.assertFalse(hook.at_safe_checkpoint_boundary)
        self.assertEqual(boundaries,[100,200])
        hook.flush_partial_unit(255)
        self.assertEqual(scheduler.state.global_step,255)
        self.assertEqual(scheduler.state.global_samples_seen,510)
        self.assertTrue(hook.at_safe_checkpoint_boundary)
        self.assertIsNone(sys.modules.get('torch'))


if __name__=='__main__':unittest.main()

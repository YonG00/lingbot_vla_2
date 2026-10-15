"""执行训练器实际 HF-only 保存/结束分支，替身阻断所有 torch/GPU 依赖。"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import textwrap
import unittest
import re

from test_al_periodic_checkpoint import run_periodic, TRAINER


def save_sequence(steps, *, fail=False, deny=False):
    source = TRAINER.read_text()
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    helper = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "_save_hf_only")
    block = textwrap.dedent(ast.get_source_segment(source, helper))
    code = ("def exercise():\n    _last_hf_only_step = None\n    max_checkpoint_used_gb = 0\n"
            + textwrap.indent(block, "    ")
            + "\n    for step in steps: _save_hf_only(step)\n    return _last_hf_only_step\n")
    calls = []
    def export(path, step):
        calls.append(("hf", step))
        return None if fail else path
    ns = dict(steps=steps, args=NS(train=NS(disk_guard=True, save_checkpoint_path="/fake/checkpoints",
              disk_guard_margin=1.1)), hf_saver=None, os=__import__("os"),
        _disk_guard_check=lambda *a, **k: (not deny, 12, 90, "disk"),
        _save_hf_direct_best_effort=export,
        _export_pool_state_here=lambda p, s: calls.append(("pool", s)),
        dist=NS(broadcast_object_list=lambda *a, **kw: None))
    exec(compile(code, str(TRAINER), "exec"), ns)
    return calls, ns["exercise"]


def ending_action(*, reached=False, al=False, last_epoch=False, save_epoch=0):
    source = TRAINER.read_text()
    tree = ast.parse(source)
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare) and n.test.comparators
        and isinstance(n.test.comparators[0], ast.Constant)
        and n.test.comparators[0].value == "hf_only"
        and any(isinstance(x, ast.Continue) for x in n.body))
    code = ("def exercise():\n    for epoch in [0]:\n"
            + textwrap.indent(textwrap.dedent(ast.get_source_segment(source, branch)), "        ")
            + "\n        raise AssertionError('fell through to DCP branch')\n")
    calls = []
    ns = dict(args=NS(train=NS(dcp_save_mode="hf_only",
        num_train_epochs=1 if last_epoch else 2, save_epochs=save_epoch)),
        reached_max_steps=reached, al_finished=al, global_step=1000,
        _save_hf_only=lambda s: calls.append(s))
    exec(compile(code, str(TRAINER), "exec"), ns)
    ns["exercise"]()
    return calls


class HFOnlyTests(unittest.TestCase):
    def test_step_reset_and_legacy_auto_offset(self):
        source = TRAINER.read_text()
        start = source.index("    _step_offset = max(0, int(getattr(args.train,")
        end = source.index("    total_train_steps =", start)
        block = textwrap.dedent(source[start:end])
        for reset, offset, expected in ((False, 0, 9680), (True, 0, 0), (False, 100, 100)):
            with self.subTest(reset=reset, offset=offset):
                ns = dict(args=NS(train=NS(step_offset=offset, reset_step_counter=reset,
                    enable_resume=False, load_checkpoint_path=None),
                    model=NS(model_path="/source/global_step_9680/hf_ckpt")),
                    re=re, logger=NS(info_rank0=lambda *a: None))
                exec(compile(block, str(TRAINER), "exec"), ns)
                self.assertEqual(ns["_step_offset"], expected)

    def test_invalid_hf_only_combinations_fail(self):
        source = TRAINER.read_text()
        main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        guard = next(n for n in main.body if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare) and isinstance(n.test.comparators[0], ast.Constant)
            and n.test.comparators[0].value == "hf_only")
        block = textwrap.dedent(ast.get_source_segment(source, guard))
        for key in ("save_hf_weights", "async_save_hf_weights", "enable_resume",
                    "load_checkpoint_path", "smoke_no_checkpoint", "hf_pass_interval"):
            train = dict(dcp_save_mode="hf_only", save_hf_weights=True,
                async_save_hf_weights=False, enable_resume=False, load_checkpoint_path=None,
                smoke_no_checkpoint=False, hf_pass_interval=0)
            train[key] = False if key == "save_hf_weights" else True
            with self.subTest(key=key), self.assertRaises(ValueError):
                exec(compile(block, str(TRAINER), "exec"),
                     dict(args=NS(train=NS(**train)), logger=NS(info_rank0=lambda *a: None)))

    def test_periodic_on_and_between_unit_boundaries(self):
        calls, (_, last_dcp, stopped) = run_periodic(
            [(200, False), (250, True), (400, True)], mode="hf_only", interval=200)
        self.assertEqual(calls, [("disk", 200), ("hf_only", 200),
                                 ("disk", 400), ("hf_only", 400)])
        self.assertIsNone(last_dcp)
        self.assertFalse(stopped)

    def test_five_periodic_exports_and_same_step_final_do_not_duplicate(self):
        calls, run = save_sequence([200, 400, 600, 800, 1000, 1000])
        self.assertEqual(run(), 1000)
        self.assertEqual([s for kind, s in calls if kind == "hf"], [200, 400, 600, 800, 1000])
        self.assertEqual([s for kind, s in calls if kind == "pool"], [200, 400, 600, 800, 1000])

    def test_off_grid_final_is_saved(self):
        calls, run = save_sequence([200, 300])
        run()
        self.assertEqual(calls, [("hf", 200), ("pool", 200), ("hf", 300), ("pool", 300)])

    def test_final_stop_al_and_epoch_never_reach_dcp(self):
        for kw in (dict(reached=True), dict(al=True), dict(last_epoch=True), dict(save_epoch=1)):
            with self.subTest(kw=kw):
                self.assertEqual(ending_action(**kw), [1000])
        self.assertEqual(ending_action(), [])

    def test_failure_or_disk_denial_cannot_claim_success(self):
        for kw in (dict(fail=True), dict(deny=True)):
            with self.subTest(kw=kw):
                calls, run = save_sequence([200], **kw)
                with self.assertRaises(RuntimeError):
                    run()
                self.assertFalse(any(kind == "pool" for kind, _ in calls))


if __name__ == "__main__":
    unittest.main()

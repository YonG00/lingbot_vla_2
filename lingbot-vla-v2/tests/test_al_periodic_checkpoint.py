"""执行训练器的真实周期存档分支；全部依赖替身，无 torch/GPU 导入。"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest


TRAINER = Path(__file__).resolve().parents[1] / "tasks/vla/train_lingbotvla.py"


def run_periodic(steps, *, mode="final_only", hf=True, smoke=False,
                 interval=220, denied=()):
    # 只提取生产分支，不复制调度规则；保留 break 的真实磁盘保护行为。
    source = TRAINER.read_text(encoding="utf-8")
    start = source.index("            # ---- [Stage B1] 存档触发")
    end = source.index("            # --- 训练中原地 open-loop validation", start)
    block = textwrap.dedent(source[start:end])
    code = ("def exercise():\n"
            "    _al_deferred_save = False\n"
            "    _last_saved_step = None\n"
            "    max_checkpoint_used_gb = 0\n"
            "    disk_guard_stop = False\n"
            "    for global_step, boundary in steps:\n"
            "        _al_hook.at_safe_checkpoint_boundary = boundary\n"
            + textwrap.indent(block, "        ")
            + "    return _al_deferred_save, _last_saved_step, disk_guard_stop\n")
    calls = []
    train = SimpleNamespace(
        save_hf_weights=hf, dcp_save_mode=mode, smoke_no_checkpoint=smoke,
        save_steps=interval, hf_pass_interval=0, disk_guard=True,
        disk_guard_margin=1.1, global_rank=0, save_checkpoint_path="/fake/checkpoints",
    )
    noop = lambda *a, **kw: None
    stateful = SimpleNamespace(state_dict=lambda: {})

    def disk_check(*a, **kw):
        step = int(kw["label"].rsplit(" ", 1)[1])
        calls.append(("disk", step))
        return step not in denied, 24, 80, "test disk guard"

    ns = dict(
        steps=steps, args=SimpleNamespace(train=train),
        os=SimpleNamespace(path=os.path, makedirs=noop),
        logger=SimpleNamespace(info_rank0=noop),
        _al_hook=SimpleNamespace(at_safe_checkpoint_boundary=False,
                                 step_in_unit=20, unit_steps=50),
        _disk_guard_check=disk_check, hf_saver=None,
        helper=SimpleNamespace(empty_cache=noop), model=object(), optimizer=object(),
        lr_scheduler=stateful, train_dataloader=stateful, environ_meter=stateful,
        torch=SimpleNamespace(get_rng_state=lambda: "unchanged RNG"),
        _auto_learning_extra_state=lambda: {}, writer=SimpleNamespace(flush=noop),
        dist=SimpleNamespace(barrier=noop), Checkpointer=object(),
        _retention_after_save=noop,
        _dcp_save_or_abort=lambda _c, _p, _s, step: calls.append(("dcp", step)),
        _save_hf_direct_best_effort=lambda _p, step: calls.append(("hf", step)),
        _export_pool_state_here=lambda _p, step: calls.append(("pool", step)),
    )
    exec(compile(ast.parse(code), str(TRAINER), "exec"), ns)
    return calls, ns["exercise"]()


class PeriodicCheckpointTests(unittest.TestCase):
    def test_hf_at_unit_boundary_is_written_once(self):
        calls, (_, last, stopped) = run_periodic([(4400, True)])
        self.assertEqual(calls, [("disk", 4400), ("hf", 4400), ("pool", 4400)])
        self.assertIsNone(last)
        self.assertFalse(stopped)

    def test_hf_mid_unit_does_not_queue_another_save(self):
        calls, (deferred, last, _) = run_periodic([(4620, False), (4650, True)])
        self.assertEqual(calls, [("disk", 4620), ("hf", 4620), ("pool", 4620)])
        self.assertFalse(deferred)
        self.assertIsNone(last)

    def test_hf_absolute_step_sequence(self):
        steps = [(step, step % 50 == 0) for step in range(4301, 4841)]
        calls, _ = run_periodic(steps)
        self.assertEqual([step for kind, step in calls if kind == "hf"],
                         [4400, 4620, 4840])
        self.assertEqual([step for kind, step in calls if kind == "pool"],
                         [4400, 4620, 4840])

    def test_dcp_still_defers_until_safe_boundary_with_or_without_hf(self):
        for hf in (False, True):
            with self.subTest(hf=hf):
                calls, (deferred, last, _) = run_periodic(
                    [(4620, False), (4650, True), (4700, True)], mode="always", hf=hf)
                self.assertEqual(calls, [("disk", 4650), ("dcp", 4650)])
                self.assertFalse(deferred)
                self.assertEqual(last, 4650)

    def test_final_only_without_hf_keeps_legacy_no_export(self):
        calls, (deferred, last, _) = run_periodic(
            [(4620, False), (4650, True)], hf=False)
        self.assertEqual(calls, [("disk", 4650)])
        self.assertFalse(deferred)
        self.assertIsNone(last)

    def test_periodic_hf_cannot_replace_final_dcp(self):
        # 收尾分支会 flush_partial_unit 后使用这个游标；HF 不能让它跳过 DCP。
        for boundary in (False, True):
            with self.subTest(boundary=boundary):
                calls, (_, last, _) = run_periodic([(4400, boundary)])
                self.assertEqual([kind for kind, _ in calls], ["disk", "hf", "pool"])
                self.assertNotEqual(last, 4400)

    def test_disk_guard_runs_before_hf_and_blocks_export(self):
        calls, (_, last, stopped) = run_periodic([(4400, True)], denied=(4400,))
        self.assertEqual(calls, [("disk", 4400)])
        self.assertIsNone(last)
        self.assertTrue(stopped)

    def test_smoke_disables_all_four_mode_combinations(self):
        for mode in ("always", "final_only"):
            for hf in (False, True):
                with self.subTest(mode=mode, hf=hf):
                    calls, (_, last, _) = run_periodic(
                        [(4400, True), (4620, False), (4650, True)],
                        mode=mode, hf=hf, smoke=True)
                    self.assertEqual(calls, [])
                    self.assertIsNone(last)

    def test_zero_interval_disables_periodic_save(self):
        calls, _ = run_periodic([(4400, True)], interval=0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

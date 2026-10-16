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
                 interval=220, denied=(), on_save_calls=None, hf_fail=False,
                 max_steps=2000, stopping=False, al_finished=False,
                 final_epoch=False, al_enabled=True, stop_file=False,
                 rank=0, rank0_hf_success=None):
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
            "        if _al_hook is not None: _al_hook.at_safe_checkpoint_boundary = boundary\n"
            + textwrap.indent(block, "        ")
            + "    return _al_deferred_save, _last_saved_step, disk_guard_stop\n")
    calls = []
    train = SimpleNamespace(
        save_hf_weights=hf, dcp_save_mode=mode, smoke_no_checkpoint=smoke,
        save_steps=interval, hf_pass_interval=0, disk_guard=True,
        disk_guard_margin=1.1, global_rank=rank, save_checkpoint_path="/fake/checkpoints",
        max_steps=max_steps, num_train_epochs=1 if final_epoch else 2, train_steps=1,
    )
    noop = lambda *a, **kw: None
    stateful = SimpleNamespace(state_dict=lambda: {})
    #: 默认空操作；传列表时记录每次 `on_save(step)` 的 step（验证"存档后重扫"触发点）。
    _on_save = (noop if on_save_calls is None
                else (lambda step, *, will_continue=True:
                      on_save_calls.append(step) if will_continue else None))

    def export_hf(path, step):
        calls.append(("hf", step))
        return None if hf_fail else path

    def broadcast(result, **kwargs):
        if rank0_hf_success is not None:
            result[0] = rank0_hf_success

    def disk_check(*a, **kw):
        step = int(kw["label"].rsplit(" ", 1)[1])
        calls.append(("disk", step))
        return step not in denied, 24, 80, "test disk guard"

    ns = dict(
        steps=steps, args=SimpleNamespace(train=train),
        os=SimpleNamespace(path=SimpleNamespace(join=os.path.join,
            exists=lambda p: stop_file), makedirs=noop),
        logger=SimpleNamespace(info_rank0=noop),
        _al_hook=SimpleNamespace(at_safe_checkpoint_boundary=False,
                                 step_in_unit=20, unit_steps=50,
                                 scheduler=SimpleNamespace(state=SimpleNamespace(finished=al_finished)),
                                 on_save=_on_save),
        _disk_guard_check=disk_check, hf_saver=None,
        helper=SimpleNamespace(empty_cache=noop), model=object(), optimizer=object(),
        lr_scheduler=stateful, train_dataloader=stateful, environ_meter=stateful,
        torch=SimpleNamespace(get_rng_state=lambda: "unchanged RNG"),
        _auto_learning_extra_state=lambda: {}, writer=SimpleNamespace(flush=noop),
        dist=SimpleNamespace(barrier=noop, broadcast_object_list=broadcast), Checkpointer=object(),
        _retention_after_save=noop,
        _dcp_save_or_abort=lambda _c, _p, _s, step: calls.append(("dcp", step)),
        _save_hf_direct_best_effort=export_hf,
        _export_pool_state_here=lambda _p, step: calls.append(("pool", step)),
        _save_hf_only=lambda step, **kw: calls.append(("hf_only", step)),
        _al_stop_pending=stopping, stop_and_save_path="/fake/STOP_AND_SAVE" if stop_file else None,
        epoch=0, epoch_step=0,
    )
    if not al_enabled:
        ns["_al_hook"] = None
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

    def test_on_save_hook_called_only_at_periodic_save_steps(self):
        # 存档后重扫入口：只在 save_steps 整数倍触发，非存档步不触发（2026-10-16）。
        recorded = []
        run_periodic([(250, False), (300, False), (500, False)],
                     interval=250, on_save_calls=recorded)
        self.assertEqual(recorded, [250, 500])
        # AL 关闭（无 hook）或全程不存档（interval=0）时一次都不触发。
        recorded_disabled = []
        run_periodic([(250, False)], interval=0, on_save_calls=recorded_disabled)
        self.assertEqual(recorded_disabled, [])

    def test_scan_requires_a_successful_artifact(self):
        for options in (dict(hf=False), dict(hf_fail=True), dict(denied=(250,)),
                        dict(smoke=True), dict(al_enabled=False)):
            with self.subTest(options=options):
                scans = []
                run_periodic([(250, True)], interval=250, on_save_calls=scans, **options)
                self.assertEqual(scans, [])

    def test_scan_skips_final_step_stop_and_finished_scheduler(self):
        for options in (dict(max_steps=250), dict(stopping=True), dict(stop_file=True),
                        dict(al_finished=True), dict(final_epoch=True)):
            with self.subTest(options=options):
                scans = []
                calls, _ = run_periodic([(250, True)], mode="hf_only", interval=250,
                                        on_save_calls=scans, **options)
                self.assertIn(("hf_only", 250), calls)
                self.assertEqual(scans, [])

    def test_successful_hf_only_and_dcp_each_trigger_one_scan(self):
        for mode in ("hf_only", "always"):
            with self.subTest(mode=mode):
                scans = []
                run_periodic([(250, True)], mode=mode, interval=250, on_save_calls=scans)
                self.assertEqual(scans, [250])

    def test_nonzero_rank_uses_rank0_export_success_for_scan_decision(self):
        for success in (True, False):
            with self.subTest(success=success):
                scans = []
                # 非零 rank 不返回导出路径，不能据此误判整次保存失败。
                run_periodic([(250, False)], interval=250, on_save_calls=scans,
                             rank=1, hf_fail=True, rank0_hf_success=success)
                self.assertEqual(scans, [250] if success else [])


if __name__ == "__main__":
    unittest.main()

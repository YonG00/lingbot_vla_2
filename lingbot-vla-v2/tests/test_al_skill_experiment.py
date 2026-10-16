"""完整调度器实验契约（假模型/假数据，无 torch）。"""
import json
from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from al_fixtures import default_al, make_cfg, scheduler_of
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.decision.thresholds import PassThresholds
from lingbotvla.auto_learning.state.persistence import config_fingerprint, collect_state, restore_state
from lingbotvla.auto_learning.types import TaskStatus


def no_score(*args, **kwargs):
    raise AssertionError("hardness scorer must never run")


class SkillExperimentTests(unittest.TestCase):
    def setUp(self):
        # 生产计时函数会懒导入 torch；模拟器不需要 CUDA 同步。
        sync = patch("lingbotvla.auto_learning.orchestration.scheduler._cuda_sync_for_timing")
        sync.start()
        self.addCleanup(sync.stop)

    def config(self, **changes):
        options = dict(pass_transition_enabled=False, task_scan_enabled=False,
            hardness_scan_enabled=False, early_defer_on_overfit=False,
            min_steps_before_defer=10000, eval_interval_steps=100, defer_retry_steps=100,
            replay_task_names=[], review_after_task_transitions=0,
            rescan_candidates_after_transition=False, max_global_steps=1000,
            batch_size=2, new_slots=2, replay_slots=0)
        options.update(changes)
        return default_al(**options)

    def test_full_1000_steps_even_when_metrics_pass_no_scout_or_score(self):
        s = scheduler_of(make_cfg(1, curves={"t0": [.1] * 12}, al=self.config(),
                                  n_train_trajs=4, samples_per_traj=3))
        s.scanner.scorer.score = no_score
        splits = []
        original = s.evaluator.evaluate
        def evaluate(task, split, ids):
            splits.append((split, tuple(ids)))
            return original(task, split, ids)
        s.evaluator.evaluate = evaluate
        s.run()
        rec = s.registry.get("t0")
        self.assertEqual(s.state.global_step, 1000)
        self.assertEqual(s.state.units_run, 10)
        self.assertEqual(rec.status, TaskStatus.CANDIDATE.value)  # 正常结束撤销当前 attempt
        self.assertEqual(rec.total_task_steps, 1000)
        self.assertFalse(s.state.newly_passed or s.state.auto_passed)
        self.assertEqual(set(x[0] for x in splits), {"train_monitor", "active_val"})
        self.assertEqual(len(splits), 22)  # 初始两次 + 每 unit 两次
        self.assertTrue(set(rec.train_monitor_ids).isdisjoint(rec.active_val_ids))
        self.assertEqual(set(rec.sample_probs), set(rec.sample_ids))
        self.assertEqual(len(set(rec.sample_probs.values())), 1)

    def test_no_pass_for_bootstrap_or_rescan_but_metrics_remain(self):
        s = scheduler_of(make_cfg(1, curves={"t0": [.1] * 4},
                                  al=self.config(task_scan_enabled=True)))
        event = s.advance()
        self.assertEqual(event["result"], "candidate")
        s._rescan()
        self.assertEqual(s.registry.get("t0").status, "CANDIDATE")
        self.assertTrue(s.registry.get("t0").eval_history)

    def test_hardness_rescue_four_combinations(self):
        for enabled in (False, True):
            for rescue in (False, True):
                with self.subTest(hardness=enabled, rescue=rescue):
                    s = scheduler_of(make_cfg(1, curves={"t0": [.9]*8},
                        n_train_trajs=4, samples_per_traj=3, al=self.config(
                            hardness_scan_enabled=enabled, defer_resample_retry=rescue,
                            min_steps_before_defer=100, min_lp50=10, max_rounds=1)))
                    calls = []
                    score = s.scanner.scorer.score
                    def counted(*args):
                        calls.append(args)
                        return score(*args)
                    s.scanner.scorer.score = counted
                    s.run()
                    self.assertEqual(len(calls), (1 + int(rescue)) if enabled else 0)
                    self.assertEqual(s.state.global_step, 200 if rescue else 100)

    def test_legacy_default_can_still_bootstrap_pass(self):
        s = scheduler_of(make_cfg(1, curves={"t0": [.1]*4}))
        s.run()
        self.assertEqual(s.registry.get("t0").status, "PASS")
        self.assertEqual(s.state.global_step, 0)

    def test_rescue_and_continue_ignore_imported_weights_when_scan_off(self):
        s = scheduler_of(make_cfg(1, curves={"t0": [.9] * 9},
            al=self.config(min_steps_before_defer=100, defer_resample_retry=True,
                           refresh_hardness_on_continue=True, max_global_steps=400)))
        s.scanner.scorer.score = no_score
        rec = s.registry.get("t0")
        raw = s.registry.to_state()
        raw["records"]["t0"]["sample_probs"] = {str(rec.sample_ids[0]): 1}
        raw["records"]["t0"]["pass_sampling_snapshot"] = {str(rec.sample_ids[0]): 1}
        s.import_pool_registry(raw)
        self.assertFalse(s.registry.get("t0").sample_probs)
        self.assertIsNone(s.registry.get("t0").pass_sampling_snapshot)
        s.run()
        rescue = [e for e in s.events if e.get("decision") == "RESCUE"]
        self.assertTrue(rescue)
        self.assertTrue(all(e["hardness_refreshed"] is False for e in rescue))
        self.assertTrue(all("均匀" in e["reason"] for e in rescue))

    def test_explicit_replay_does_not_claim_pass_or_enter_learning(self):
        s = scheduler_of(make_cfg(2, al=self.config(learning_task_names=["t1"],
            replay_task_names=["t0"], new_slots=1, replay_slots=1)))
        s.scanner.scorer.score = no_score
        self.assertEqual(s.replay_plan().tasks, ["t0"])
        self.assertEqual(s.registry.get("t0").status, "CANDIDATE")
        s.run()
        self.assertEqual(s.state.trained_tasks, ["t1"])
        self.assertTrue(all(e["old_tasks"] == ["t0"] for e in s.events
                            if e.get("action") == "train_unit"))
        self.assertFalse(s.registry.get("t0").ever_passed)

    def test_reference_and_initial_ratios_persist(self):
        cfg = self.config(pass_metric="gmean_mse", pass_thresholds_file="fixture.json")
        cfg.pass_thresholds = PassThresholds(tasks={"t0": .01}, stat="geomean")
        s = scheduler_of(make_cfg(1, al=cfg))
        s.run()
        rec = s.registry.get("t0")
        history = [e for e in rec.eval_history if e["kind"] == "active_val"]
        self.assertEqual(history[0]["val_to_initial"], 1)
        self.assertAlmostEqual(history[-1]["val_to_reference"], history[-1]["gmean_mse"] / .01)
        snap = collect_state(s)
        clone = scheduler_of(make_cfg(1, al=cfg))
        restore_state(clone, snap)
        self.assertEqual(clone.registry.get("t0").initial_val_gmean_mse, rec.initial_val_gmean_mse)

    def test_invalid_initial_eval_does_not_become_post_training_baseline(self):
        cfg = self.config(pass_metric="gmean_mse", pass_thresholds_file="fixture.json")
        cfg.pass_thresholds = PassThresholds(tasks={"t0": .01}, stat="geomean")
        s = scheduler_of(make_cfg(1, al=cfg))
        evaluate = s.evaluator.evaluate
        initial = [True]
        def bad_first(task, split, ids):
            metrics = evaluate(task, split, ids)
            if split == "active_val" and initial[0]:
                initial[0] = False
                return replace(metrics, per_traj_mse={}, metric_valid=False)
            return metrics
        s.evaluator.evaluate = bad_first
        s.run()
        self.assertIsNone(s.registry.get("t0").initial_val_gmean_mse)
        self.assertTrue(all(e.get("val_to_initial") is None
                            for e in s.registry.get("t0").eval_history))

    def test_default_fingerprint_unchanged_new_modes_are_guarded(self):
        base = AutoLearningConfig()
        fp = config_fingerprint(base, ["t0"])
        self.assertNotIn("pass_transition_enabled", fp)
        self.assertNotEqual(fp, config_fingerprint(self.config(), ["t0"]))
        for changes in (dict(learning_task_names=["t0", "t0"]),
                        dict(learning_task_names=["t0"], replay_task_names=["t0"]),
                        dict(task_scan_enabled=False, pool_filter_by_gmean_ratio=True),
                        dict(pass_transition_enabled=False, target_total_passed_tasks=1)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                AutoLearningConfig(**changes)
        with self.assertRaises(ValueError):
            scheduler_of(make_cfg(1, al=self.config(learning_task_names=["missing"])))

    def test_reference_is_raw_geomean_for_exact_active_val_ids(self):
        root = Path(__file__).resolve().parents[1]
        ref = json.loads((root / "configs/auto_learning/pass_thresholds_gmean1_active4.json").read_text())
        raw = [json.loads(s) for s in (root / ref["calibration"]["source_file"]).read_text().splitlines()]
        for task, ids in ref["trajectory_ids"].items():
            values = [r["mse"] for r in raw if r["task"] == task and r["traj"] in ids]
            self.assertEqual(len(values), 4)
            self.assertAlmostEqual(ref["tasks"][task], math.exp(math.fsum(math.log(v) for v in values)/4))


    # =================================================================== #
    # 存档后样本重扫（2026-10-16 用户要求：任务启动 + 周期存档 两个触发机制）
    # =================================================================== #
    def _to_training(self, s, task="t0"):
        for _ in range(50):
            if s.registry.get(task).status == TaskStatus.TRAINING.value:
                return s.registry.get(task)
            s.advance()
        self.fail(f"{task} 未在 50 次 advance 内进入 TRAINING")

    def test_refresh_current_hardness_after_save_updates_probs_for_next_unit(self):
        s = scheduler_of(make_cfg(1, curves={"t0": [.9] * 8}, n_train_trajs=4,
                                  samples_per_traj=3,
                                  al=self.config(hardness_scan_enabled=True)))
        rec = self._to_training(s)
        v0 = rec.hardness_version
        self.assertGreaterEqual(v0, 1)          # 任务启动时已扫过一次
        ev = s.refresh_current_hardness()
        self.assertIsNotNone(ev)
        self.assertEqual(ev["action"], "hardness_refresh_save")
        self.assertEqual(ev["task"], "t0")
        self.assertEqual(rec.hardness_version, v0 + 1)   # 轮转窗口 ⇒ 重新扫
        self.assertEqual(set(rec.sample_probs), set(rec.sample_ids))
        # 新 probs 在**下一个 unit 边界**生效：下一次 _prepare_train_request 取到它
        req, _ = s._prepare_train_request()
        self.assertEqual(req.probs, rec.sample_probs)

    def test_refresh_current_hardness_noop_when_disabled_or_idle(self):
        s = scheduler_of(make_cfg(1, curves={"t0": [.9] * 4},
                                  al=self.config(hardness_scan_enabled=False)))
        s.scanner.scorer.score = no_score        # 关闭后连 scorer 都不许调用
        self._to_training(s)
        self.assertIsNone(s.refresh_current_hardness())

    def test_hook_on_save_triggers_refresh_and_records_event(self):
        from lingbotvla.auto_learning.real.hook import AutoLearnLoopHook
        s = scheduler_of(make_cfg(1, curves={"t0": [.9] * 6}, n_train_trajs=4,
                                  samples_per_traj=3,
                                  al=self.config(hardness_scan_enabled=True)))
        hook = AutoLearnLoopHook(scheduler=s, sampler=None, cfg=s.al)
        rec = self._to_training(s)
        v0 = rec.hardness_version
        ev = hook.on_save(100)
        self.assertEqual(ev["action"], "hardness_refresh_save")
        self.assertEqual(rec.hardness_version, v0 + 1)
        self.assertTrue(any(e.get("action") == "hardness_refresh_save"
                            for e in hook._events))

    def test_save_scans_with_real_hook_and_sampler_across_1000_steps(self):
        from lingbotvla.auto_learning.real.hook import AutoLearnLoopHook
        from lingbotvla.auto_learning.real.sampler import AutoLearnSampler
        from lingbotvla.auto_learning.sampling.sampler import BatchSampler
        s = scheduler_of(make_cfg(1, curves={"t0": [.9] * 20}, n_train_trajs=4,
                                  samples_per_traj=3,
                                  al=self.config(hardness_scan_enabled=True)))
        sampler = AutoLearnSampler(BatchSampler(s.al, s.backend.resolver,
                                                s.backend.catalog, s.rng), batch_size=2)
        hook = AutoLearnLoopHook(scheduler=s, sampler=sampler, cfg=s.al)
        score = s.scanner.scorer.score
        with patch.object(s.scanner.scorer, "score", wraps=score) as initial, \
                patch.object(s.scanner.scorer, "score_fresh", wraps=score, create=True) as fresh:
            events, requests = [], {}
            for step in range(1, 1001):
                directive = hook.on_step_begin(step - 1)
                self.assertFalse(directive.finished)
                if directive.rebuild_iterator:
                    iterator = iter(sampler)
                    requests[step] = s.registry.get("t0").hardness_version
                for _ in range(2):
                    next(iterator)
                hook.on_step_end(step, .9)
                if step % 250 == 0:
                    before = dict(sampler._prepared.request.probs)
                    # 绝对存档步号与 AL 相对步号可以不同；不能改写 Scheduler 的记账。
                    ev = hook.on_save(10000 + step)
                    if ev is not None:
                        events.append(ev)
                    self.assertEqual(sampler._prepared.request.probs, before)
            self.assertEqual(initial.call_count, 1)
            self.assertEqual(fresh.call_count, 3)  # 250/500/750；1000 收尾不再扫描。
        self.assertEqual([e["step"] for e in events], [250, 500, 750])
        self.assertEqual([e["scheduler_step"] for e in events], [200, 500, 700])
        self.assertEqual([e["checkpoint_step"] for e in events], [10250, 10500, 10750])
        self.assertEqual([requests[n] for n in (1, 201, 301, 501, 701, 801)],
                         [1, 1, 2, 3, 3, 4])
        self.assertEqual(s.state.global_step, 1000)

    def test_save_scan_skips_idle_finished_budget_and_explicit_stop(self):
        from lingbotvla.auto_learning.real.hook import AutoLearnLoopHook
        s = scheduler_of(make_cfg(1, curves={"t0": [.9] * 6}, n_train_trajs=4,
                                  samples_per_traj=3,
                                  al=self.config(hardness_scan_enabled=True)))
        self.assertIsNone(s.refresh_current_hardness())  # 尚未选择任务。
        self._to_training(s)
        s.scanner.scorer.score = no_score
        hook = AutoLearnLoopHook(scheduler=s, sampler=None, cfg=s.al)
        self.assertIsNone(hook.on_save(250, will_continue=False))
        self.assertIsNone(s.refresh_current_hardness(step=1000))
        s.state.finished = True
        self.assertIsNone(hook.on_save(250))


if __name__ == "__main__":
    unittest.main()

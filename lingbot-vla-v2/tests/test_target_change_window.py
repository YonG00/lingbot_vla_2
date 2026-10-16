"""CPU contracts: event detection, readable cache, coordinate remap and AL gates."""

from dataclasses import replace
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.ports import TaskEntry
from lingbotvla.auto_learning.sampling.target_change_window import (
    WindowSettings,
    cache_path,
    joint_mapping,
    prepare_task,
    scan_episode,
    TargetChangeScanner,
    write_report,
)
from lingbotvla.auto_learning.state.persistence import collect_state, restore_state
from al_fixtures import default_al, make_cfg, scheduler_of

ARMS = [[f"l_{i}" for i in range(6)], [f"r_{i}" for i in range(6)]]
WRISTS = [["l_4", "l_5"], ["r_4", "r_5"]]
MAPPING = joint_mapping({"arm_joints_name": ARMS}, WRISTS)


def rows(ep=1, n=60, offset=100):
    result = []
    for i in range(n):
        state = [0.0] * 14
        # A sustained gripper command transition, then a hold. Small wrist
        # jitter is below threshold. Neither means an actual force measurement.
        state[6] = max(0, min(1, (i - 15) / 10))
        state[4] = 0.0001 * (i % 2)
        result.append(
            dict(
                index=offset + i,
                episode_index=ep,
                frame_index=i,
                timestamp=i / 20,
                **{"observation.state": state},
            )
        )
    return result


class TargetChangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.embodiment = self.root / "config.json"
        self.embodiment.write_text(json.dumps({"arm_joints_name": ARMS}))
        self.cfg = AutoLearningConfig(
            sample_scan_method="target_change_window",
            target_change_cache_dir=str(self.root / "cache"),
            target_change_dataset_root=str(self.root),
            target_change_embodiment_config=str(self.embodiment),
            target_change_wrist_joint_names=WRISTS,
        )

    def prepare(self, task="cabinet", episodes=(1,), getter=rows, **kw):
        return prepare_task(
            self.cfg, self.root, task, list(episodes), 20, MAPPING, getter, **kw
        )

    def test_names_are_resolved_and_never_guessed(self):
        swapped = [ARMS[0][::-1], ARMS[1]]
        mapping = joint_mapping({"arm_joints_name": swapped}, WRISTS)
        self.assertEqual(mapping[0]["index"], 1)
        with self.assertRaisesRegex(ValueError, "absent"):
            joint_mapping({"arm_joints_name": ARMS}, [["wrist"], WRISTS[1]])
        with self.assertRaises(ValueError):
            joint_mapping({"arm_joints_name": ARMS}, [])

    def test_sustained_change_start_stop_and_window_mass(self):
        raw = self.prepare()
        ep = raw["episodes"]["1"]
        self.assertEqual(
            [(e["signal"], e["kind"], e["frame"]) for e in ep["events"]],
            [
                ("left:gripper_command", "start", 16),
                ("left:gripper_command", "stop", 26),
            ],
        )
        self.assertEqual(ep["windows"], [[11, 31]])
        self.assertEqual(ep["window_frames"], 21)
        self.assertAlmostEqual(raw["summary"]["window_probability_mass"], 42 / 81)
        self.assertTrue(all(w > 0 for w in ep["weights"]))

    def test_static_hold_and_single_spike_create_no_events(self):
        data = rows()
        for r in data:
            r["observation.state"] = [0.0] * 14
        data[20]["observation.state"][6] = 0.001
        out = scan_episode(data, 20, MAPPING, WindowSettings())
        self.assertEqual(out["events"], [])
        self.assertEqual(out["weights"], [1.0] * len(data))

    def test_isolated_large_spike_is_not_a_grasp(self):
        data = rows()
        for r in data:
            r["observation.state"] = [0.0] * 14
        data[20]["observation.state"][6] = 1.0
        # Consecutive opposite deltas from a spike should not become an event.
        out = scan_episode(data, 20, MAPPING, WindowSettings())
        self.assertEqual(out["events"], [])

    def test_window_radius_clips_at_edges_and_uses_fps(self):
        out = scan_episode(rows(), 20, MAPPING, WindowSettings(window_seconds=10))
        self.assertEqual(out["windows"], [[0, 59]])
        out = scan_episode(rows(), 20, MAPPING, WindowSettings(window_seconds=0))
        self.assertEqual(out["window_frames"], 2)

    def test_bad_frames_timestamps_and_nonfinite_rejected(self):
        for key, value in [("frame_index", 100), ("timestamp", 100), ("index", 100)]:
            data = rows()
            data[20][key] = value
            with self.assertRaises(ValueError):
                scan_episode(data, 20, MAPPING, WindowSettings())
        data = rows()
        data[0]["observation.state"][6] = float("nan")
        with self.assertRaises(ValueError):
            scan_episode(data, 20, MAPPING, WindowSettings())

    def test_invalid_settings_fail_as_config_errors(self):
        for field, value in (
            ("target_change_release_ratio", "0.5"),
            ("target_change_release_ratio", float("nan")),
            ("target_change_release_ratio", True),
            ("target_change_window_weight", 0.5),
            ("target_change_window_seconds", float("inf")),
            ("target_change_persistence_frames", 1.5),
            ("low_noise_fm_enabled", "false"),
            ("sample_scan_method", "mixed"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(self.cfg, **{field: value})

    def test_cache_reuses_without_reading_data_and_rejects_settings(self):
        a = self.prepare()
        with patch(__name__ + ".rows", side_effect=AssertionError("must not rescan")):
            b = self.prepare(getter=rows)
        self.assertEqual(a, b)
        self.cfg.target_change_window_seconds = 0.5
        with self.assertRaisesRegex(ValueError, "changed"):
            self.prepare()
        self.prepare(refresh=True)

    def test_cache_checks_task_episodes_and_readable_source(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.prepare(episodes=(2,))
        path = cache_path(self.cfg.target_change_cache_dir, "cabinet")
        data = json.loads(path.read_text())
        data["source_root"] = "wrong"
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.prepare()

    def test_sample_id_remap_uses_absolute_frame_and_episode(self):
        self.prepare()
        rec = NS(task_name="cabinet", sample_ids=list(range(500, 560)))
        entry = TaskEntry(name="cabinet", train_traj_ids=[1])
        resolver = NS(
            resolve=lambda task, sid: NS(episode_id=1, frame_id=100 + sid - 500)
        )
        scanner = TargetChangeScanner(self.cfg, resolver)
        result = scanner.scan(rec, entry)
        self.assertEqual(set(result.probs), set(rec.sample_ids))
        self.assertAlmostEqual(sum(result.probs.values()), 1.0)
        self.assertEqual(result.weights[516], 2.0)
        self.assertEqual(result.weights[500], 1.0)
        self.assertEqual(result.losses, {})
        from lingbotvla.auto_learning.sampling.hardness_scan import HardnessScan

        restored = HardnessScan.from_state(json.loads(json.dumps(result.to_state())))
        self.assertEqual(restored.scanned_sample_ids, result.scanned_sample_ids)
        self.assertEqual(restored.method, "target_change_window")
        # Cached in process; force_fresh from a legacy caller cannot rescan.
        with patch("pathlib.Path.read_text", side_effect=AssertionError("no reads")):
            self.assertIs(scanner.scan(rec, entry, force_fresh=True), result)
        resolver.resolve = lambda task, sid: NS(episode_id=2, frame_id=100 + sid - 500)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            TargetChangeScanner(self.cfg, resolver).scan(rec, entry)
        resolver.resolve = lambda task, sid: NS(episode_id=1, frame_id=100 + sid - 500)
        rec.sample_ids.pop()
        with self.assertRaisesRegex(ValueError, "frame set mismatch"):
            TargetChangeScanner(self.cfg, resolver).scan(rec, entry)

    def test_report_contains_curves_events_and_video_seek(self):
        raw = self.prepare()
        path = self.root / "report.html"
        write_report(raw, path, {1: ["episode.mp4"]})
        report = path.read_text()
        for text in (
            "polyline",
            "absolute=116",
            "episode.mp4",
            "currentTime=0.8",
            "控制目标变化",
        ):
            self.assertIn(text, report)

    def test_static_scheduler_never_scores_at_select_save_rescue_or_resume(self):
        options = dict(
            pass_transition_enabled=False,
            task_scan_enabled=False,
            hardness_scan_enabled=True,
            sample_scan_method="target_change_window",
            target_change_cache_dir=self.cfg.target_change_cache_dir,
            target_change_dataset_root=self.cfg.target_change_dataset_root,
            target_change_embodiment_config=str(self.embodiment),
            target_change_wrist_joint_names=WRISTS,
            early_defer_on_overfit=False,
            min_steps_before_defer=100,
            min_lp50=10,
            eval_interval_steps=100,
            defer_retry_steps=100,
            defer_resample_retry=True,
            replay_task_names=[],
            review_after_task_transitions=0,
            rescan_candidates_after_transition=False,
            max_global_steps=300,
            max_rounds=1,
            batch_size=2,
            new_slots=2,
            replay_slots=0,
        )
        al = default_al(**options)
        # Sim resolver's absolute frame coordinates are its own synthetic IDs.
        base = scheduler_of(
            make_cfg(
                1,
                al=replace(al, hardness_scan_enabled=False),
                n_train_trajs=4,
                samples_per_traj=60,
            )
        )
        entry = base.backend.catalog.entry("t0")

        def getter(ep):
            data = rows(ep, n=60, offset=0)
            for row, sid in zip(data, entry.samples_by_traj[ep]):
                row["index"] = base.backend.resolver.resolve("t0", sid).frame_id
            return data

        self.prepare("t0", entry.train_traj_ids, getter)
        base.al = al
        from lingbotvla.auto_learning.sampling.target_change_window import (
            TargetChangeScanner,
        )

        base.scanner = TargetChangeScanner(al, base.backend.resolver)
        base.restore_static_sampling()
        with patch(
            "lingbotvla.auto_learning.orchestration.scheduler._cuda_sync_for_timing"
        ), patch.object(
            base.backend.scorer, "score", side_effect=AssertionError("no model scan")
        ):
            base.run(max_actions=2)  # select, then a real simulated training unit
            self.assertEqual(base.events[0]["action"], "select")
            self.assertEqual(base.state.global_step, 100)
            self.assertEqual(base.events[-1]["decision"], "RESCUE")
            self.assertIn("复用静态", base.events[-1]["reason"])
            from lingbotvla.auto_learning.real.hook import AutoLearnLoopHook

            hook = AutoLearnLoopHook(scheduler=base, sampler=None, cfg=al)
            self.assertIsNone(hook.on_save(100))
            # The real hook switches Scheduler into externally driven mode.
            # Return to the simulation trainer for the remainder of this test.
            base.defer_train = False
            self.assertIsNone(base.refresh_current_hardness(step=50))
            rec = base.registry.get("t0")
            before = dict(rec.sample_probs)
            event = {}
            base._refresh_hardness(rec, event, force_fresh=True)
            self.assertFalse(event["hardness_refreshed"])
            state = collect_state(base)
            # IDs are valid but an older model/method used different weights.
            state["registry"]["records"]["t0"]["sample_probs"] = {
                str(rec.sample_ids[0]): 1.0
            }
            restore_state(base, state)
            self.assertEqual(base.registry.get("t0").sample_probs, before)
            base.run(max_actions=10)
            self.assertTrue(base.state.finished)
            self.assertTrue(any(e.get("status_after") == "DEFER" for e in base.events))

    def test_static_constructor_pool_import_and_explicit_replay_use_current_cache(self):
        al = replace(
            self.cfg,
            task_scan_enabled=False,
            pass_transition_enabled=False,
            learning_task_names=["t0"],
            replay_task_names=["t1"],
        )
        fixture = make_cfg(
            2,
            al=replace(al, hardness_scan_enabled=False),
            n_train_trajs=4,
            samples_per_traj=60,
        )
        source = scheduler_of(fixture)
        for task in ("t0", "t1"):
            entry = source.backend.catalog.entry(task)

            def getter(ep):
                data = rows(ep)
                for row, sid in zip(data, entry.samples_by_traj[ep]):
                    row["index"] = source.backend.resolver.resolve(task, sid).frame_id
                return data

            self.prepare(task, entry.train_traj_ids, getter)
        # Exercise the real constructor, rather than injecting a scanner.
        fixture.auto_learning = al
        sched = scheduler_of(fixture)
        expected = dict(sched.registry.get("t1").sample_probs)
        pool = sched.registry.to_state()
        record = pool["records"]["t1"]
        record["status"] = "PASS"
        record["sample_probs"] = {str(next(iter(expected))): 1.0}
        record["pass_sampling_snapshot"] = dict(record["sample_probs"])
        sched.import_pool_registry(pool)
        plan = sched.replay_plan()
        self.assertEqual(plan.tasks, ["t1"])
        self.assertEqual(plan.probs["t1"], expected)
        self.assertEqual(sched.registry.get("t1").pass_sampling_snapshot, expected)
        self.assertEqual(set(plan.probs["t1"]), set(plan.sample_ids["t1"]))
        self.assertAlmostEqual(max(expected.values()) / min(expected.values()), 2.0)

    def test_parquet_preparation_train_only_cache_report_and_real_resolver(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from experiment.robotwin.prepare_target_change_windows import prepare
        from lingbotvla.auto_learning.resolver import SampleResolver

        (self.root / "meta").mkdir()
        (self.root / "data").mkdir()
        video = self.root / "videos/episode_000001.mp4"
        video.parent.mkdir()
        video.write_bytes(b"video-link-fixture")
        (self.root / "meta/info.json").write_text(
            json.dumps(
                dict(
                    fps=20,
                    chunks_size=1000,
                    video_path="videos/episode_{episode_index:06d}.mp4",
                    features={"observation.images.head": {"dtype": "video"}},
                )
            )
        )
        train = rows(ep=1)
        validation = rows(ep=2, offset=1000)
        # A validation NaN would fail scanning: proving no validation trajectory
        # is included, rather than merely checking a train-only filename.
        for row in validation:
            row["observation.state"][6] = float("nan")
        pq.write_table(
            pa.Table.from_pylist(validation + train),
            self.root / "data/episodes.parquet",
        )
        manifest = {"tasks": {"cabinet": {"train_ids": [1], "val_ids": [2]}}}
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.cfg, self.root, manifest, ["cabinet"])
        path = cache_path(self.cfg.target_change_cache_dir, "cabinet")
        raw = json.loads(path.read_text())
        self.assertEqual(set(raw["episodes"]), {"1"})
        self.assertEqual(raw["summary"]["frames"], 60)
        self.assertIn(video.resolve().as_uri(), path.with_suffix(".html").read_text())
        resolver = SampleResolver(
            episode_index=np.array([1] * 60),
            frame_index=np.arange(100, 160),
            task_of_episode=lambda ep: "cabinet",
        )
        rec = NS(task_name="cabinet", sample_ids=list(range(60)))
        result = TargetChangeScanner(self.cfg, resolver).scan(
            rec, TaskEntry(name="cabinet", train_traj_ids=[1])
        )
        self.assertAlmostEqual(sum(result.probs.values()), 1.0)
        self.assertAlmostEqual(result.probs[16] / result.probs[0], 2.0)
        # Existing cache reuse must not scan the Parquet numeric rows again.
        with patch("pyarrow.dataset.dataset") as source, contextlib.redirect_stdout(
            io.StringIO()
        ):
            source.return_value.to_table.side_effect = AssertionError(
                "unexpected rescan"
            )
            prepare(self.cfg, self.root, manifest, ["cabinet"])
        self.assertEqual(json.loads(path.read_text()), raw)
        bad = {"tasks": {"cabinet": {"train_ids": [1], "val_ids": [1]}}}
        with self.assertRaisesRegex(ValueError, "train/val"):
            prepare(self.cfg, self.root, bad, ["cabinet"])

    def test_scan_off_ignores_static_files_and_stays_uniform(self):
        al = default_al(
            hardness_scan_enabled=False,
            sample_scan_method="target_change_window",
            task_scan_enabled=False,
            pass_transition_enabled=False,
            target_change_cache_dir=str(self.root / "missing"),
        )
        base = scheduler_of(make_cfg(1, al=al))
        with patch.object(
            base.backend.scorer, "score", side_effect=AssertionError("no model scan")
        ):
            base.advance()
        self.assertEqual(
            set(base.registry.get("t0").sample_probs.values()),
            {1 / len(base.registry.get("t0").sample_ids)},
        )


if __name__ == "__main__":
    unittest.main()

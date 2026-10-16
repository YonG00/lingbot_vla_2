"""Pure CPU contracts for the single-GPU RoboTwin benchmark adapter."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "experiment/robotwin/run_clean_benchmark.py"
spec = importlib.util.spec_from_file_location("clean_benchmark_adapter", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark)


class NvidiaAdapterTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "precision": "fp32",
            "use_compile": False,
            "video": False,
            "global_view": False,
            "global_view_fps": 60,
            "episodes": 3,
            "model_path": Path("/models/run/checkpoints/global_step_100/hf_ckpt"),
            "robotwin_root": Path("/sim/RoboTwin"),
            "base_port": 9330,
            "inference_env": "lingbotvla",
            "sim_env": "RoboTwin",
            "conda_sh": Path("/opt/conda/etc/profile.d/conda.sh"),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_launcher_is_single_cuda_slot_and_uses_repo_client(self):
        command = benchmark._launcher_command(
            self._args(precision="bf16", use_compile=True, video=True),
            Path("/tmp/run"), "demo_clean", Path("/tmp/tasks.txt"),
        )
        self.assertEqual(command[command.index("--num_gpus") + 1], "1")
        self.assertEqual(command[command.index("--num_per_gpu") + 1], "1")
        self.assertEqual(command[command.index("--use_bf16") + 1], "True")
        self.assertEqual(command[command.index("--use_fp32") + 1], "False")
        self.assertIn("--enable_video", command)
        self.assertIn(str(benchmark.REPO_ROOT / "experiment/robotwin/start_robotwin_infer_and_eval.sh"), command)
        workdir = command[command.index("--inference_workdir") + 1]
        self.assertEqual(workdir, f"{benchmark.REPO_ROOT}/")
        self.assertTrue((Path(workdir) / "experiment/robotwin/eval_policy_client_lingbotvla.py").is_file())

    def test_result_reader_uses_last_complete_client_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "task.log"
            log.write_text("Success rate: 1/2 => 50.0%\r\n\x1b[92mSuccess rate: 2/3 => 66.7%\x1b[0m\n")
            self.assertEqual(benchmark.read_result(log), (2, 3))

    def test_global_view_is_independent_of_head_and_passes_fps(self):
        for head in (False, True):
            for enabled in (False, True):
                with self.subTest(head=head, enabled=enabled):
                    command = benchmark._launcher_command(
                        self._args(video=head, global_view=enabled, global_view_fps=30),
                        Path('/tmp/run'), 'demo_clean', Path('/tmp/tasks.txt'))
                    self.assertEqual('--global_view' in command, enabled)
                    self.assertIn('--enable_video' if head else '--no_video', command)
                    if enabled:
                        self.assertEqual(command[command.index('--global_view_fps')+1], '30')

    def test_global_view_manifest_separates_fps_but_preserves_old_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / 'model.safetensors.index.json').write_text('{}')
            args = self._args(model_path=model, qwen3vl_path=model, task_config='demo_clean')
            old = benchmark._manifest(args, ['click_bell'])
            self.assertNotIn('global_view', old)
            args.global_view_fps = 30
            self.assertEqual(benchmark._manifest(args, ['click_bell']), old)
            args.global_view = True
            recording = benchmark._manifest(args, ['click_bell'])
            self.assertEqual(recording['global_view']['fps'], 30)
            args.global_view_fps = 60
            self.assertNotEqual(benchmark._manifest(args, ['click_bell']), recording)

    def test_only_full_episode_results_create_resume_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            launcher_run = run / "launcher"
            logs = launcher_run / "eval_logs"
            logs.mkdir(parents=True)
            (logs / "click_bell.log").write_text("Success rate: 2/3 => 66.7%\n")
            self.assertEqual(
                benchmark._record_task(run, "demo_clean", "click_bell", 3, launcher_run),
                (2, 3),
            )
            self.assertTrue(benchmark._done_path(run, "demo_clean", "click_bell").is_file())
            (logs / "place_fan.log").write_text("Success rate: 1/2 => 50.0%\n")
            self.assertIsNone(
                benchmark._record_task(run, "demo_clean", "place_fan", 3, launcher_run)
            )
            self.assertFalse(benchmark._done_path(run, "demo_clean", "place_fan").exists())

    def test_both_runs_have_independent_clean_and_randomized_configs(self):
        self.assertEqual(
            benchmark.benchmark_configs("both"),
            ("demo_clean", "demo_randomized"),
        )

    def test_task_list_matches_existing_nvidia_launcher(self):
        tasks = benchmark.load_tasks()
        self.assertEqual(len(tasks), 50)
        self.assertEqual(len(set(tasks)), 50)
        self.assertEqual(tasks[0], "lift_pot")

    def test_launcher_success_records_only_requested_complete_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            args = self._args(qwen3vl_path=Path("/models/qwen"))

            def fake_run(command, **kwargs):
                self.assertEqual(command[command.index("--num_gpus") + 1], "1")
                output_base = Path(command[command.index("--output_base") + 1])
                launcher_run = output_base / "fake_run"
                logs = launcher_run / "eval_logs"
                logs.mkdir(parents=True)
                (logs / "click_bell.log").write_text("Success rate: 2/3 => 66.7%\n")
                return SimpleNamespace(returncode=0)

            with patch.object(benchmark.subprocess, "run", side_effect=fake_run):
                benchmark._run_config(args, run, "demo_clean", ["click_bell"])
            marker = benchmark._done_path(run, "demo_clean", "click_bell")
            self.assertEqual(benchmark.json.loads(marker.read_text())["successes"], 2)


if __name__ == "__main__":
    unittest.main()

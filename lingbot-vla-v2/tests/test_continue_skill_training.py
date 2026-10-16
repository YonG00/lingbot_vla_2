"""便捷续练/关机流程的 CPU 契约；所有子进程均替身，绝不执行训练或关机。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("continue_skill", ROOT / "experiment/robotwin/continue_skill_training.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


class ContinueSkillTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.source = self.root / "old_run"
        self.source.mkdir()
        self.model = self.source / "checkpoints/global_step_1000/hf_ckpt"
        self.make_hf(self.model)
        self.out = self.root / "new_run"
        self.final = self.out / "checkpoints/global_step_500/hf_ckpt"
        self.audit = self.root / "audit.json"
        self.qwen = self.root / "Qwen3-VL"
        (self.source / "lingbotvla_cli.yaml").write_text(json.dumps(
            dict(model=dict(tokenizer_path=str(self.qwen)))))
        (self.source / "datasets.txt").write_text(f"robotwin {self.root / 'dataset'}\n")
        (self.source / "al_config.json").write_text(json.dumps(dict(data_audit_file=str(self.audit))))
        self.args = NS(from_hf=self.model, qwen3vl=None, out=self.out,
            output_root=self.root, steps=500, micro=24, gas=1, shutdown=False,
            al_config=ROOT / "configs/auto_learning/learn_skill_mix_task.json",
            manifest=ROOT / "refs/al_artifacts/al/task_splits_50/manifest.json",
            learning_tasks=None, master_port=62510, runtime_dir=self.root / "launches")
        self.args.run_name = "cabinet_from1000_500"

    def make_hf(self, model):
        model.mkdir(parents=True)
        header = json.dumps(dict(weight=dict(dtype="BF16", shape=[1], data_offsets=[0, 2]))).encode()
        (model / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"weight": "part.safetensors"})))
        (model / "part.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (model / name).write_text("{}")

    def job(self, shutdown=False):
        return dict(command=["fake-trainer"], source_model=str(self.model), output_dir=str(self.out),
                    steps=500, shutdown=shutdown, status_file=str(self.root / "status.json"))

    def run_job(self, job, returncode=0):
        with patch.object(entry.subprocess, "run", return_value=NS(returncode=returncode)) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            rc = entry.run_job(job)
        return rc, run

    def test_source_config_and_data_are_derived_and_validated_by_existing_launcher(self):
        with patch.object(entry.launcher, "build_plan", return_value={}) as build:
            job, _ = entry.make_job(self.args)
        cfg = build.call_args.args[0]
        self.assertEqual(cfg.model, self.model)
        self.assertEqual(cfg.train_config, self.source / "lingbotvla_cli.yaml")
        self.assertEqual(cfg.qwen3vl, self.qwen)
        self.assertEqual(cfg.audit_report, self.audit)
        self.assertEqual((cfg.steps, cfg.micro, cfg.gas), (500, 24, 1))
        self.assertFalse(job["shutdown"])
        self.assertIn("cabinet_from1000_500", job["log"])
        self.assertEqual(job["command"][job["command"].index("--steps") + 1], "500")
        self.assertFalse(self.out.exists())

    def test_al_parameters_can_be_overridden_on_command_line(self):
        self.args.sample_scan = False
        self.args.eval_interval_steps = 50
        self.args.defer_steps = 5000
        self.args.review_every = 0
        self.args.low_noise_fm = True
        self.args.sample_scan_method = 'target_change_window'
        self.args.al_set = ["hardness_weight_max=2.5"]
        with patch.object(entry.launcher, "build_plan", return_value={"config": {}}) as build:
            entry.make_job(self.args)
        cfg = build.call_args.args[0].al_config_data
        self.assertFalse(cfg["hardness_scan_enabled"])
        self.assertEqual(cfg["eval_interval_steps"], 50)
        self.assertEqual(cfg["min_steps_before_defer"], 5000)
        self.assertEqual(cfg["review_after_task_transitions"], 0)
        self.assertEqual(cfg["hardness_weight_max"], 2.5)
        self.assertTrue(cfg["low_noise_fm_enabled"])
        self.assertEqual(cfg["sample_scan_method"], 'target_change_window')

    def test_al_overrides_are_typed_validated_and_cannot_replace_launcher_budget(self):
        self.args.al_set = ["task_names=[\"another_task\"]"]
        with self.assertRaisesRegex(ValueError, "launcher-owned"):
            entry.make_job(self.args)
        self.args.al_set = ["eval_interval_steps=0"]
        with self.assertRaisesRegex(ValueError, "eval_interval_steps"):
            entry.make_job(self.args)

    def test_multiple_source_datasets_fail_instead_of_guessing(self):
        (self.source / "datasets.txt").write_text("robotwin /data/a\nrobotwin /data/b\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            entry.make_job(self.args)

    def test_background_job_is_detached_and_freezes_config(self):
        snapshot={"hardness_scan_enabled": True,"sample_scan_method":"target_change_window",
                  "low_noise_fm_enabled":True,"target_change_window_seconds":0.4}
        with patch.object(entry.launcher, "build_plan", return_value={"config": snapshot}):
            job, plan = entry.make_job(self.args)
        with patch.object(entry.subprocess, "Popen", return_value=NS(pid=123)) as spawn:
            self.assertEqual(entry.start_background(job, plan), 123)
        self.assertEqual(spawn.call_args.args[0][:2], ["setsid", "nohup"])
        self.assertEqual(spawn.call_args.kwargs["stdin"], entry.subprocess.DEVNULL)
        saved = entry.launcher.read_json(Path(job["job_file"]))
        cfg = Path(saved["command"][saved["command"].index("--al-config") + 1])
        self.assertEqual(entry.launcher.read_json(cfg), plan["config"])

    def test_existing_launch_artifacts_are_not_overwritten(self):
        with patch.object(entry.launcher, "build_plan", return_value={"config": {}}):
            job, plan = entry.make_job(self.args)
        Path(job["job_file"]).parent.mkdir(parents=True)
        Path(job["job_file"]).write_text("preserve")
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            entry.start_background(job, plan)
        self.assertEqual(Path(job["job_file"]).read_text(), "preserve")

    def test_shutdown_defaults_off_and_can_be_enabled_or_disabled_explicitly(self):
        for flags, expected in (([], False), (["--shutdown"], True), (["--no-shutdown"], False)):
            with self.subTest(flags=flags), patch.object(entry, "make_job") as make, \
                    patch.object(entry, "start_background") as spawn, contextlib.redirect_stdout(io.StringIO()):
                def fake(args):
                    self.assertEqual(args.shutdown, expected)
                    return dict(source_model="source", output_dir="out", log="log", status_file="status"), \
                        {"config": {"learning_task_names": ["put_object_cabinet"], "hardness_scan_enabled": True}}
                make.side_effect = fake
                self.assertEqual(entry.main(["--from-hf", str(self.model), *flags]), 0)
                spawn.assert_not_called()  # 默认预览不训练，也不关机。

    def test_success_without_shutdown_only_runs_training(self):
        self.make_hf(self.final)
        rc, run = self.run_job(self.job())
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(entry.launcher.read_json(self.root / "status.json")["state"], "complete")

    def test_success_with_shutdown_runs_bash_after_final_checkpoint_validation(self):
        self.make_hf(self.final)
        rc, run = self.run_job(self.job(shutdown=True))
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0], ["bash", "/usr/bin/shutdown"])
        self.assertTrue(entry.launcher.read_json(self.root / "status.json")["training_complete"])

    def test_training_failure_never_shuts_down_even_with_checkpoint_present(self):
        self.make_hf(self.final)
        rc, run = self.run_job(self.job(shutdown=True), returncode=1)
        self.assertEqual(rc, 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(entry.launcher.read_json(self.root / "status.json")["state"], "training_failed")

    def test_missing_final_asset_prevents_shutdown(self):
        self.make_hf(self.final)
        (self.final / "tokenizer.json").unlink()
        rc, run = self.run_job(self.job(shutdown=True))
        self.assertEqual(rc, 1)
        self.assertEqual(run.call_count, 1)

    def test_shutdown_command_failure_is_recorded_without_retry(self):
        self.make_hf(self.final)
        with patch.object(entry.subprocess, "run", side_effect=[NS(returncode=0), OSError("shutdown failed")]) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.run_job(self.job(shutdown=True)), 1)
        status = entry.launcher.read_json(self.root / "status.json")
        self.assertEqual(status["state"], "shutdown_failed")
        self.assertTrue(status["training_complete"])
        self.assertEqual(run.call_count, 2)

    def test_missing_truncated_or_wrong_final_checkpoint_never_shuts_down(self):
        for mutation in ("missing", "truncated", "wrong_tensors"):
            with self.subTest(mutation=mutation):
                if mutation == "truncated":
                    self.make_hf(self.final)
                    (self.final / "part.safetensors").write_bytes(b"broken")
                elif mutation == "wrong_tensors":
                    (self.final / "part.safetensors").write_bytes(
                        (self.model / "part.safetensors").read_bytes())
                    raw = entry.launcher.read_json(self.model / "model.safetensors.index.json")
                    raw["weight_map"]["extra"] = "part.safetensors"
                    (self.model / "model.safetensors.index.json").write_text(json.dumps(raw))
                rc, run = self.run_job(self.job(shutdown=True))
                self.assertEqual(rc, 1)
                self.assertEqual(run.call_count, 1)

    def test_busy_gpu_or_insufficient_disk_prevents_start(self):
        for pids, free in (("123\n", 100 * 1024**3), ("", 0)):
            with self.subTest(pids=pids), patch.object(entry.subprocess, "run", return_value=NS(stdout=pids)), \
                    patch.object(entry.shutil, "disk_usage", return_value=NS(free=free)):
                with self.assertRaises(RuntimeError):
                    entry.check_machine(self.job(), {"checkpoint_inventory": [{"size": 12 * 1024**3}]})


if __name__ == "__main__":
    unittest.main()

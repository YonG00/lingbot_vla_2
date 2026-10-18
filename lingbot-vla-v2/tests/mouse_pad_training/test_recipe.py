"""鼠标单任务配方CPU契约：不启动模型、GPU训练或关机。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_skill_mix_task_launcher as launcher_fixtures

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mouse_recipe", ROOT / "experiment/robotwin/recipes/20261018_place_mouse_pad_9680_ee_lnfm.py"
)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


class MouseRecipeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = launcher_fixtures.LauncherTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.args = recipe.parser().parse_args([
            "--model", str(f.model), "--train-config", str(f.train),
            "--dataset-root", str(f.args.dataset_root), "--qwen3vl", str(f.args.qwen3vl),
            "--output-root", str(f.root / "outputs"),
            "--runtime-dir", str(f.root / "launches"),
            "--cache-root", str(f.root / "mouse_cache"), "--run-name", "mouse_test",
        ])

    def test_exact_mouse_task_split_and_explicit_full_training(self):
        job, plan, snapshot = recipe.build_job(self.args)
        self.assertEqual(plan["config"]["task_names"], ["place_mouse_pad"])
        self.assertEqual(plan["counts"]["place_mouse_pad"], {"train": 40, "val": 10})
        manifest = recipe.launcher.read_json(self.args.manifest)["tasks"]["place_mouse_pad"]
        self.assertEqual(plan["train_ids"], sorted(manifest["train_ids"]))
        self.assertTrue(all(800 <= ep < 850 for ep in plan["train_ids"]))
        self.assertFalse(set(plan["train_ids"]) & set(manifest["val_ids"]))
        self.assertEqual(job["source_model"], str(self.fixture.model.resolve()))
        self.assertEqual(job["steps"], 500)
        self.assertTrue(plan["config"]["low_noise_fm_enabled"])
        self.assertEqual(plan["config"]["data_audit_instruction_patterns"], {"place_mouse_pad": "mouse"})
        for key in ("freeze_vit", "freeze_vision_encoder", "use_lora", "train_expert_only"):
            self.assertIs(plan["training_config"]["train"][key], False)
        for flag, value in (("--train.ee_fk_loss_enabled", "true"),
                            ("--train.ee_fk_eval_enabled", "true"),
                            ("--train.ee_fk_position_loss_weight", "3.0"),
                            ("--train.ee_fk_orientation_loss_weight", "0.8"),
                            ("--train.ee_fk_gripper_loss_weight", "1.5")):
            self.assertEqual(plan["command"][plan["command"].index(flag) + 1], value)
        self.assertFalse(snapshot.exists())
        self.assertFalse(Path(job["output_dir"]).exists())

    def test_worker_rebuild_retains_full_training_even_if_source_was_frozen(self):
        source = recipe.launcher.read_training_config(self.fixture.train)
        source["train"] = dict(freeze_vit=True, freeze_vision_encoder=True, use_lora=True,
                               train_expert_only=True)
        self.fixture.train.write_text(json.dumps(source))
        job, plan, snapshot = recipe.build_job(self.args)
        snapshot.parent.mkdir()
        snapshot.write_text(json.dumps(plan["training_config"]))
        frozen_al = snapshot.with_name("frozen.al.json")
        frozen_al.write_text(json.dumps(plan["config"]))
        command = job["command"][2:]
        command[command.index("--al-config") + 1] = str(frozen_al)
        command[command.index("--execute")] = "--prepare-only"
        with contextlib.redirect_stdout(io.StringIO()):
            recipe.launcher.main(command)
        out = Path(job["output_dir"])
        training = json.loads((out / "trainer_config.json").read_text())
        for key in ("freeze_vit", "freeze_vision_encoder", "use_lora", "train_expert_only"):
            self.assertIs(training["train"][key], False)
        self.assertEqual(json.loads((out / "al_config.json").read_text())["task_names"], ["place_mouse_pad"])

    def test_default_checkpoint_and_shutdown_and_step_overrides(self):
        args = recipe.parser().parse_args([])
        self.assertIn("global_step_9680/hf_ckpt", str(args.model))
        self.assertEqual(args.steps, 500)
        self.assertFalse(args.execute)
        self.assertFalse(args.shutdown)
        self.args.steps = 750
        self.args.shutdown = True
        job, plan, _ = recipe.build_job(self.args)
        self.assertEqual(job["steps"], 750)
        self.assertTrue(job["shutdown"])
        self.assertEqual(plan["config"]["max_global_steps"], 750)

    def test_busy_gpu_or_failed_preflight_cannot_start(self):
        argv = ["--model", str(self.fixture.model), "--train-config", str(self.fixture.train),
                "--dataset-root", str(self.args.dataset_root), "--qwen3vl", str(self.args.qwen3vl),
                "--output-root", str(self.args.output_root), "--runtime-dir", str(self.args.runtime_dir),
                "--cache-root", str(self.args.cache_root), "--execute"]
        with patch.object(recipe.continuation, "check_machine", side_effect=RuntimeError("GPU busy")), \
             patch.object(recipe.continuation, "start_background") as start, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                recipe.main(argv)
        start.assert_not_called()
        self.assertFalse(self.args.runtime_dir.exists())


if __name__ == "__main__":
    unittest.main()

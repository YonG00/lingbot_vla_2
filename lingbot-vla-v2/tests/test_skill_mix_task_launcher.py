"""实际启动器预检/命令生成；纯 CPU，无训练、无 torch。"""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest

from lingbotvla.auto_learning.config import AutoLearningConfig

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("skill_launcher", ROOT / "experiment/robotwin/launch_skill_mix_task.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model = self.root / "global_step_9680/hf_ckpt"
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text("{}")
        (self.model / "tokenizer_config.json").write_text("{}")
        (self.model / "tokenizer.json").write_text("{}")
        self.write_shard()
        (self.model / "model.safetensors.index.json").write_text(json.dumps(
            dict(weight_map={"weight": "model-01.safetensors"})))
        self.norm = self.root / "norm.json"
        self.norm.write_text("{}")
        self.train = self.root / "source_run_config.json"
        self.train.write_text(json.dumps(dict(data=dict(norm_stats_file=str(self.norm)))))
        (self.root / "dataset/meta").mkdir(parents=True)
        (self.root / "Qwen3-VL").mkdir()
        (self.root / "Qwen3-VL/tokenizer.json").write_text("{}")
        self.args = NS(al_config=ROOT / "configs/auto_learning/learn_skill_mix_task.json",
            learning_tasks=None, replay_tasks=None, steps=1000, micro=1, gas=10, new_ratio=1.,
            out=self.root / "new_run", model=self.model, train_config=self.train,
            dataset_root=self.root / "dataset", qwen3vl=self.root / "Qwen3-VL",
            manifest=ROOT / "refs/al_artifacts/al/task_splits_50/manifest.json",
            audit_report=self.root / "audit.json", python="python", master_port=62510)

    def write_shard(self, dtype="BF16"):
        header = json.dumps({"weight": dict(dtype=dtype, shape=[1], data_offsets=[0, 2])}).encode()
        (self.model / "model-01.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))

    def test_dry_plan_exact_task_split_and_all_effective_flags(self):
        plan = launcher.build_plan(self.args)
        self.assertFalse(self.args.out.exists())
        cfg = AutoLearningConfig(**plan["config"])
        self.assertEqual(cfg.task_names, ["put_object_cabinet"])
        self.assertEqual(plan["counts"]["put_object_cabinet"], dict(train=40, val=10))
        self.assertEqual(len(plan["train_ids"]), 40)
        self.assertTrue(all(2350 <= ep < 2400 for ep in plan["train_ids"]))
        cmd = plan["command"]
        # 固定入口参数通过真实数组核对，避免 shell 拆词/引号问题。
        for flag, expected in (("train.dcp_save_mode", "hf_only"),
            ("train.hf_export_dtype", "bf16"), ("train.reset_step_counter", "true"),
            ("train.max_steps", "1000"), ("train.save_steps", "200"),
            ("train.hf_pass_interval", "0"), ("data.norm_stats_file", str(self.norm.resolve()))):
            self.assertEqual(cmd[cmd.index("--" + flag)+1], expected)
        self.assertIn(str(self.norm), plan["source_files"])
        self.assertIsNone(plan["training_config"]["model"]["vlm_repo_id"])
        self.assertTrue(plan["training_config"]["model"]["post_training"])

    def test_configurable_learning_and_explicit_replay(self):
        self.args.learning_tasks = ["put_object_cabinet"]
        self.args.replay_tasks = ["click_bell"]
        self.args.new_ratio = .7
        cfg = launcher.build_plan(self.args)["config"]
        self.assertEqual(AutoLearningConfig(**cfg).replay_slots, 3)
        self.assertEqual(cfg["replay_task_names"], ["click_bell"])
        self.args.learning_tasks = ["typo"]
        with self.assertRaisesRegex(ValueError, "unknown tasks"):
            launcher.build_plan(self.args)

    def test_incomplete_checkpoint_and_normalization_fail_before_output(self):
        (self.model / "model-01.safetensors").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "truncated"):
            launcher.build_plan(self.args)
        self.write_shard()
        self.train.write_text('{"data": {}}')
        with self.assertRaisesRegex(ValueError, "norm_stats_file"):
            launcher.build_plan(self.args)
        self.assertFalse(self.args.out.exists())

    def test_duplicate_output_and_wrong_source_dtype_fail(self):
        self.write_shard("F32")
        with self.assertRaisesRegex(ValueError, "BF16"):
            launcher.build_plan(self.args)
        self.write_shard()
        self.args.out.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh output"):
            launcher.build_plan(self.args)


if __name__ == "__main__":
    unittest.main()

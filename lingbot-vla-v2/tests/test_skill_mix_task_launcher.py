"""实际启动器预检/命令生成；纯 CPU，无训练、无 torch。"""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from tokenizers import Tokenizer, models

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
        tokenizer = Tokenizer(models.BPE(
            vocab={"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5},
            merges=[("a", "b"), ("ab", "c"), ("b", "c")]))
        tokenizer.add_special_tokens(["[PAD]"])
        self.tokenizer = json.loads(tokenizer.to_str())
        (self.model / "tokenizer.json").write_text(tokenizer.to_str())
        self.write_shard()
        (self.model / "model.safetensors.index.json").write_text(json.dumps(
            dict(weight_map={"weight": "model-01.safetensors"})))
        self.norm = self.root / "norm.json"
        self.norm.write_text("{}")
        self.train = self.root / "source_run_config.json"
        self.train.write_text(json.dumps(dict(data=dict(norm_stats_file=str(self.norm)))))
        (self.root / "dataset/meta").mkdir(parents=True)
        (self.root / "Qwen3-VL").mkdir()
        (self.root / "Qwen3-VL/tokenizer.json").write_text(tokenizer.to_str())
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
            ("train.max_steps", "1000"), ("train.save_steps", "250"),
            ("train.hf_pass_interval", "0"), ("data.norm_stats_file", str(self.norm.resolve()))):
            self.assertEqual(cmd[cmd.index("--" + flag)+1], expected)
        self.assertIn(str(self.norm), plan["source_files"])
        self.assertIsNone(plan["training_config"]["model"]["vlm_repo_id"])
        self.assertTrue(plan["training_config"]["model"]["post_training"])

    def test_vision_encoder_freeze_is_explicit_and_can_be_disabled(self):
        self.args.freeze_vision_encoder = False
        self.args.gradient_checkpointing = True
        plan = launcher.build_plan(self.args)
        self.assertFalse(plan["training_config"]["train"]["freeze_vision_encoder"])
        self.assertTrue(plan["training_config"]["train"]["enable_gradient_checkpointing"])
        cmd = plan["command"]
        self.assertEqual(cmd[cmd.index("--train.freeze_vision_encoder") + 1], "false")

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

    def test_low_noise_and_static_config_propagate_without_scanning_in_preview(self):
        cfg=launcher.read_json(self.args.al_config)
        cfg.update(low_noise_fm_enabled=True,sample_scan_method='target_change_window',
                   target_change_cache_dir=str(self.root/'cache'),
                   target_change_embodiment_config=str(self.root/'actual_robot.yml'),
                   target_change_wrist_joint_names=[['left_wrist'],['right_wrist']])
        self.args.al_config_data=cfg
        plan=launcher.build_plan(self.args)
        self.assertTrue(plan['training_config']['train']['low_noise_fm_enabled'])
        self.assertEqual(plan['config']['sample_scan_method'],'target_change_window')
        self.assertEqual(plan['config']['target_change_dataset_root'],str(self.args.dataset_root.resolve()))
        self.assertFalse(self.args.out.exists())
        self.assertFalse((self.root/'cache').exists())
        # Source-run settings must not silently enable the experiment switch.
        self.args.al_config_data=launcher.read_json(self.args.al_config)
        self.train.write_text(json.dumps(dict(data=dict(norm_stats_file=str(self.norm)),
                                             train=dict(low_noise_fm_enabled=True))))
        self.assertFalse(launcher.build_plan(self.args)['training_config']['train']['low_noise_fm_enabled'])

    def test_static_preparation_is_before_exec_and_failure_leaves_no_run(self):
        cfg=launcher.read_json(self.args.al_config)
        cfg.update(sample_scan_method='target_change_window',
                   target_change_cache_dir=str(self.root/'cache'),
                   target_change_embodiment_config=str(self.root/'robot.yml'),
                   target_change_wrist_joint_names=[['l'],['r']])
        config_file=self.root/'al.json';config_file.write_text(json.dumps(cfg))
        argv=['--model',str(self.model),'--train-config',str(self.train),
              '--dataset-root',str(self.args.dataset_root),'--qwen3vl',str(self.args.qwen3vl),
              '--out',str(self.args.out),'--audit-report',str(self.args.audit_report),
              '--al-config',str(config_file),'--execute']
        target='experiment.robotwin.prepare_target_change_windows.prepare'
        with patch(target,side_effect=ValueError('bad static cache')), \
             patch.object(launcher.os,'execvpe') as execute, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'bad static cache'):
                launcher.main(argv)
            execute.assert_not_called()
        self.assertFalse(self.args.out.exists())
        order=[]
        with patch(target,side_effect=lambda *a,**k:order.append('prepare')) as prepare, \
             patch.object(launcher.os,'chdir'), \
             patch.object(launcher.os,'execvpe',side_effect=lambda *a:order.append('exec')), \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.main(argv)
        self.assertEqual(order,['prepare','exec'])
        self.assertEqual(prepare.call_args.args[3],['put_object_cabinet'])

    def test_continue_entry_runs_real_source_and_tokenizer_preflight(self):
        import shutil
        from test_continue_skill_training import entry
        source = self.root / "old_run"
        model = source / "checkpoints/global_step_1000/hf_ckpt"
        shutil.copytree(self.model, model)
        (source / "lingbotvla_cli.yaml").write_text(json.dumps(dict(
            data=dict(norm_stats_file=str(self.norm)),
            model=dict(tokenizer_path=str(self.args.qwen3vl)))))
        (source / "datasets.txt").write_text(f"robotwin {self.args.dataset_root}\n")
        (source / "al_config.json").write_text(json.dumps(dict(data_audit_file=str(self.args.audit_report))))
        args = NS(from_hf=model, qwen3vl=None, out=self.args.out,
            output_root=self.root, steps=500, micro=24, gas=1, shutdown=True,
            al_config=self.args.al_config, manifest=self.args.manifest,
            learning_tasks=None, master_port=62510, runtime_dir=None)
        args.run_name = "preflight_fixture"
        job, plan = entry.make_job(args)
        self.assertEqual(plan["config"]["batch_size"], 24)
        self.assertEqual(plan["config"]["max_global_steps"], 500)
        self.assertTrue(plan["config"]["hardness_scan_enabled"])
        self.assertEqual(plan["counts"]["put_object_cabinet"], dict(train=40, val=10))
        self.assertTrue(job["shutdown"])
        self.assertFalse(self.args.out.exists())

    def test_execute_worker_imports_experiment_before_installed_package(self):
        installed = self.root / "old_install"
        shadow = installed / "lingbotvla"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text("raise RuntimeError('imported old production package')\n")
        argv = ["--model", str(self.model), "--train-config", str(self.train),
                "--dataset-root", str(self.args.dataset_root), "--qwen3vl", str(self.args.qwen3vl),
                "--out", str(self.args.out), "--audit-report", str(self.args.audit_report), "--execute"]
        with patch.dict(os.environ, PYTHONPATH=str(installed), CUDA_VISIBLE_DEVICES="0",
                        AL_HARDNESS_CACHE_FILE="/tmp/old_hardness.json",
                        AL_HARDNESS_CACHE="/tmp/old_alias.json"), \
                patch.object(launcher.os, "chdir"), patch.object(launcher.os, "execvpe") as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            launcher.main(argv)
        env = execute.call_args.args[2]
        self.assertEqual(env["AL_HARDNESS_CACHE_FILE"], "")
        self.assertEqual(env["AL_HARDNESS_CACHE"], "")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [str(ROOT), str(installed)])
        # 真实 Python 子进程：脚本目录与包根不同，且存在竞争的旧包。只导入 AL，禁用 GPU。
        worker = self.root / "worker"
        worker.mkdir()
        probe = worker / "probe.py"
        probe.write_text(
            "import json\nfrom pathlib import Path\n"
            "import lingbotvla.auto_learning.config as module\n"
            f"cfg=module.AutoLearningConfig.from_dict(json.loads(Path({str(self.args.out / 'al_config.json')!r}).read_text()))\n"
            "assert not cfg.task_scan_enabled and not cfg.pass_transition_enabled and not cfg.import_pool_state\n"
            "print(Path(module.__file__).resolve())\n")
        actual = subprocess.check_output([sys.executable, str(probe)], cwd=self.root,
                    env=dict(env, CUDA_VISIBLE_DEVICES=""), text=True).strip()
        self.assertEqual(Path(actual), ROOT / "lingbotvla/auto_learning/config.py")

    def test_serialization_and_runtime_settings_are_not_semantic_mismatches(self):
        source = Tokenizer.from_str(json.dumps(self.tokenizer))
        source.enable_padding(length=72, pad_id=6, pad_token="[PAD]")
        source.enable_truncation(max_length=72)
        source_path = self.model / "tokenizer.json"
        source_path.write_text(source.to_str())
        original = source_path.read_bytes()
        selected = json.loads(json.dumps(self.tokenizer))
        selected["model"].pop("ignore_merges", None)
        selected["model"]["merges"] = [" ".join(pair) if isinstance(pair, list) else pair
                                              for pair in selected["model"]["merges"]]
        (self.args.qwen3vl / "tokenizer.json").write_text(json.dumps(selected))
        plan = launcher.build_plan(self.args)
        check = plan["tokenizer_validation"]
        self.assertEqual(len(check["normalized_sha256"]), 64)
        self.assertEqual(check["ignored_runtime_settings"][0]["truncation"]["max_length"], 72)
        self.assertIsNone(check["ignored_runtime_settings"][1]["truncation"])
        self.assertIn(str(source_path.resolve()), plan["source_files"])
        self.assertEqual(source_path.read_bytes(), original)
        self.assertFalse(self.args.out.exists())

    def test_tokenizer_semantic_changes_fail_before_output(self):
        for mutation in ("token_ids", "merges", "special_token", "pre_tokenizer", "ignore_merges"):
            with self.subTest(mutation=mutation):
                selected = json.loads(json.dumps(self.tokenizer))
                if mutation == "token_ids":
                    selected["model"]["vocab"].update(a=1, b=0)
                elif mutation == "merges":
                    selected["model"]["merges"][1:] = selected["model"]["merges"][1:][::-1]
                elif mutation == "special_token":
                    selected["added_tokens"][0]["special"] = False
                elif mutation == "pre_tokenizer":
                    selected["pre_tokenizer"] = {"type": "Whitespace"}
                else:
                    selected["model"]["ignore_merges"] = True
                (self.args.qwen3vl / "tokenizer.json").write_text(json.dumps(selected))
                with self.assertRaisesRegex(ValueError, "tokenizer disagrees"):
                    launcher.build_plan(self.args)
                self.assertFalse(self.args.out.exists())

    def test_invalid_tokenizer_is_not_treated_as_compatible(self):
        (self.args.qwen3vl / "tokenizer.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "cannot validate tokenizer"):
            launcher.build_plan(self.args)
        self.assertFalse(self.args.out.exists())

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

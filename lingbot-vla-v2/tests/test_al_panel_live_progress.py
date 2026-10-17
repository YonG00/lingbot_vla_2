"""AL面板实时训练器/进度来源契约，不访问训练机或GPU。"""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PANEL_PATH = Path(__file__).resolve().parents[1] / "tools/auto_learning_pipeline/al_panel.py"
spec = importlib.util.spec_from_file_location("al_panel_live_progress", PANEL_PATH)
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


class TrainerDiscoveryTests(unittest.TestCase):
    def _discover(self, launcher):
        cmdlines = {"100": launcher, "101": "/python -u tasks/vla/train_lingbotvla.py"}
        with (patch.object(panel, "_pids", return_value=["100", "101"]),
              patch.object(panel, "_ppid", side_effect=lambda pid: {"100": "1", "101": "100"}[pid]),
              patch.object(panel, "_cmdline", side_effect=cmdlines.get),
              patch.object(panel, "_environ", return_value={
                  "TRAIN_OUT": "/data/outputs/al_future_formal",
                  "WORLD_SIZE": "1", "CUDA_VISIBLE_DEVICES": "0",
              }),
              patch.object(panel, "_start_ticks", return_value=42)):
            return panel.discover_trainers()

    def test_detects_torchrun_console_script_for_future_run_name(self):
        trainers = self._discover("/opt/venv/bin/torchrun\0--nproc-per-node\0" + "1")
        self.assertEqual(len(trainers), 1)
        self.assertEqual(trainers[0]["run"], "al_future_formal")
        self.assertEqual(panel.find_trainer(trainers, "al_future_formal"), trainers[0])

    def test_detects_python_module_launcher(self):
        trainers = self._discover("/opt/venv/bin/python\0-m\0torch.distributed.run\0train.py")
        self.assertEqual(len(trainers), 1)
        self.assertEqual(trainers[0]["run"], "al_future_formal")

    def test_rejects_unrelated_process_command(self):
        self.assertFalse(panel._is_torchrun_cmdline("/usr/bin/python\0train.py"))

    def test_output_argument_identifies_run_without_legacy_environment(self):
        for option in ('--train.output_dir\0', '--train.output_dir='):
            with self.subTest(option=option), patch.object(panel, '_pids', return_value=['100', '101']), \
                 patch.object(panel, '_ppid', side_effect=lambda p: '100' if p == '101' else '1'), \
                 patch.object(panel, '_cmdline', side_effect=lambda p: (
                     '/python\0-m\0torch.distributed.run' if p == '100' else
                     '/python\0train.py\0' + option + '/data/outputs/al_skill_m24')), \
                 patch.object(panel, '_environ', return_value={'WORLD_SIZE': '1'}), \
                 patch.object(panel, '_start_ticks', return_value=1):
                trainers = panel.discover_trainers()
                self.assertEqual(trainers[0]['run'], 'al_skill_m24')
                self.assertIsNone(panel.find_trainer(trainers, 'al_skill_m2'))


class SkillPanelTests(unittest.TestCase):
    def test_json_config_and_selection_show_before_first_unit_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rd = root / 'al_skill'
            rd.mkdir()
            al = rd / 'al_config.json'
            al.write_text(json.dumps(dict(max_global_steps=2000, eval_interval_steps=100,
                pass_transition_enabled=False, task_scan_enabled=False, hardness_scan_enabled=False)))
            (rd / 'lingbotvla_cli.yaml').write_text(
                f'train:\n  auto_learning: {al}\n  max_steps: 1000\n  save_steps: 250\n')
            (rd / 'auto_learning_events.jsonl').write_text(json.dumps(
                dict(action='select', task='put_object_cabinet', step=0)) + '\n')
            logs = root / 'logs'
            logs.mkdir()
            (logs / 'train_al_skill.log').write_text('Step: 69/1000 [03:27<46:33, 3.00s/it]\r'
                                                   'Step: 70/1000 [03:30<46:30, 3.00s/it]')
            with patch.object(panel, 'ROOT', root), \
                 patch.object(panel, 'discover_trainers', return_value=[dict(run='al_skill', ranks=1, world_size=1, gpus='0')]), \
                 patch.object(panel, 'read_tb_latest', return_value={}), \
                 patch('sys.argv', ['al_panel.py', 'al_skill']), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                cfg = panel.run_cfg('al_skill')
                self.assertEqual(cfg['max_steps'], 1000)
                self.assertFalse(cfg['pass_enabled'])
                self.assertEqual(panel.main(), 0)
                with patch('sys.argv', ['al_panel.py', 'al_skill', '--wide']):
                    self.assertEqual(panel.main(), 0)
            text = output.getvalue()
            for wanted in ('Step 70/1000', 'PASS关闭', 'put_object_cabinet', '剩余约 0.8h',
                           '任务扫描=关', '样本扫描=关', '评测每100步', '存档每250步'):
                self.assertIn(wanted, text)
            for unwanted in ('无进程!', '预扫/初始化阶段', '16.6h', '第一份 DCP'):
                self.assertNotIn(unwanted, text)

    def test_legacy_yaml_keeps_goal_and_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rd = root / 'train_al_v99'
            rd.mkdir()
            al = rd / 'old.yaml'
            al.write_text('target_total_passed_tasks: 50\neval_interval_steps: 50\nmax_global_steps: 20000\n')
            (rd / 'lingbotvla_cli.yaml').write_text(f'train:\n  auto_learning: {al}\n')
            with patch.object(panel, 'ROOT', root):
                cfg = panel.run_cfg('al_v99')
            self.assertTrue(cfg['pass_enabled'])
            self.assertEqual((cfg['target'], cfg['interval'], cfg['max_steps']), (50, 50, 20000))

    def test_selection_beats_last_completed_task(self):
        with tempfile.TemporaryDirectory() as temp:
            events = Path(temp) / 'events.jsonl'
            events.write_text('\n'.join(json.dumps(x) for x in [
                dict(name='curriculum/current_task_name', value='old'),
                dict(action='select', task='new')]))
            last, _ = panel.read_events(events)
            self.assertEqual(last['curriculum/current_task_name'], 'new')


class ProgressSourceTests(unittest.TestCase):
    def test_newer_tensorboard_step_and_steptime_win(self):
        log = (6809, 2.4, 80, 3.0, 1000.0, 2000.0)
        tb = {"training/loss": (6850, 0.04), "steptime": (6850, 8.7)}
        result = panel.prefer_tensorboard_progress(log, tb)
        self.assertEqual(result, (6850, 2.4, 80, 8.7, 1000.0, 2000.0))

    def test_tensorboard_supplies_progress_when_log_missing(self):
        log = (None, None, 0, None, None, None)
        tb = {"training/loss": (6850, 0.04), "steptime": (6850, 8.7)}
        self.assertEqual(panel.prefer_tensorboard_progress(log, tb),
                         (6850, 8.7, 0, 8.7, None, None))

    def test_newer_log_step_wins_over_stale_tensorboard(self):
        log = (6860, 2.4, 80, 2.8, 1000.0, 2000.0)
        tb = {"training/loss": (6850, 0.04), "steptime": (6850, 8.7)}
        self.assertEqual(panel.prefer_tensorboard_progress(log, tb), log)

    def test_missing_tensorboard_keeps_log_progress(self):
        log = (6860, 2.4, 80, 2.8, 1000.0, 2000.0)
        self.assertEqual(panel.prefer_tensorboard_progress(log, {}), log)

    def test_invalid_tensorboard_step_is_ignored(self):
        log = (6860, 2.4, 80, 2.8, 1000.0, 2000.0)
        for point in ((None, 0.04), (float("nan"), 0.04), "bad"):
            with self.subTest(point=point):
                self.assertEqual(panel.prefer_tensorboard_progress(
                    log, {"training/loss": point}), log)

    def test_default_log_paths_cover_panel_and_start_script_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "outputs"
            log_root = Path(temp) / "tmp"
            root.mkdir()
            log_root.mkdir()
            panel.ROOT = root
            (log_root / "al_future_formal.log").write_text("training")
            with patch.dict("os.environ", {
                "AL_LOG_ROOT": str(log_root), "AL_LOG_PATH": ""
            }, clear=False):
                self.assertEqual(panel.train_log_path("al_future_formal"),
                                 log_root / "al_future_formal.log")
            expected = root / "logs" / "train_al_future_formal.log"
            self.assertEqual(panel.train_log_path("al_future_formal"), expected)

    def test_explicit_log_path_precedes_newer_fallback_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "outputs"
            log_root = Path(temp) / "tmp"
            root_logs = root / "logs"
            root_logs.mkdir(parents=True)
            log_root.mkdir()
            explicit = Path(temp) / "chosen.log"
            explicit.write_text("explicit")
            fallback = log_root / "al_future_formal.log"
            fallback.write_text("fallback")
            os.utime(explicit, (1, 1))
            panel.ROOT = root
            with patch.dict("os.environ", {
                "AL_LOG_PATH": str(explicit), "AL_LOG_ROOT": str(log_root)
            }, clear=False):
                self.assertEqual(panel.train_log_path("al_future_formal"), explicit)


if __name__ == "__main__":
    unittest.main()

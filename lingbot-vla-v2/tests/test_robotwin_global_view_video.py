"""纯 CPU：真实录像时钟/编码 + 实际客户端函数契约，不导入仿真或 torch。"""
from __future__ import annotations

import ast
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'lingbot-vla-v2'
HELPER = ROOT / 'RoboTwin-lingbot/envs/utils/global_view_video.py'
CLIENT = PACKAGE / 'experiment/robotwin/eval_policy_client_lingbotvla.py'
BASE = ROOT / 'RoboTwin-lingbot/envs/_base_task.py'
LAUNCHER = PACKAGE / 'experiment/robotwin/start_robotwin_infer_and_eval.sh'
spec = importlib.util.spec_from_file_location('global_view_video', HELPER)
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


class Frame:
    shape = (4, 4, 3)
    dtype = 'uint8'

    def __init__(self, value=0):
        self.value = value

    def tobytes(self):
        return bytes([self.value % 256, 64, 128]) * 16


class Pipe:
    def __init__(self):
        self.payload = bytearray()
        self.closed = False

    def write(self, value):
        self.payload.extend(value)

    def close(self):
        self.closed = True


class Encoder:
    def __init__(self):
        self.stdin = Pipe()
        self.waits = 0
        self.killed = False

    def wait(self, timeout):
        self.waits += 1
        return 0

    def kill(self):
        self.killed = True


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class RecorderTests(unittest.TestCase):
    def make_recorder(self, tmp, **kwargs):
        frames = []
        def capture():
            frames.append(len(frames))
            return Frame(len(frames))
        rec = video.GlobalViewRecorder(Path(tmp) / 'episode0_global.mp4', capture,
                                       timestep=1/250, width=4, height=4, **kwargs)
        return rec, frames

    def test_250hz_to_60fps_without_drift_or_duplicate_capture(self):
        for fps in (1, 30, 60, 120):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as tmp:
                rec, frames = self.make_recorder(tmp, fps=fps)
                encoder = Encoder()
                state = random.getstate()
                with patch.object(video.subprocess, 'Popen', return_value=encoder), redirect_stdout(io.StringIO()):
                    rec.start()
                    for _ in range(2500):
                        rec.after_step()
                    self.assertEqual(len(frames), 1 + fps * 10)
                    rec.close('success')
                self.assertEqual(random.getstate(), state)
                meta = json.loads(rec.path.with_suffix('.json').read_text())
                self.assertEqual(meta['simulation_seconds'], 10)
                self.assertAlmostEqual(meta['video_seconds'], 10 + 1/fps)
                self.assertEqual(meta['result'], 'success')
                self.assertTrue(meta['encoder_ok'])

    def test_terminal_pose_between_scheduled_frames_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, frames = self.make_recorder(tmp)
            with patch.object(video.subprocess, 'Popen', return_value=Encoder()), redirect_stdout(io.StringIO()):
                rec.start()
                for _ in range(6):
                    rec.after_step()
                self.assertEqual(len(frames), 2)  # t=0, tick5
                rec.close('success')
                self.assertEqual(len(frames), 3)  # terminal tick6
                rec.close()
                self.assertEqual(len(frames), 3)

    def test_initial_preview_and_encoder_cleanup_on_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            previews = []
            rec, _ = self.make_recorder(tmp, preview_writer=lambda frame, path: previews.append(path))
            encoder = Encoder()
            with patch.object(video.subprocess, 'Popen', return_value=encoder), redirect_stdout(io.StringIO()):
                rec.start()
                rec.close()
                rec.close()
            self.assertEqual(previews, [rec.path.with_suffix('.preview.png')])
            self.assertTrue(encoder.stdin.closed)
            self.assertEqual(encoder.waits, 1)
            self.assertEqual(json.loads(rec.path.with_suffix('.json').read_text())['result'], 'interrupted')

    def test_invalid_fps_dimensions_and_clock_rejected(self):
        for kwargs in ({'fps': 0}, {'fps': 121}, {'fps': 2.5}, {'fps': True},
                       {'fps': 60, 'timestep': 1/30}, {'timestep': float('nan')},
                       {'width': 3}, {'height': -2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                options = dict(timestep=1/250, width=4, height=4)
                options.update(kwargs)
                video.GlobalViewRecorder('/tmp/unused.mp4', Frame, **options)

    def test_encoder_failure_is_reported_and_resources_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, _ = self.make_recorder(tmp)
            encoder = Encoder()
            encoder.wait = lambda timeout: 7
            with patch.object(video.subprocess, 'Popen', return_value=encoder), redirect_stdout(io.StringIO()):
                rec.start()
                with self.assertRaisesRegex(RuntimeError, '退出码 7'):
                    rec.close('failure')
            self.assertTrue(encoder.stdin.closed)
            self.assertTrue(rec.error_stream.closed)
            self.assertFalse(json.loads(rec.path.with_suffix('.json').read_text())['encoder_ok'])

    def test_encoder_timeout_is_bounded_and_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, _ = self.make_recorder(tmp)
            encoder = Encoder()
            def wait(timeout):
                if not encoder.killed:
                    raise subprocess.TimeoutExpired('ffmpeg', timeout)
                return -9
            encoder.wait = wait
            with patch.object(video.subprocess, 'Popen', return_value=encoder), redirect_stdout(io.StringIO()):
                rec.start()
                with self.assertRaisesRegex(RuntimeError, '退出码 -1'):
                    rec.close()
            self.assertTrue(encoder.killed)
            self.assertTrue(rec.error_stream.closed)

    def test_look_at_basis_is_orthonormal(self):
        matrix = video.look_at_matrix((1.2, -1.2, 1.8), (0, 0, 0.85))
        columns = [[matrix[r][c] for r in range(3)] for c in range(3)]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(sum(a*b for a,b in zip(columns[i], columns[j])), float(i == j))
        for target in ((1.2, -1.2, 1.8), (1.2, -1.2, 2.8), (float('nan'), 0, 0)):
            with self.assertRaises(ValueError):
                video.look_at_matrix((1.2, -1.2, 1.8), target)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), '需要 ffmpeg/ffprobe')
    def test_real_h264_file_has_60fps_and_expected_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, _ = self.make_recorder(tmp)
            with redirect_stdout(io.StringIO()):
                rec.start()
                try:
                    for _ in range(250):
                        rec.after_step()
                finally:
                    rec.close('success')
            result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                     '-show_entries', 'stream=codec_name,nb_frames,r_frame_rate,width,height,duration',
                                     '-of', 'json', str(rec.path)], check=True, capture_output=True, text=True)
            stream = json.loads(result.stdout)['streams'][0]
            self.assertEqual(stream['codec_name'], 'h264')
            self.assertEqual(stream['r_frame_rate'], '60/1')
            self.assertEqual(int(stream['nb_frames']), 61)
            self.assertAlmostEqual(float(stream['duration']), 61/60, places=5)


class ClientContractTests(unittest.TestCase):
    def run_client(self, *, head=False, global_view=False, failure=False, startup_failure=False, success=True):
        class Env:
            plan_success = True
            render_freq = 0
            step_lim = 1
            test_num = 0
            starts = 0
            head_closed = 0
            results = []
            _eval_global_recorder = None

            def setup_demo(self, **kwargs):
                self.take_action_cnt = 0
                self.eval_success = False
                self.eval_video_path = '/tmp/unused' if head else None

            def play_once(self): return {'info': {}}
            def check_success(self): return True
            def close_env(self, **kwargs): pass
            def set_instruction(self, **kwargs): pass
            def get_instruction(self): return 'test'
            def get_obs(self):
                return {'observation': {name: {'rgb': name} for name in ('head_camera', 'left_camera', 'right_camera')},
                        'joint_action': {'vector': [0]}}
            def _set_eval_video_ffmpeg(self, encoder): pass
            def _del_eval_video_ffmpeg(self): self.head_closed += 1
            def _start_eval_global_video(self, path, **kwargs):
                self.starts += 1
                self._eval_global_recorder = SimpleNamespace(close=lambda result: self.results.append(result))
                if startup_failure:
                    raise RuntimeError('startup failure')
            def take_action(self, action):
                self.take_action_cnt += 1
                self.eval_success = success

        Env._stop_eval_global_video = load_function(BASE, '_stop_eval_global_video', {}, 'Base_Task')
        observations = []
        class Model:
            def infer(self, payload):
                observations.append(payload)
                if failure:
                    raise RuntimeError('inference failure')
                return {'action': SimpleNamespace(shape=(14,)), 'server_timing': 0}

        function = load_function(CLIENT, 'eval_policy', dict(
            ExitStack=ExitStack, Path=Path, os=os, subprocess=SimpleNamespace(Popen=lambda *a, **k: Encoder(), PIPE=-1),
            np=SimpleNamespace(random=SimpleNamespace(choice=lambda x: x[0])), UnStableError=RuntimeError,
            generate_episode_descriptions=lambda *a: [{'unseen': ['test']}]))
        env = Env()
        args = dict(task_name='put_object_cabinet', policy_name='ACT', clear_cache_freq=5, render_freq=0,
                    task_config='demo_clean', ckpt_setting='test', eval_global_view_log=global_view,
                    global_view_fps=60, eval_global_view_save_dir='/tmp/unused')
        with redirect_stdout(io.StringIO()):
            if failure or startup_failure:
                with self.assertRaisesRegex(RuntimeError, 'failure'):
                    function('test', env, args, Model(), 0, test_num=1, instruction_type='unseen', usr_args={'robo_name': 'test'})
            else:
                function('test', env, args, Model(), 0, test_num=1, instruction_type='unseen', usr_args={'robo_name': 'test'})
        return env, observations

    def test_head_global_four_combinations_and_model_payload(self):
        reference = None
        for head in (False, True):
            for enabled in (False, True):
                with self.subTest(head=head, global_view=enabled):
                    env, payload = self.run_client(head=head, global_view=enabled)
                    self.assertEqual(env.starts, int(enabled))
                    self.assertEqual(env.head_closed, int(head))
                    self.assertEqual(env.results, ['success'] if enabled else [])
                    if reference is None:
                        reference = payload
                    self.assertEqual(payload, reference)

    def test_inference_and_startup_exceptions_close_both_recorders(self):
        for kwargs in ({'failure': True}, {'startup_failure': True}):
            with self.subTest(kwargs=kwargs):
                env, _ = self.run_client(head=True, global_view=True, **kwargs)
                self.assertEqual(env.results, ['interrupted'])
                self.assertEqual(env.head_closed, 1)

    def test_unsuccessful_episode_records_failure(self):
        env, _ = self.run_client(global_view=True, success=False)
        self.assertEqual(env.results, ['failure'])

    def test_capture_is_inside_physics_loop_before_success_return(self):
        tree = ast.parse(BASE.read_text())
        task = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Base_Task')
        action = next(n for n in task.body if isinstance(n, ast.FunctionDef) and n.name == 'take_action')
        loop = next(n for n in ast.walk(action) if isinstance(n, (ast.For, ast.While))
                    and any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'self.scene.step' for c in ast.walk(n)))
        code = ast.unparse(loop)
        self.assertLess(code.index('self.scene.step()'), code.index('self._eval_global_recorder.after_step()'))
        self.assertLess(code.index('self._eval_global_recorder.after_step()'), code.index('self.check_success()'))
        start = next(n for n in task.body if isinstance(n, ast.FunctionDef) and n.name == '_start_eval_global_video')
        capture = next(n for n in ast.walk(start) if isinstance(n, ast.FunctionDef) and n.name == 'capture_rgb')
        self.assertNotIn('self._update_render()', ast.unparse(capture))
        self.assertNotIn('self.get_obs()', ast.unparse(capture))


class LauncherContractTests(unittest.TestCase):
    def test_real_shell_argument_parser_and_range_guards(self):
        parser = LAUNCHER.read_text().split('# ===== Common environment =====')[0]
        script = parser + '\nprintf "%s %s" "$global_view" "$global_view_fps"\n'
        for args, expected in (([], 'False 60'), (['--global_view', '--global_view_fps', '30'], 'True 30'),
                               (['--global_view', '--no_video'], 'True 60')):
            result = subprocess.run(['bash', '-c', script, 'parser', *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)
        for value in ('0', '121', '-1', '60.0', 'abc', '10000000000000000000000'):
            result = subprocess.run(['bash', '-c', script, 'parser', '--global_view_fps', value], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
        result = subprocess.run(['bash', '-c', script, 'parser', '--global_view_fps'],
                                capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_launcher_passes_flags_to_actual_client_and_guards_stale_sim(self):
        source = LAUNCHER.read_text()
        self.assertIn('--eval_global_view_log ${global_view}', source)
        self.assertIn('--global_view_fps ${global_view_fps}', source)
        self.assertIn('experiment/robotwin/eval_policy_client_lingbotvla.py', source)
        self.assertLess(source.index('requires updated RoboTwin'), source.index('# Phase 1:'))


if __name__ == '__main__':
    unittest.main()

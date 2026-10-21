import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiment.official_control import evaluate
from experiment.official_control.recording import video_count,install_recording_limit


class RecordingTests(unittest.TestCase):
    def test_first_two_recorders_do_not_change_episode_count(self):
        class Task:
            test_num=0
            def _start_eval_global_video(self,path):self.paths.append(path)
        install_recording_limit(Task,2)
        task=Task();task.paths=[]
        for i in range(3):
            task.test_num=i;task._start_eval_global_video(str(i))
        self.assertEqual(task.paths,['0','1'])
        self.assertEqual(video_count(3,2),2)
        self.assertEqual(video_count(1,2),1)
        self.assertEqual(video_count(3),3)
        self.assertEqual(video_count(3,0),0)
        for bad in [-1,True,1.5]:
            with self.assertRaises(ValueError):video_count(3,bad)

    def test_three_rollout_results_with_only_two_videos_pass_real_summary_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);robo=root/'robo';(robo/'script').mkdir(parents=True)
            (robo/'script/eval_policy_client_lingbotvla.py').write_text('# unchanged client')
            dest=root/'evaluation';dest.mkdir();audit=root/'precision.json'
            audit.write_text(json.dumps(dict(parameter_numel_by_dtype={'torch.float32':1},forward_samples={'projection':{}})))
            commands=[]
            def launch(command,**kwargs):
                commands.append(command)
                results=dest/'results'/'a';results.mkdir(parents=True)
                for i in range(2):(results/f'episode{i}_global.mp4').write_bytes(b'video')
                (results/'_result.txt').write_text(str(2/3))
                kwargs['stdout'].write('OIDN 2.5.1 CUDA=True\nSuccess rate: 1/1 current seed: 100000\nSuccess rate: 1/2 current seed: 100001\nSuccess rate: 2/3 current seed: 100002\n')
                return SimpleNamespace(wait=lambda **kwargs:0,poll=lambda:0)
            job=dict(task='a',label='probe',episodes=3,video_episodes=2,model='/m',destination=str(dest),
                robotwin=str(robo),qwen='/q',sim_python='/python',require_baseline=False)
            shared=dict(model='/m',process=SimpleNamespace(pid=42,poll=lambda:None),port=123,audit=audit)
            with patch.object(evaluate.subprocess,'Popen',side_effect=launch),patch.object(evaluate.subprocess,'check_output',return_value='/site'),patch.object(evaluate.urllib.request,'urlopen',return_value=contextlib.nullcontext(SimpleNamespace(status=200))),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.execute(job,shared=shared),0)
            summary=json.loads((dest/'summary.json').read_text())
            self.assertEqual(summary['episodes'],3);self.assertEqual(summary['recorded_episodes'],2)
            self.assertEqual(summary['successes'],2)
            command=commands[0]
            self.assertEqual(command[command.index('--test_num')+1],'3')
            self.assertEqual(command[command.index('--global_view_episodes')+1],'2')


if __name__=='__main__':unittest.main()

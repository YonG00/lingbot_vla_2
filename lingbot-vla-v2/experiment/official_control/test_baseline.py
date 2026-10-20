"""基准快照、重新准备和HF精度/完整性契约；仅标准库。"""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from experiment.official_control import bf16_trial as trial
from experiment.official_control.baseline_pipeline import verify_checkpoint

BASELINE=Path(trial.__file__).parent/'baselines/adjust_bottle_bf16_200.json'

class BaselineTests(unittest.TestCase):
    def test_recorded_evidence_and_resource_invariants(self):
        b=json.loads(BASELINE.read_text())
        t=b['trainer_config']['train']
        self.assertEqual((t['micro_batch_size'],t['global_batch_size'],t['gradient_accumulation_steps']), (56,56,1))
        self.assertEqual((t['max_steps'],t['save_steps'],t['hf_export_dtype']), (200,250,'bf16'))
        self.assertFalse(any(t[k] for k in ['enable_mixed_precision','enable_fp32','action_fp32','ee_fk_loss_enabled','low_noise_fm_enabled','auto_learning']))
        self.assertEqual(b['evaluation']['successes'],3)
        self.assertEqual(b['evaluation']['seeds'],[100001,100002,100005])
        self.assertEqual(b['episode_ids'],list(range(50)))

    def test_prepare_reproduces_snapshot_without_old_run(self):
        b=json.loads(BASELINE.read_text())
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'new'
            with patch('shutil.disk_usage') as disk, patch('experiment.official_control.prepare_adjust_bottle.validate_identity'):
                disk.return_value.free=100*2**30
                trial.prepare(None,out)
                with self.assertRaises(FileExistsError): trial.prepare(None,out)
            actual=json.loads((out/'trainer_config.json').read_text())
            expected=copy.deepcopy(b['trainer_config'])
            expected['data'].update(train_path=str(out/'datasets.txt'),episode_ids_file=str(out/'train_ids.json'),robot_config_root=str(trial.ROOT/'configs/robot_configs'))
            expected['train']['output_dir']=str(out)
            self.assertEqual(actual,expected)
            self.assertEqual(json.loads((out/'train_ids.json').read_text()),b['episode_ids'])
            self.assertIn('global_step_9680',actual['model']['model_path'])

    def test_hf_headers_reject_wrong_precision_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'weight':'part.safetensors'})))
            def write(dtype,end=4):
                h=json.dumps(dict(weight=dict(dtype=dtype,shape=[2],data_offsets=[0,end]))).encode()
                (root/'part.safetensors').write_bytes(struct.pack('<Q',len(h))+h+b'\0'*4)
            write('BF16')
            self.assertEqual(verify_checkpoint(root)['tensors'],1)
            write('F32')
            with self.assertRaises(AssertionError): verify_checkpoint(root)
            write('BF16',100)
            with self.assertRaises(AssertionError): verify_checkpoint(root)

    def test_default_pipeline_only_prints_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'absent'
            script=Path(trial.__file__).with_name('baseline_pipeline.py')
            result=subprocess.run([sys.executable,str(script),'--out',str(out)],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout)['final_checkpoint'],200)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_name(out.name+'_pipeline').exists())

if __name__=='__main__': unittest.main()

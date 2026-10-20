import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiment.official_control import single_task_trial as trial

class SingleTaskTests(unittest.TestCase):
    def test_only_task_and_source_budget_paths_change(self):
        baseline=json.loads((Path(trial.__file__).parent/'baselines/adjust_bottle_bf16_200.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir();out=Path(d)/'new'
            (source/'trainer_config.json').write_text(json.dumps(baseline['trainer_config']))
            (source/'datasets.txt').write_text(baseline['datasets'])
            with patch('experiment.official_control.baseline_pipeline.verify_checkpoint'),patch('experiment.official_control.prepare_adjust_bottle.validate_identity'),patch('shutil.disk_usage') as disk:
                disk.return_value.free=100*2**30
                trial.prepare(out,source,'place_container_plate')
            cfg=json.loads((out/'trainer_config.json').read_text())
            plan=json.loads((out/'plan.json').read_text())
            self.assertEqual((plan['task'],plan['steps'],plan['expected_frames']),('place_container_plate',200,7934))
            self.assertEqual(cfg['train']['save_steps'],200)
            self.assertEqual(json.loads((out/'train_ids.json').read_text()),list(range(1250,1300)))
            self.assertEqual(cfg['model']['model_path'],str(source/'checkpoints/global_step_200/hf_ckpt'))
            before=baseline['trainer_config']['train']
            for key,value in cfg['train'].items():
                if key not in ['output_dir','max_steps','save_steps','enable_resume','load_checkpoint_path','step_offset','reset_step_counter']:
                    self.assertEqual(value,before[key],key)

if __name__=='__main__':unittest.main()

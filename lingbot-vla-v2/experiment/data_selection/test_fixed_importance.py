import unittest,json,tempfile,hashlib
from pathlib import Path
from experiment.data_selection.fixed_importance import fixed_policies
from experiment.data_selection.inheritance import freeze_registry
from experiment.data_selection.weights import SelectionWeights
class FixedImportanceTests(unittest.TestCase):
 def fixture(self,p):
  folder=p/'configs/data_selection';folder.mkdir(parents=True);payload=json.dumps([dict(episode_index=1450,frame_index=2)]).encode();(p/'selected.json').write_bytes(payload)
  cfg=dict(version=1,task='stack_bowls_two',dataset_name='robotwin',dataset_root=str(p/'dataset'),normalization='sample_count',multiplier=2,selected_frames='selected.json',selected_frames_sha256=hashlib.sha256(payload).hexdigest());(p/'fixed.json').write_text(json.dumps(cfg));(folder/'fixed_task_importance.json').write_text(json.dumps(dict(version=1,policies={'stack_bowls_two':'fixed.json'})))
 def test_replay_task_uses_fixed_two_without_source_registry(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.fixture(p);fixed=fixed_policies(p,{'move_stapler_pad':[150],'stack_bowls_two':[1450]},p/'dataset');out=p/'run';out.mkdir();registry=freeze_registry(p/'empty',out,fixed=fixed);policy=SelectionWeights(registry)
   self.assertEqual(policy.weight('robotwin',1450,2),2);self.assertEqual(policy.weight('robotwin',1450,3),1);self.assertEqual(policy.weight('robotwin',150,2),1)
 def test_unrelated_task_does_not_inject_fixed_task(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.fixture(p);self.assertEqual(fixed_policies(p,{'move_stapler_pad':[]},p/'dataset'),{})
 def test_wrong_dataset_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.fixture(p)
   with self.assertRaises(ValueError):fixed_policies(p,{'stack_bowls_two':[]},p/'other')

 def test_conflicting_explicit_task_weight_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);self.fixture(p);cfg=json.loads((p/'fixed.json').read_text());cfg['multiplier']=3;(p/'override.json').write_text(json.dumps(cfg));out=p/'out';out.mkdir()
   with self.assertRaisesRegex(ValueError,'conflicts with fixed'):
    freeze_registry(p/'empty',out,explicit=p/'override.json',fixed={'stack_bowls_two':p/'fixed.json'})

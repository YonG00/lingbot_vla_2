import unittest,json,tempfile,hashlib
from pathlib import Path
import torch
from experiment.task_recipes.runtime import StreamDataset
from experiment.data_selection.weights import SelectionWeights
class RoleLossContracts(unittest.TestCase):
 def test_same_frame_has_occurrence_weight_without_mutation(self):
  original={'actions':torch.ones(2),'sample_loss_weight':torch.tensor(3.)}
  dataset=StreamDataset([original],100)
  a=dataset[(0,3.,'left_grasp')];b=dataset[(0,2.,'second_bowl_place')]
  self.assertEqual(a['sample_loss_weight'].item(),3.)
  self.assertEqual(b['sample_loss_weight'].item(),2.)
  self.assertEqual(original['sample_loss_weight'].item(),3.)
  self.assertEqual(a['sample_loss_weight'].dtype,torch.float32)
 def test_coordinate_fallback_keeps_left_three(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);rows=[dict(episode_index=1,frame_index=0,loss_weight=3),dict(episode_index=1,frame_index=1,loss_weight=2)]
   payload=json.dumps(rows).encode();(p/'selected.json').write_bytes(payload)
   config=dict(version=1,normalization='sample_count',dataset_name='robotwin',dataset_root=tmp,multiplier=2,use_selected_frame_weights=True,selected_frames='selected.json',selected_frames_sha256=hashlib.sha256(payload).hexdigest())
   (p/'weight.json').write_text(json.dumps(config));policy=SelectionWeights(p/'weight.json')
   self.assertEqual(policy.weight('robotwin',1,0),3);self.assertEqual(policy.weight('robotwin',1,1),2);self.assertEqual(policy.weight('robotwin',1,2),1)

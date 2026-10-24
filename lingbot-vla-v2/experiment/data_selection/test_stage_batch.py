import tempfile,unittest,json,hashlib
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
import torch
from .test_full_stages import fixture
from .stage_batch import build_stage_fields
from .weights import SelectionWeights
class StageBatchTests(unittest.TestCase):
 def transform(self):
  info=SimpleNamespace(joints=['arm.position','end.position','effector.position'],joints_max_dim={'arm.position':14,'end.position':14,'effector.position':2})
  mapping={'action.effector.position':dict(origin_keys=OrderedDict([('action',dict(start=6,end=7)),('action*',dict(start=13,end=14))])),'action.arm.position':dict(origin_keys=OrderedDict([('action',dict(start=0,end=6)),('action*',dict(start=7,end=13))]))}
  return SimpleNamespace(feature_config=info,key_mapping=mapping,actions_convert_from_state=set())
 def entry(self,padding=None):
  active=torch.zeros(30,dtype=torch.bool);active[[*range(12),28,29]]=True
  return dict(actions=torch.zeros(50,30),action_joint_mask=active,action_is_pad=torch.tensor(padding or [False]*50))
 def test_actual_post_transform_stage_fields_are_per_time_and_arm(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);fixture(p);policy=SelectionWeights(p/'weight_config.json');v=build_stage_fields(policy,'robotwin',7,100,self.entry(),self.transform());self.assertEqual(v['stage_fm_weight'][20][0],5);self.assertEqual(v['stage_fm_weight'][20][6],1);self.assertEqual(v['stage_arm_index'][28],0);self.assertEqual(v['stage_arm_index'][29],1)
 def test_dataset_padding_disagreement_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);fixture(p);policy=SelectionWeights(p/'weight_config.json')
   with self.assertRaises(ValueError):build_stage_fields(policy,'robotwin',7,198,self.entry(),self.transform())
 def test_legacy_replay_is_folded_once_without_stage_labels(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);fixture(p);policy=SelectionWeights(p/'weight_config.json');policy.coordinate_weights[(8,0)]=2.
   v=build_stage_fields(policy,'robotwin',8,0,self.entry([False]*40+[True]*10),self.transform());self.assertEqual(v['stage_fm_weight'][0][0],2);self.assertEqual(v['stage_fm_weight'][40][0],0);self.assertEqual(v['stage_ids'][0],[-1,-1]);self.assertEqual(v['stage_current_depth_weight'],2);self.assertEqual(v['stage_future_depth_weight'],0)

 def test_scalar_weight_sum_without_any_stage_table(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p)
   for key in ['stage_table','stage_table_sha256','stage_policy','stage_policy_sha256']:cfg.pop(key)
   cfg.update(normalization='weight_sum',multiplier=1,use_selected_frame_weights=True)
   selected=[dict(episode_index=8,frame_index=f,loss_weight=w)for f,w in enumerate([20,15,1])]
   (p/'selected.json').write_text(json.dumps(selected));cfg['selected_frames_sha256']=hashlib.sha256((p/'selected.json').read_bytes()).hexdigest();(p/'weight_config.json').write_text(json.dumps(cfg));policy=SelectionWeights(p/'weight_config.json')
   self.assertFalse(policy.stage_tables);self.assertTrue(policy.weight_sum)
   for frame,expected in enumerate([20,15,1]):
    result=build_stage_fields(policy,'robotwin',8,frame,self.entry([False]*40+[True]*10),self.transform())
    self.assertEqual(result['stage_fm_weight'][0][0],expected);self.assertEqual(result['stage_current_depth_weight'],expected);self.assertEqual(result['stage_future_depth_weight'],0);self.assertEqual(result['stage_ids'][0],[-1,-1]);self.assertEqual(result['stage_fm_weight'][40][0],0)

 def test_legacy_target_coordinates_now_use_each_future_action_and_image_time(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);fixture(p);policy=SelectionWeights(p/'weight_config.json');policy.stage_tables={};policy.coordinate_weights={(8,0):1.,(8,1):15.,(8,2):20.,(8,49):10.}
   v=build_stage_fields(policy,'robotwin',8,0,self.entry(),self.transform())
   self.assertEqual([v['stage_fm_weight'][k][0]for k in range(3)],[1,15,20]);self.assertEqual(v['stage_current_depth_weight'],1);self.assertEqual(v['stage_future_depth_weight'],10);self.assertEqual(v['stage_future_video_weight'],10)
   transform=self.transform();transform.actions_convert_from_state={'action'}
   shifted=build_stage_fields(policy,'robotwin',8,0,self.entry(),transform)
   self.assertEqual([shifted['stage_fm_weight'][k][0]for k in range(2)],[15,20]);self.assertEqual(shifted['stage_future_depth_weight'],10)

import unittest
from .stapler_white_grasp import merge_coordinates,select_white_grasps
class WhiteStaplerTests(unittest.TestCase):
 def test_overlap_keeps_one_sample_with_two_labels(self):
  a=dict(episode_index=159,frame_index=80)
  merged=merge_coordinates([a],[dict(a,label=dict(stage='lift'))])
  self.assertEqual(len(merged),1);self.assertEqual(len(merged[0]['labels']),2)
  self.assertNotIn('loss_weight',merged[0])
 def test_original_broad_frames_preserved(self):
  a=dict(episode_index=150,frame_index=90);b=dict(episode_index=159,frame_index=20)
  self.assertEqual({(v['episode_index'],v['frame_index'])for v in merge_coordinates([a],[b])},{(150,90),(159,20)})
 def test_duplicate_coordinates_rejected(self):
  a=dict(episode_index=159,frame_index=80)
  with self.assertRaises(ValueError):merge_coordinates([a,a],[])
 def test_catalog_mismatch_rejected(self):
  with self.assertRaises(ValueError):select_white_grasps([], [dict(episode_index=159)],{},15,[],[])

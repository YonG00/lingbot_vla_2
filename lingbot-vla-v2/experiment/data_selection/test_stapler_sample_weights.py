import unittest
from .stapler_sample_weights import build,build_early_late
class StaplerSampleWeightsTests(unittest.TestCase):
 def catalog(self):return [dict(episode_index=1,length=4,scene={'{A}':'048_stapler/base3'}),dict(episode_index=2,length=4,scene={'{A}':'048_stapler/base0'})]
 def test_white_whole_demo_placement_priority_and_background(self):
  rows,s=build(self.catalog(),[dict(episode_index=1,frame_index=2),dict(episode_index=2,frame_index=3)])
  self.assertEqual([v['loss_weight']for v in rows],[15,15,20,15,1,1,1,20]);self.assertEqual(s['overlap_frames'],1);self.assertEqual(s['placement_frames'],2);self.assertEqual(s['white_all_frames'],4)
 def test_duplicate_or_foreign_placement_rejected(self):
  for rows in [[dict(episode_index=9,frame_index=0)],[dict(episode_index=1,frame_index=0)]*2]:
   with self.assertRaises(ValueError):build(self.catalog(),rows)
 def test_missing_white_or_place_rejected(self):
  with self.assertRaises(ValueError):build(self.catalog(),[])

 def test_early_late_white10_priority_and_boundary(self):
  events=[dict(episode_index=ep,arm='left',accepted=True,broad_postgrasp_start_frame=1,preplace_end=1,release_start_frame=2,release_end_frame=2,end_frame=3)for ep in [1,2]]
  placement=[dict(episode_index=ep,frame_index=f)for ep in [1,2]for f in [1,2,3]]
  rows,s=build_early_late(self.catalog(),placement,events)
  self.assertEqual([v['loss_weight']for v in rows],[10,15,20,20,1,15,20,20]);self.assertEqual(s['early_placement_frames'],2);self.assertEqual(s['late_placement_frames'],4)
 def test_split_rejects_missing_or_mismatched_event_coordinates(self):
  with self.assertRaises(ValueError):build_early_late(self.catalog(),[dict(episode_index=1,frame_index=2)],[])

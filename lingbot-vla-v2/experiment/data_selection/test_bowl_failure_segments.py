import unittest
from experiment.data_selection.bowl_failure_segments import collect_segments,FailureSegmentSettings
class FailureSegmentsTest(unittest.TestCase):
 def grasp(self,arm='left',accepted=True):return dict(episode_index=1,arm=arm,closure_crossing_frame=8,accepted=accepted,stages=dict(aim=[1,3],align=[4,5],descend=[6,6],settle=[7,7],close=[8,9],lift=[10,12]))
 def place(self):return dict(episode_index=1,arm='left',accepted=True,placement_ordinal=2,stages=dict(place_aim=[12,14],place_approach=[15,17],place_settle=[18,19],place_release=[20,21],place_retreat=[22,24]))
 def test_complete_contiguous_union_and_labels(self):
  segments,rows,cross=collect_segments([self.grasp()],[self.place()],{(1,'left',8):.3},{1:40},FailureSegmentSettings(action_horizon=3))
  self.assertEqual([r['frame_index'] for r in rows],list(range(1,25)))
  self.assertEqual(len(rows[11]['labels']),2)
  self.assertEqual(segments[0]['frame_count'],12)
  self.assertEqual(cross,dict(far_left_grasp=2,second_bowl_place=2))
 def test_filters_left_distance_accepted_and_second_only(self):
  wrong=self.place();wrong['placement_ordinal']=1
  segments,_,_=collect_segments([self.grasp(),self.grasp('right'),self.grasp(accepted=False)],[wrong],{(1,'left',8):.24},{1:40})
  self.assertEqual(segments,[])
 def test_bounds_and_horizon_contract(self):
  with self.assertRaises(ValueError):collect_segments([self.grasp()],[],{(1,'left',8):.3},{1:10})
  with self.assertRaises(ValueError):FailureSegmentSettings(action_horizon=0).validate()

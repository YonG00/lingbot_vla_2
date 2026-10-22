import unittest
from experiment.data_selection.bowl_grasp_place10 import placement_rows,AddPlaceSettings
class PlaceTenTests(unittest.TestCase):
 def event(self,ordinal=2):return dict(accepted=True,placement_ordinal=ordinal,episode_index=1,arm='left',approach_start=20,release_end=35,stages=dict(place_aim=[8,19],place_approach=[20,24],place_settle=[25,27],place_release=[28,35],place_retreat=[36,40]))
 def test_contiguous_context_through_release_no_retreat(self):
  events,rows=placement_rows([self.event(),self.event(1)],15,1/3)
  self.assertEqual(len(events),1);self.assertEqual([r['frame_index']for r in rows],list(range(15,36)))
 def test_context_clamps_to_aim_boundary(self):
  _,rows=placement_rows([self.event()],15,100)
  self.assertEqual(rows[0]['frame_index'],8)
 def test_invalid_settings(self):
  with self.assertRaises(ValueError):AddPlaceSettings(-1).validate()

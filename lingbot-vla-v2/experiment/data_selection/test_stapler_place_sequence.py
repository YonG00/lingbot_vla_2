import unittest,numpy as np
from experiment.data_selection.stapler_place_sequence import detect_episode,StaplerPlaceSettings
class StaplerPlaceTests(unittest.TestCase):
 def trajectory(self):
  n=140;p=np.zeros((n,3));p[:,0]=.2;p[:,2]=.8;p[31:51,2]=np.linspace(.8,.9,20);p[51:85,2]=.9;p[85:111,2]=np.linspace(.9,.8,26);p[51:71,0]=np.linspace(.2,0,20);p[71:,0]=0
  g=np.ones(n);g[21:31]=np.linspace(.9,0,10);g[31:121]=0;g[121:131]=np.linspace(.1,1,10)
  return p,g
 def test_selects_preparation_through_episode_end(self):
  p,g=self.trajectory();e=detect_episode(p,g,15)[0];self.assertTrue(e['accepted'],e);self.assertGreaterEqual(e['start_frame'],e['lift_confirm_frame']);self.assertLess(e['start_frame'],e['preplace_start']);self.assertEqual(e['end_frame'],139);self.assertLess(e['release_end_frame'],e['end_frame'])
 def test_no_lift_rejected(self):
  p,g=self.trajectory();p[:,2]=.8;self.assertFalse(detect_episode(p,g,15)[0]['accepted'])
 def test_reclose_after_release_rejected(self):
  p,g=self.trajectory();g[-1]=.7;e=detect_episode(p,g,15)[0];self.assertFalse(e['accepted']);self.assertIn('reclosed_after_release',e['rejection'])
 def test_no_finite_pose_and_bad_settings(self):
  p,g=self.trajectory();p[0,0]=float('nan')
  with self.assertRaises(ValueError):detect_episode(p,g,15)
  with self.assertRaises(ValueError):StaplerPlaceSettings(lift_min_m=-1).validate()

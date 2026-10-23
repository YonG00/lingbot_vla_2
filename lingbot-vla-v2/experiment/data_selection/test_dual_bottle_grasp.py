import unittest,json
from unittest.mock import patch
import numpy as np
from .dual_bottle_grasp import detect_arm,select_rows,Settings
class DualBottleTests(unittest.TestCase):
 def trajectory(self):
  n=100;p=np.zeros((n,3));p[:,2]=.8;p[:21,0]=np.linspace(-.4,-.08,21);p[21:32,0]=-.08;p[32:52,0]=np.linspace(-.08,0,20);p[63:89,2]=np.linspace(.8,.9,26);p[89:,2]=.9
  d=np.tile([1.,0,0],(n,1));g=np.ones(n);g[52:63]=np.linspace(.9,0,11);g[63:]=0;return p,d,g
 def test_horizontal_reach_and_lift_without_vertical_descent(self):
  p,d,g=self.trajectory();e=detect_arm(p,d,g,15)[0];self.assertTrue(e['accepted'],e);self.assertEqual(e['start_frame'],0);self.assertGreater(e['end_frame'],e['close_end_frame']);self.assertGreaterEqual(e['lift_m'],.08);json.dumps(e)
 def test_downward_grasp_not_misclassified(self):
  p,d,g=self.trajectory();d[:]=[0,0,-1];self.assertFalse(detect_arm(p,d,g,15)[0]['accepted'])
 def test_unlifted_grasp_rejected(self):
  p,d,g=self.trajectory();p[:,2]=.8;self.assertEqual(detect_arm(p,d,g,15)[0]['rejection'],'no_confirmed_lift')
 def test_one_arm_missing_excludes_entire_episode(self):
  p,d,g=self.trajectory();rows=[]
  for frame in range(len(g)):
   state=[0.]*14;state[6]=g[frame];state[13]=1.;rows.append(dict(episode_index=1,frame_index=frame,timestamp=frame/15,**{'observation.state':state}))
  with patch('experiment.data_selection.dual_bottle_grasp.forward_kinematics',return_value=(p,d)):
   summary,events,selected=select_rows(rows,{'left':{},'right':{}},15)
  self.assertEqual(selected,[]);self.assertEqual(summary['paired_episodes'],0)
 def test_two_hands_same_frames_deduplicated_with_both_labels(self):
  p,d,g=self.trajectory();rows=[]
  for frame in range(len(g)):
   state=[0.]*14;state[6]=g[frame];state[13]=g[frame];rows.append(dict(episode_index=1,frame_index=frame,timestamp=frame/15,**{'observation.state':state}))
  with patch('experiment.data_selection.dual_bottle_grasp.forward_kinematics',return_value=(p,d)):
   summary,events,selected=select_rows(rows,{'left':{},'right':{}},15)
  self.assertEqual(summary['paired_episodes'],1);self.assertEqual(summary['accepted_sequences'],2);self.assertEqual(summary['both_arm_frames'],len(selected));self.assertEqual(len(selected),len({v['frame_index']for v in selected}))
 def test_invalid_input_settings_rejected(self):
  with self.assertRaises(ValueError):Settings(horizontal_degrees=90).validate()
  p,d,g=self.trajectory();p[0,0]=float('nan')
  with self.assertRaises(ValueError):detect_arm(p,d,g,15)

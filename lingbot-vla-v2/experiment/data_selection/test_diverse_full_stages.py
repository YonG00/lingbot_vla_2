import unittest
from unittest.mock import patch
from . import test_dual_bottle_grasp as fixtures
from .dual_bottle_grasp import detect_arm
from .dual_bottle_full_stages import build

class DiverseFullStageTests(unittest.TestCase):
 def fixture(self):
  p,d,g=fixtures.DualBottleTests().trajectory();rows=[]
  for f in range(len(g)):
   state=[0.]*14;state[6]=float(g[f]);state[13]=float(g[f]);rows.append(dict(episode_index=1650,frame_index=f,timestamp=f/15,**{'observation.state':state}))
  event=detect_arm(p,d,g,15)[0];events=[dict(event,episode_index=1650,arm=a)for a in ['left','right']]
  return p,d,rows,events
 def run_build(self,task,objects=None):
  p,d,rows,events=self.fixture()
  with patch('experiment.data_selection.dual_bottle_full_stages.select_rows',return_value=({'paired_episodes':1,'episodes':1},events,[])),patch('experiment.data_selection.dual_bottle_full_stages.forward_kinematics',return_value=(p,d)):
   return build(rows,{'left':{},'right':{}},15,task,objects)
 def test_actual_random_object_identity_and_full_coverage(self):
  t,s=self.run_build('pick_diverse_bottles',{'1650':{'left':'001_bottle/base10','right':'001_bottle/base14'}})
  self.assertEqual(t['task'],'pick_diverse_bottles');self.assertTrue(s['full_coverage']);self.assertFalse(s['release_stage_present'])
  for arm,obj in [('left','001_bottle/base10'),('right','001_bottle/base14')]:
   seg=t['episodes'][0]['arms'][arm];self.assertEqual(sum(v['end']-v['start']for v in seg),100);self.assertEqual({v['object_id']for v in seg},{obj})
 def test_missing_objects_rejected(self):
  for objects in [None,{}, {'1650':{'left':'001_bottle/base10'}}]:
   with self.assertRaises(ValueError):self.run_build('pick_diverse_bottles',objects)
 def test_original_task_identity_preserved(self):
  t,_=self.run_build('pick_dual_bottles');self.assertEqual(t['episodes'][0]['arms']['left'][0]['object_id'],'001_bottle/base13')
 def test_unknown_task_rejected(self):
  with self.assertRaises(ValueError):self.run_build('other')

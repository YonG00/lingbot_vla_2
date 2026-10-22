import unittest
import numpy as np
from .placement_sequence import select_from_arrays,PlacementSettings

class PlacementTests(unittest.TestCase):
    def data(self):
        n=160;p=np.zeros((n,3));p[:,2]=.94;g=np.ones(n);events=[]
        for offset in [0,80]:
            g[offset:offset+11]=np.linspace(1,0,11);g[offset+11:offset+40]=0
            g[offset+40:offset+51]=np.linspace(0,1,11)
            p[offset:offset+20,0]=.04
            p[offset+20:offset+31,0]=np.linspace(.04,0,11)
            p[offset+20:offset+31,2]=np.linspace(.94,.86,11)
            p[offset+31:offset+51,2]=.86
            p[offset+51:offset+66,2]=np.linspace(.86,.94,15)
            events.append(dict(episode_index=7,arm='left',closure_crossing_frame=offset+5,lift_confirmation_frame=offset+12,accepted=True))
        state=np.zeros((n,14));state[:,6]=g;state[:,13]=1
        x=dict(episode=np.full(n,7),frame=np.arange(n),state=state,left_tcp=p,right_tcp=p.copy())
        return x,events
    def test_second_only_and_retreat(self):
        x,g=self.data();s,e,selected=select_from_arrays(x,g,15)
        self.assertEqual(s['accepted_target_sequences'],1)
        self.assertTrue(selected);self.assertTrue(all(r['frame_index']>=92 for r in selected))
        self.assertTrue(all(label['placement_ordinal']==2 for r in selected for label in r['labels']))
        self.assertIn('place_retreat',s['stage_frames'])
    def test_first_can_be_selected_explicitly(self):
        x,g=self.data();s,e,rows=select_from_arrays(x,g,15,PlacementSettings(placement_ordinal=1))
        self.assertEqual(s['accepted_target_sequences'],1);self.assertTrue(all(r['frame_index']<80 for r in rows))
    def test_missing_grasp_never_relabels_first_as_second(self):
        x,g=self.data();s,e,rows=select_from_arrays(x,g[:1],15)
        self.assertFalse(rows);self.assertEqual(s['rejections']['no_confirmed_grasp_proxy'],1)
    def test_missing_retreat_rejected(self):
        x,g=self.data();x['left_tcp'][131:,2]=.86
        s,e,rows=select_from_arrays(x,g,15);self.assertFalse(rows);self.assertIn('no_retreat',s['rejections'])
    def test_release_movement_rejected(self):
        x,g=self.data();x['left_tcp'][126,0]+=.06
        s,e,rows=select_from_arrays(x,g,15);self.assertFalse(rows);self.assertIn('moving_during_release',s['rejections'])
    def test_bad_frames_and_settings(self):
        x,g=self.data();x['frame'][2]=3
        with self.assertRaises(ValueError):select_from_arrays(x,g,15)
        with self.assertRaises(ValueError):PlacementSettings(placement_ordinal=3).validate()

if __name__=='__main__':unittest.main()

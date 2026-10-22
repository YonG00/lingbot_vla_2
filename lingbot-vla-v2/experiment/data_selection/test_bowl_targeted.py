import unittest
import numpy as np
from .bowl_targeted import select_arrays,TargetedSettings

class TargetedSelectionTests(unittest.TestCase):
    def data(self):
        n=42;p=np.zeros((n,3));p[:,2]=.8;p[:5,0]=.3;p[5:15,0]=np.linspace(.18,.06,10)
        p[30:36,2]=[.70,.68,.66,.64,.62,.60];p[36:,2]=.6
        direction=np.tile([0.,0.,-1.],(n,1));state=np.zeros((n,14));state[:,6]=1.;state[25:30,6]=0.;state[:,13]=1.
        x=dict(episode=np.full(n,7),frame=np.arange(n),state=state,left_tcp=p,right_tcp=p.copy(),left_direction=direction,right_direction=direction.copy())
        grasp=[dict(accepted=True,episode_index=7,arm='left',closure_crossing_frame=27,stages=dict(align=[15,19],descend=[20,22],settle=[23,24],close=[25,29]))]
        place=[dict(accepted=True,episode_index=7,arm='left',placement_ordinal=2,approach_start=30,stages=dict(place_release=[38,41],place_settle=[36,37]))]
        return x,grasp,place
    def test_far_scene_keeps_position_and_core(self):
        x,g,p=self.data();s,events,rows=select_arrays(x,g,p,15)
        self.assertEqual(s['qualified_far_grasps'],1);self.assertIn('far_grasp_position',s['stage_frames']);self.assertIn('far_grasp_close',s['stage_frames'])
        self.assertTrue(all(r['episode_index']==7 for r in rows))
    def test_near_initial_scene_excludes_entire_grasp(self):
        x,g,p=self.data();x['left_tcp'][0,0]=.2
        s,_,rows=select_arrays(x,g,p,15)
        self.assertEqual(s['qualified_far_grasps'],0)
        self.assertTrue(all(label['stage']=='second_place_height_band' for row in rows for label in row['labels']))
    def test_place_only_height_band_without_release_or_retreat(self):
        x,g,p=self.data();s,_,rows=select_arrays(x,[],p,15)
        self.assertEqual([r['frame_index'] for r in rows],[31,32,33])
        self.assertEqual(s['second_placements_with_height_samples'],1)
    def test_first_placement_is_not_selected(self):
        x,g,p=self.data();p[0]['placement_ordinal']=1
        s,_,rows=select_arrays(x,[],p,15);self.assertFalse(rows)
    def test_idle_far_position_is_not_selected(self):
        x,g,p=self.data();x['left_tcp'][5:15,0]=.1
        _,_,rows=select_arrays(x,g,[],15)
        labels=[label['stage']for row in rows if row['frame_index']==10 for label in row['labels']]
        self.assertNotIn('far_grasp_position',labels)
    def test_orientation_improvement_counts_as_positioning(self):
        x,g,p=self.data();x['left_tcp'][5:15,0]=.1
        for i,angle in enumerate(np.linspace(.8,0,10),5):x['left_direction'][i]=[np.sin(angle),0,-np.cos(angle)]
        _,_,rows=select_arrays(x,g,[],15)
        self.assertTrue(any(row['frame_index']==10 and any(l['stage']=='far_grasp_position'for l in row['labels'])for row in rows))
    def test_core_can_be_disabled_and_thresholds_validated(self):
        x,g,p=self.data();s,_,rows=select_arrays(x,g,[],15,TargetedSettings(include_far_grasp_core=False))
        self.assertNotIn('far_grasp_close',s['stage_frames'])
        with self.assertRaises(ValueError):TargetedSettings(place_height_min_m=.1,place_height_max_m=.03).validate()
        with self.assertRaises(ValueError):TargetedSettings(far_initial_xy_min_m=float('nan')).validate()

if __name__=='__main__':unittest.main()

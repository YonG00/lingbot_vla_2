import unittest
from .bowl_calibration import uniform_frames,calibrate_events,CalibrationSettings

class CalibrationTests(unittest.TestCase):
    def event(self,ep):return dict(accepted=True,episode_index=ep,arm='left',closure_crossing_frame=47,stages=dict(aim=[0,19],align=[20,27],descend=[28,37],settle=[38,43],close=[44,54]))
    def test_uniform_endpoints_and_short_segments(self):
        self.assertEqual(uniform_frames(0,9,4),[0,3,6,9]);self.assertEqual(uniform_frames(5,7,10),[5,6,7]);self.assertEqual(uniform_frames(5,4,10),[])
    def test_all_grasps_preserved_and_quota_limited(self):
        events=[self.event(7),self.event(8)];diagnostics,rows=calibrate_events(events,[])
        self.assertEqual(len(diagnostics),2);self.assertEqual({r['episode_index']for r in rows},{7,8})
        for event in diagnostics:
            self.assertEqual(len(event['anchors']['aim']),13);self.assertEqual(len(event['anchors']['close']),8)
            self.assertEqual(event['anchors']['close'][0],44);self.assertEqual(event['anchors']['close'][-1],54)
    def test_extra_overlap_deduplicated_and_unrelated_stages_ignored(self):
        extras=[dict(episode_index=7,frame_index=0,labels=[dict(stage='far_grasp_position',arm='left')]),dict(episode_index=7,frame_index=90,labels=[dict(stage='second_place_height_band',arm='left')]),dict(episode_index=7,frame_index=91,labels=[dict(stage='place_retreat',arm='left')])]
        _,rows=calibrate_events([self.event(7)],extras);keys=[r['frame_index']for r in rows]
        self.assertEqual(keys.count(0),1);self.assertIn(90,keys);self.assertNotIn(91,keys)
        self.assertEqual(len(next(r for r in rows if r['frame_index']==0)['labels']),2)
    def test_invalid_quotas_rejected(self):
        with self.assertRaises(ValueError):CalibrationSettings(stage_quota=dict(aim=1)).validate()
        with self.assertRaises(ValueError):CalibrationSettings(target_fraction=float('nan')).validate()

if __name__=='__main__':unittest.main()

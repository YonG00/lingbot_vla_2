import unittest,tempfile,json
from pathlib import Path
from .combine import combine_directories

class UnionTests(unittest.TestCase):
    def make(self,root,name,frames,source='same'):
        p=Path(root)/name;p.mkdir()
        (p/'summary.json').write_text(json.dumps(dict(task='stack_bowls_two',fps=15,numeric_source_sha256=source,spec_sha256='spec',rule_version=name,frames=10)))
        (p/'selected_frames.json').write_text(json.dumps([dict(episode_index=7,frame_index=f,labels=[dict(stage=name)])for f in frames]))
        return p
    def test_overlap_is_one_sample_with_both_labels(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.make(root,'grasp',[1,2]);b=self.make(root,'place',[2,3])
            s,rows=combine_directories([a,b]);self.assertEqual(s['selected_frames'],3);self.assertEqual(s['duplicate_memberships'],1)
            self.assertEqual(len(rows[1]['labels']),2)
    def test_different_dataset_fingerprint_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.make(root,'grasp',[1]);b=self.make(root,'place',[2],'other')
            with self.assertRaises(ValueError):combine_directories([a,b])

class EvaluationDependencyTests(unittest.TestCase):
    def test_explicit_simulator_root_and_all_videos(self):
        import os
        from unittest.mock import patch
        from experiment.task_recipes.resident_evaluation import jobs_for
        c=dict(evaluation=dict(tasks=['stack_bowls_two'],episodes_per_task=3,video_episodes=3))
        with patch.dict(os.environ,{'LINGBOT_ROBOTWIN_ROOT':'/tmp/validated_simulator'}):
            jobs=jobs_for(c,Path('/tmp/run'),200,dict(model=dict(tokenizer_path='/tmp/qwen')))
        self.assertEqual(jobs[0]['robotwin'],str(Path('/tmp/validated_simulator').resolve()))
        self.assertEqual(jobs[0]['video_episodes'],3)

if __name__=='__main__':unittest.main()

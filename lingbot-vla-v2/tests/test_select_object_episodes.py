"""合成Parquet验证来源、v3偏移、审核门控；无模型依赖。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
SPEC=importlib.util.spec_from_file_location('selector',Path(__file__).resolve().parents[1]/'tools/select_object_episodes.py')
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)

class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'dataset'
        (self.root/'meta/episodes').mkdir(parents=True);(self.root/'data').mkdir()
        self.envs=self.base/'envs';self.envs.mkdir()
        (self.envs/'aux.py').write_text('objects=["047_mouse"]')
        self.emb=self.base/'robot.yml'
        self.emb.write_text('arm_joints_name: '+json.dumps([[f'{a}_joint{i}' for i in range(1,7)] for a in ('fl','fr')]))
        texts=['place mouse on pad','lift mouse onto stand','move box left of mouse','move plain object']
        meta=[];self.frames=[]
        for ep,text in enumerate(texts):
            meta.append(dict(episode_index=ep,tasks=[text],length=4,dataset_from_index=ep*4,dataset_to_index=(ep+1)*4))
            for i in range(4):self.frames.append(dict(index=ep*4+i,episode_index=ep,frame_index=i,timestamp=i/5,task_index=ep,**{'observation.state':[0.0]*14}))
        pq.write_table(pa.Table.from_pylist(meta),self.root/'meta/episodes/file.parquet')
        pq.write_table(pa.Table.from_pylist(self.frames),self.root/'data/file.parquet')
        pq.write_table(pa.Table.from_pylist([{'task_index':i,'__index_level_0__':t} for i,t in enumerate(texts)]),self.root/'meta/tasks.parquet')
        (self.root/'meta/info.json').write_text(json.dumps(dict(fps=5,total_episodes=4,total_frames=16,features={})))
        self.manifest=self.base/'manifest.json'
        self.manifest.write_text(json.dumps({'tasks':{'main':{'train_ids':[0],'val_ids':[],'n_total':1},'aux':{'train_ids':[1,2],'val_ids':[3],'n_total':3}}}))
    def args(self,name='out',review=None):
        return SimpleNamespace(dataset_root=self.root,target='mouse',keyword=[],object_id=[],primary_task='main',output_dir=self.base/name,manifest=self.manifest,source_envs=self.envs,candidate_task=[],embodiment_config=self.emb,review=review,extract_keyframes=False,ffmpeg='ffmpeg')
    def records(self):return [json.loads(x) for x in (self.base/'out/candidates.jsonl').read_text().splitlines()]
    def test_conservative_recall_and_no_automatic_approval(self):
        summary=s.run(self.args())
        self.assertEqual(summary['candidate_episodes'],3)
        self.assertEqual(summary['keyword_auxiliary_episodes'],2)
        self.assertEqual(summary['quality_errors'],0)
        self.assertEqual([r['episode_index'] for r in self.records()],[1,2,3])
        self.assertEqual(json.loads((self.base/'out/approved.json').read_text())['episodes'],[])
        self.assertNotIn('local_idx',json.dumps(self.records()))
        self.assertEqual(pq.read_table(self.root/'data/file.parquet').to_pylist(),self.frames)
        with self.assertRaises(FileExistsError):s.run(self.args())
    def test_review_fingerprint_success_and_identity_required(self):
        summary=s.run(self.args());fp=summary['dataset_fingerprint'];records=self.records()
        d=dict(episode_index=1,decision='approved',reviewer='tester',evidence='video reviewed',manipulated_target_verified=True,demonstration_success_verified=True,phase_windows={'grasp_lift':[[0,2]]})
        review=dict(dataset_fingerprint=fp,decisions=[d])
        self.assertEqual(len(s.apply_reviews(records,review,fp)),1)
        for key,val in [('manipulated_target_verified',False),('demonstration_success_verified',False),('phase_windows',{'grasp_lift':[[0,4]]}),('episode_index',0)]:
            with self.subTest(key=key),self.assertRaises(ValueError):s.apply_reviews(records,{**review,'decisions':[{**d,key:val}]},fp)
        with self.assertRaises(ValueError):s.apply_reviews(records,review,'stale')
        p=self.base/'review.json';p.write_text(json.dumps(review))
        self.assertEqual(s.run(self.args('reviewed',review=p))['approved_episodes'],1)
    def test_instruction_and_frame_mapping_rejected(self):
        meta=dict(episode_index=0,tasks=['real'],length=4,dataset_from_index=0,dataset_to_index=4);rows=self.frames[:4]
        with self.assertRaisesRegex(ValueError,'instruction'):s.validate_frames(rows,meta,5,{0:'wrong'})
        with self.assertRaisesRegex(ValueError,'absolute'):s.validate_frames([{**r,'index':r['index']+1} for r in rows],meta,5,{0:'real'})
        with self.assertRaisesRegex(ValueError,'frame_index'):s.validate_frames(rows[:-1],meta,5,{0:'real'})
    def test_v3_offsets_never_guessed(self):
        info=dict(video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',features={'cam':{'dtype':'video'}})
        row={'videos/cam/'+k:v for k,v in dict(chunk_index=1,file_index=2,from_timestamp=123.4,to_timestamp=130.0).items()}
        v=s.video_segments(self.root,info,row)[0]
        self.assertEqual(v['from_timestamp'],123.4);self.assertTrue(v['path'].endswith('chunk-001/file-002.mp4'))
        self.assertEqual(s.video_segments(self.root,info,{}),[])
        with self.assertRaises(ValueError):s.inside(self.root,'../outside.mp4')
    def test_target_parameter_and_word_boundaries(self):
        words,ids=s.target_terms('mouse');self.assertTrue(s.matches('computer mouse',words));self.assertFalse(s.matches('mousepad',words));self.assertIn('047_mouse',ids)
        words,_=s.target_terms('stapler',['订书机'],['048_stapler']);self.assertFalse(s.matches('computer mouse',words));self.assertTrue(s.matches('blue stapler',words))
    def test_bad_episode_cannot_be_approved(self):
        bad=[{**r,'timestamp':10.0} if r['episode_index']==1 else r for r in self.frames]
        pq.write_table(pa.Table.from_pylist(bad),self.root/'data/file.parquet')
        summary=s.run(self.args());self.assertEqual(summary['quality_errors'],1)
        d=dict(episode_index=1,decision='approved',reviewer='tester',evidence='video',manipulated_target_verified=True,demonstration_success_verified=True)
        with self.assertRaises(ValueError):s.apply_reviews(self.records(),dict(dataset_fingerprint=summary['dataset_fingerprint'],decisions=[d]),summary['dataset_fingerprint'])

if __name__=='__main__':unittest.main()

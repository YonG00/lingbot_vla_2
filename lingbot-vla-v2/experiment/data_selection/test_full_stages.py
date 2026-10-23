import hashlib,json,tempfile,unittest,shutil
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from .full_stages import StageTable,validate_table,derive_arm_layout,STAGES,policy_coverage
from .weights import SelectionWeights
from .inheritance import freeze_registry

def fixture(p,length=200):
 def seg(a,b,s):return dict(start=a,end=b,stage=s,substage=s,event_id='e1',object_id='bottle',proxy_only=True)
 table=dict(schema_version=1,dataset_name='robotwin',task='test_task',fps=15,coordinate_space='original_episode_frame',source_action_channels={'action':{'left':list(range(7)),'right':list(range(7,14))}},episodes=[dict(episode_index=7,length=length,arms={'left':[seg(0,120,'reach_aim'),seg(120,126,'pre_contact'),seg(126,length,'transport')],'right':[seg(0,length,'transport')]})])
 weights={s:1 for s in STAGES};weights.update(pre_contact=5,gripper_close=5,fine_align=2,grasp_lift=2)
 policy=dict(version=1,normalization='weight_sum',fm=weights,visual=dict(weights),future_video_rule='target_endpoint')
 for name,value in [('segments',table),('policy',policy),('selected', [dict(episode_index=7,frame_index=f)for f in range(length)])]:(p/(name+'.json')).write_text(json.dumps(value))
 sha=lambda f:hashlib.sha256((p/f).read_bytes()).hexdigest()
 cfg=dict(version=1,task='test_task',dataset_name='robotwin',dataset_root='/tmp/shared_robotwin',normalization='weight_sum',multiplier=1,selected_frames='selected.json',selected_frames_sha256=sha('selected.json'),stage_table='segments.json',stage_table_sha256=sha('segments.json'),stage_policy='policy.json',stage_policy_sha256=sha('policy.json'))
 (p/'weight_config.json').write_text(json.dumps(cfg));return cfg,table,policy

class FullStageTests(unittest.TestCase):
 def test_chunk_queries_future_stages_without_anchor_broadcast(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p);w=StageTable(cfg,p);v=w.sample(7,100,50,2,dict(left=[0],right=[1]))
   self.assertEqual([r[0]for r in v['fm']],[1.]*20+[5.]*6+[1.]*24);self.assertEqual([r[1]for r in v['fm']],[1.]*50)
   self.assertEqual(v['current_depth'],1);self.assertEqual(v['future_depth'],1)
 def test_visual_max_of_hands_at_its_own_time(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p);w=StageTable(cfg,p);v=w.sample(7,120,50,2,dict(left=[0],right=[1]))
   self.assertEqual(v['current_depth'],5);self.assertEqual(v['future_depth'],1);self.assertEqual(v['current_video'],5);self.assertEqual(v['future_video'],1)
 def test_padding_never_inherits_last_high_weight_or_crosses_episode(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p);w=StageTable(cfg,p);v=w.sample(7,198,50,2,dict(left=[0],right=[1]))
   self.assertEqual(v['valid'],[True,True]+[False]*48);self.assertTrue(all(row==[0.,0.]for row in v['fm'][2:]));self.assertEqual(v['future_depth'],0);self.assertEqual(v['future_video'],0)
 def test_state_conversion_has_one_step_offset_without_moving_visual_clock(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p);w=StageTable(cfg,p);v=w.sample(7,119,2,2,dict(left=[0],right=[1]),action_offset=1)
   self.assertEqual(v['fm'][0][0],5);self.assertEqual(v['current_depth'],1);self.assertEqual(v['future_depth'],5)
 def test_gap_overlap_and_missing_arm_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,table,_=fixture(p)
   for value in [119,121]:
    obj=json.loads(json.dumps(table));obj['episodes'][0]['arms']['left'][1]['start']=value
    with self.assertRaises(ValueError):validate_table(obj)
   del table['episodes'][0]['arms']['right']
   with self.assertRaises(ValueError):validate_table(table)
 def test_unknown_cannot_be_promoted(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,policy=fixture(p);policy['fm']['unknown']=5;(p/'policy.json').write_text(json.dumps(policy));cfg['stage_policy_sha256']=hashlib.sha256((p/'policy.json').read_bytes()).hexdigest()
   with self.assertRaises(ValueError):StageTable(cfg,p)
 def test_fingerprints_enforced(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p);(p/'segments.json').write_text('{}')
   with self.assertRaises(ValueError):StageTable(cfg,p)
 def test_action_layout_follows_reordered_features(self):
  info=SimpleNamespace(joints=['effector.position','arm.position','end.position'],joints_max_dim={'effector.position':2,'arm.position':14,'end.position':14})
  mapping={'action.effector.position':dict(origin_keys=OrderedDict([('action',dict(start=6,end=7)),('action*',dict(start=13,end=14))])),'action.arm.position':dict(origin_keys=OrderedDict([('action',dict(start=0,end=6)),('action*',dict(start=7,end=13))]))}
  channels={'action':dict(left=list(range(7)),right=list(range(7,14)))}
  layout=derive_arm_layout(info,mapping,list(range(14)),channels);self.assertEqual(layout,dict(left=[0]+list(range(2,8)),right=[1]+list(range(8,14))))
  with self.assertRaises(ValueError):derive_arm_layout(info,mapping,list(range(13)),channels)
 def test_stage_dependencies_freeze_without_original_files(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);source=p/'source';source.mkdir();cfg,_,_=fixture(source);out=p/'out';out.mkdir();registry=freeze_registry(p/'empty',out,source/'weight_config.json');shutil.rmtree(source)
   loaded=SelectionWeights(registry);self.assertEqual(set(loaded.stage_tables),{7});self.assertEqual(loaded.weight('robotwin',7,120),1);self.assertFalse(loaded.action_overrides)
   self.assertEqual(loaded.stage_tables[7].sample(7,120,1,2,dict(left=[0],right=[1]))['fm'][0],[5.,1.])
 def test_substage_override_is_local_to_arm_branch_and_time(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,policy=fixture(p);policy['overrides']={'fm':{'left':{'reach_aim/reach_aim':3}},'current_depth':{'right':{'transport/transport':2}}};(p/'policy.json').write_text(json.dumps(policy));cfg['stage_policy_sha256']=hashlib.sha256((p/'policy.json').read_bytes()).hexdigest();w=StageTable(cfg,p)
   sample=w.sample(7,100,50,2,dict(left=[0],right=[1]));self.assertEqual(sample['fm'][0],[3.,1.]);self.assertEqual(sample['fm'][20],[5.,1.]);self.assertEqual(sample['current_depth'],2);self.assertEqual(sample['future_depth'],1)
 def test_unknown_substage_override_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,policy=fixture(p);policy['overrides']={'fm':{'left':{'reach_aim/misspelled':5}}};(p/'policy.json').write_text(json.dumps(policy));cfg['stage_policy_sha256']=hashlib.sha256((p/'policy.json').read_bytes()).hexdigest()
   with self.assertRaises(ValueError):StageTable(cfg,p)
 def test_coverage_counts_real_steps_and_endpoint_branches(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p);w=StageTable(cfg,p);stats=policy_coverage(w,50)
   self.assertEqual(stats['all_anchor_padded_time_steps'],1225)
   self.assertEqual(sum(stats['all_anchor_chunk_fm_arm_steps']['left'].values()),200*50-1225)
   self.assertEqual(stats['visual_weight_samples']['future_depth'][0.],49)
 def test_stage_rejects_legacy_arm_multiplier_stacking(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p);cfg['multiplier']=2;(p/'weight_config.json').write_text(json.dumps(cfg))
   with self.assertRaises(ValueError):SelectionWeights(p/'weight_config.json')

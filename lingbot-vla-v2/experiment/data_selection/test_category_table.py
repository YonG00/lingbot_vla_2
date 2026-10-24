import unittest,json,tempfile
from pathlib import Path
from .category_conversion import from_targets,from_stages,base_policy,write_snapshot
from .category_table import CategoryTable
from .test_full_stages import fixture
from .weights import SelectionWeights
from .inheritance import freeze_registry
class CategoryTableTests(unittest.TestCase):
 def target(self,p,arm=None):
  cfg=dict(task='t',dataset_name='robotwin',dataset_root='/tmp/shared_robotwin');policy=base_policy({'background':{'fm':1,'visual':1},'early':{'fm':15,'visual':15},'late':{'fm':20,'visual':20},'white':{'fm':10,'visual':10}})
  def label(name):return dict(source=name,**({'arm':arm}if arm else {}))
  rows=[dict(episode_index=1,frame_index=1,labels=[label('early'),label('white')]),dict(episode_index=1,frame_index=2,labels=[label('late'),label('white')]),dict(episode_index=1,frame_index=3,labels=[label('white')])]
  table,policy=from_targets(cfg,rows,[dict(episode_index=1,length=5)],policy);return write_snapshot(cfg,table,policy,p)
 def test_chunk_crosses_categories_and_visual_uses_actual_target(self):
  with tempfile.TemporaryDirectory()as t:
   s=self.target(Path(t)/'labels');v=s.sample(1,0,3,2,{'left':[0],'right':[1]})
   self.assertEqual([r[0]for r in v['fm']],[1,15,20]);self.assertEqual(v['current_depth'],1);self.assertEqual(v['future_depth'],20);self.assertEqual(v['future_video'],20)
 def test_offset_changes_actions_not_image_clock_and_padding_zero(self):
  with tempfile.TemporaryDirectory()as t:
   s=self.target(Path(t)/'labels');v=s.sample(1,2,4,2,{'left':[0],'right':[1]},1)
   self.assertEqual([r[0]for r in v['fm']],[10,1,0,0]);self.assertEqual(v['future_depth'],0);self.assertEqual(v['ids'][-1],[-1,-1])
 def test_unilateral_category_and_shared_visual_max(self):
  with tempfile.TemporaryDirectory()as t:
   s=self.target(Path(t)/'labels','left');v=s.sample(1,0,3,2,{'left':[0],'right':[1]});self.assertEqual(v['fm'][2],[20,1]);self.assertEqual(v['future_depth'],20)
 def test_full_stage_adapter_matches_every_existing_weight_and_sample(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg,_,_=fixture(p);table,policy=from_stages(cfg,p);new=write_snapshot(cfg,table,policy,p/'converted')
   from .full_stages import StageTable
   old=StageTable(cfg,p)
   for b in ['fm','current_depth','future_depth','current_video','future_video']:self.assertEqual(old.frame_weights[b],new.frame_weights[b])
   for f in [0,100,121,198]:self.assertEqual(old.sample(7,f,50,2,{'left':[0],'right':[1]}),new.sample(7,f,50,2,{'left':[0],'right':[1]}))
 def test_freeze_registry_copies_both_category_dependencies(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);self.target(p/'labels');registry=freeze_registry(p/'empty',p/'frozen',explicit=p/'labels/weight_config.json');w=SelectionWeights(registry);self.assertTrue(w.weight_sum);self.assertIn(1,w.stage_tables)
   (p/'labels/category_loss_weights.json').write_text('{}');SelectionWeights(registry)
 def test_unknown_class_fails_and_background_cannot_be_high(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);s=self.target(p/'labels');cfg=json.loads((p/'labels/weight_config.json').read_text());policy=s.policy;policy['classes']['background']['fm']=2
   with self.assertRaises(ValueError):write_snapshot(cfg,s.table,policy,p/'bad')

 def test_unknown_target_class_is_not_silently_background(self):
  cfg=dict(task='t',dataset_name='robotwin',dataset_root='/tmp/shared_robotwin');policy=base_policy({'background':{'fm':1,'visual':1}})
  with self.assertRaises(ValueError):from_targets(cfg,[dict(episode_index=1,frame_index=0,labels=[dict(source='misspelled')])],[dict(episode_index=1,length=2)],policy)
 def test_bad_normalization_is_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);s=self.target(p/'labels');cfg=json.loads((p/'labels/weight_config.json').read_text());policy=dict(s.policy,normalization='sample_count')
   with self.assertRaises(ValueError):write_snapshot(cfg,s.table,policy,p/'bad')

 def test_target_arm_typo_is_not_silently_ignored(self):
  cfg=dict(task='t',dataset_name='robotwin',dataset_root='/tmp/shared_robotwin');policy=base_policy({'background':{'fm':1,'visual':1}})
  with self.assertRaises(ValueError):from_targets(cfg,[dict(episode_index=1,frame_index=0,labels=[dict(source='background',arm='lef')])],[dict(episode_index=1,length=2)],policy)

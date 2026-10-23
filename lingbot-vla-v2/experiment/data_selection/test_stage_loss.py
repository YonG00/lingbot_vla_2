import ast,json,tempfile,types,unittest
from pathlib import Path
import torch
from lingbotvla.utils.sample_loss_weight import weighted_mean,fm_weighted_mean,stage_fm_mean,stage_fm_metrics,stage_valid_mask,normalized_sample_weights
ROOT=Path(__file__).resolve().parents[2]
class StageLossTests(unittest.TestCase):
 def test_weight_sum_not_sample_count(self):
  x=torch.tensor([[[1.,3.]]],requires_grad=True);w=torch.tensor([[[5.,1.]]]);loss=stage_fm_mean(x,w,None,torch.zeros(1,1,dtype=torch.bool));self.assertAlmostEqual(loss.item(),8/6);loss.backward();torch.testing.assert_close(x.grad,torch.tensor([[[5/6,1/6]]]))
 def test_constant_weights_do_not_amplify_global_gradient(self):
  x=torch.arange(12,dtype=torch.float32).reshape(2,2,3).requires_grad_();pad=torch.zeros(2,2,dtype=torch.bool);base=stage_fm_mean(x,torch.ones_like(x),None,pad);changed=stage_fm_mean(x,torch.full_like(x,5),None,pad)
  torch.testing.assert_close(base,changed);torch.testing.assert_close(torch.autograd.grad(base,x,retain_graph=True)[0],torch.autograd.grad(changed,x)[0])
 def test_padding_and_invalid_joints_have_zero_gradient(self):
  x=torch.tensor([[[2.,1000.],[999.,999.]]],requires_grad=True);w=torch.full_like(x,5);mask=torch.tensor([[[True,False],[True,False]]]);pad=torch.tensor([[False,True]])
  loss=stage_fm_mean(x,w,mask,pad);self.assertEqual(loss.item(),2);loss.backward();torch.testing.assert_close(x.grad,torch.tensor([[[1.,0.],[0.,0.]]]))
 def test_empty_masks_are_finite_zero(self):
  x=torch.ones(2,3,4,requires_grad=True);loss=stage_fm_mean(x,torch.zeros_like(x),None,torch.ones(2,3,dtype=torch.bool));self.assertEqual(loss.item(),0);loss.backward();self.assertFalse(x.grad.any())
  out=weighted_mean(x,normalized_sample_weights(torch.zeros(2)));self.assertEqual(out.item(),0)
 def test_auxiliary_weight_sum_excludes_invalid_future_samples(self):
  x=torch.tensor([[1.,1.],[3.,3.],[999.,999.]],requires_grad=True);loss=weighted_mean(x,normalized_sample_weights(torch.tensor([5.,1.,0.])));self.assertAlmostEqual(loss.item(),8/6);loss.backward();self.assertFalse(x.grad[2].any())
 def policy_method(self):
  path=ROOT/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py';tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='LingbotVlaV2Policy');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='forward');method.returns=None
  env=dict(torch=torch,weighted_mean=weighted_mean,fm_weighted_mean=fm_weighted_mean,stage_fm_mean=stage_fm_mean,stage_fm_metrics=stage_fm_metrics,stage_valid_mask=stage_valid_mask,normalized_sample_weights=normalized_sample_weights);exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env);return env['forward']
 def test_actual_policy_uses_absolute_fm_and_routes_four_auxiliary_vectors(self):
  captured={}
  class Core:
   def forward(self,*args,**kw):
    captured.update(kw);v=args[5];mean=lambda key:weighted_mean(v,kw[key]);return v,mean('stage_current_depth_weight'),mean('stage_future_depth_weight'),mean('stage_future_video_weight')+mean('stage_current_video_weight'),None,torch.tensor(3.),torch.tensor(5.),{},None,None,None
  obj=types.SimpleNamespace(config=types.SimpleNamespace(loss_type='fm',action_dim=2,align_params={}),model=Core());x=torch.tensor([[[1.,1.]],[[3.,3.]]],requires_grad=True);fm=torch.tensor([[[5.,1.]],[[1.,1.]]]);pad=torch.zeros(2,1,dtype=torch.bool)
  result=self.policy_method()(obj,None,None,x,None,None,x,action_is_pad=pad,sample_loss_weight=torch.full((2,),99.),action_loss_weight=torch.full((2,2),99.),stage_fm_weight=fm,stage_current_depth_weight=torch.tensor([5.,1.]),stage_future_depth_weight=torch.tensor([1.,5.]),stage_current_video_weight=torch.tensor([1.,1.]),stage_future_video_weight=torch.tensor([0.,1.]))
  self.assertIsNone(captured['sample_loss_weight']);self.assertAlmostEqual(result[1].item(),12/8);self.assertAlmostEqual(result[2].item(),8/6,places=6);self.assertAlmostEqual(result[3].item(),16/6,places=6);self.assertAlmostEqual(result[4].item(),5);self.assertEqual(result[5].item(),3)
  self.assertAlmostEqual(result[0].item(),1.5+8/6+16/6+5+8,places=5)
 def test_real_depth_helper_keeps_current_and_future_normalizers_separate(self):
  path=ROOT/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py';tree=ast.parse(path.read_text());methods={}
  for cls in tree.body:
   if isinstance(cls,ast.ClassDef):
    for n in cls.body:
     if isinstance(n,ast.FunctionDef)and n.name in ['_emb_loss','depth_emb_forward']:methods[n.name]=n
  import einops
  env=dict(torch=torch,F=torch.nn.functional,weighted_mean=weighted_mean,einops=einops);exec(compile(ast.Module(body=list(methods.values()),type_ignores=[]),str(path),'exec'),env)
  pred=torch.tensor([[[1.]],[[3.]]],requires_grad=True);obj=types.SimpleNamespace(llm_image_token_size=1,depth_align_embs=torch.zeros(1,1,1),future_depth_align_embs=torch.zeros(1,1,1),use_future_depth=True,depth_align_head=lambda *a:pred,future_depth_align_head=lambda *a:pred,_current_depth_task_tokens=lambda x,**kw:x[:,:1],_future_depth_task_tokens=lambda x:x[:,:1]);obj._emb_loss=types.MethodType(env['_emb_loss'],obj)
  current,future,*_=env['depth_emb_forward'](obj,torch.ones(2,3,1),torch.zeros_like(pred),None,torch.zeros_like(pred),normalized_sample_weights(torch.tensor([5.,1.])),normalized_sample_weights(torch.tensor([1.,5.])))
  self.assertAlmostEqual(current.item(),5/6,places=6);self.assertAlmostEqual(future.item(),13/6,places=6)
 def test_shape_contracts_fail_before_broadcast(self):
  with self.assertRaises(ValueError):stage_fm_mean(torch.ones(2,3,4),torch.ones(2,4),None,torch.zeros(2,3,dtype=torch.bool))
  with self.assertRaises(ValueError):stage_fm_mean(torch.ones(2,3,4),torch.ones(2,3,4),None,None)

class StageCollatorTests(unittest.TestCase):
 def method(self):
  path=ROOT/'lingbotvla/data/multimodal/data_collator.py';tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='VLADataCollatorWithPacking');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__call__');method.returns=None
  for arg in method.args.args:arg.annotation=None
  env=dict(torch=torch,default_collate=torch.utils.data.default_collate);exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env);return env['__call__']
 def feature(self):
  return dict(actions=torch.ones(2,2),action_is_pad=torch.zeros(2,dtype=torch.bool),stage_fm_weight=torch.ones(2,2),stage_ids=torch.ones(2,2,dtype=torch.int64),stage_arm_index=torch.tensor([0,1]),**{name:torch.tensor(1.)for name in ['stage_current_depth_weight','stage_future_depth_weight','stage_current_video_weight','stage_future_video_weight']})
 def test_actual_collator_stacks_every_stage_field(self):
  obj=types.SimpleNamespace(state_features=['actions','action_is_pad']);result=self.method()(obj,[self.feature(),self.feature()]);self.assertEqual(result['stage_fm_weight'].shape,(2,2,2));self.assertEqual(result['stage_current_depth_weight'].shape,(2,))
 def test_missing_metadata_in_mixed_batch_rejected(self):
  a,b=self.feature(),self.feature();del b['stage_ids']
  with self.assertRaises(ValueError):self.method()(types.SimpleNamespace(state_features=[]),[a,b])
 def test_wrong_auxiliary_or_time_shape_rejected(self):
  for name,value in [('stage_current_depth_weight',torch.ones(2)),('stage_fm_weight',torch.ones(3,2))]:
   a=self.feature();a[name]=value
   with self.assertRaises(ValueError):self.method()(types.SimpleNamespace(state_features=[]),[a])

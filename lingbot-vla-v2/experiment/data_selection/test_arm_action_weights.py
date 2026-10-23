import hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from .weights import SelectionWeights
from .inheritance import freeze_registry
from lingbotvla.utils.sample_loss_weight import weighted_mean,fm_weighted_mean
class ArmActionWeightsTests(unittest.TestCase):
 def config(self,p):
  rows=[dict(episode_index=2150,frame_index=i,labels=[dict(arm=a)for a in arms])for i,arms in [(0,['left','right']),(1,['right']),(2,['left'])]]
  payload=json.dumps(rows).encode();(p/'selected.json').write_bytes(payload)
  c=dict(version=1,task='pick_dual_bottles',dataset_name='robotwin',dataset_root='/tmp/robotwin',normalization='sample_count',multiplier=2,selected_frames='selected.json',selected_frames_sha256=hashlib.sha256(payload).hexdigest(),arm_action_weights=dict(left=3,right=2),arm_action_dimensions=dict(left=list(range(6))+[28],right=list(range(6,12))+[29]))
  (p/'config.json').write_text(json.dumps(c));return p/'config.json'
 def test_overlap_and_one_arm_boundaries(self):
  with tempfile.TemporaryDirectory()as t:
   w=SelectionWeights(self.config(Path(t)))
   self.assertEqual(w.relative_action_weights('robotwin',2150,0,55)[0],1.5)
   self.assertEqual(w.relative_action_weights('robotwin',2150,0,55)[6],1)
   self.assertEqual(w.relative_action_weights('robotwin',2150,1,55)[0],.5)
   self.assertEqual(w.relative_action_weights('robotwin',2150,2,55)[6],.5)
   self.assertEqual(w.relative_action_weights('robotwin',2150,3,55),[1.]*55)
 def test_fm_gradients_exact_left3_right2_padding_ignored(self):
  with tempfile.TemporaryDirectory()as t:
   w=SelectionWeights(self.config(Path(t)));v=torch.ones(1,2,55,requires_grad=True);mask=torch.zeros_like(v);mask[:,:,[*range(12),28,29]]=1
   action=torch.tensor([w.relative_action_weights('robotwin',2150,0,55)]);loss=fm_weighted_mean(v,torch.tensor([2.]),mask,action);loss.backward()
   self.assertAlmostEqual(loss.item(),2.5);self.assertAlmostEqual(v.grad[0,0,0].item(),3/28);self.assertAlmostEqual(v.grad[0,0,6].item(),2/28);self.assertEqual(v.grad[0,0,14].item(),0)
   self.assertAlmostEqual(v.grad[0,0,28].item(),3/28);self.assertAlmostEqual(v.grad[0,0,29].item(),2/28)
 def test_right_gradient_does_not_change_when_left_increased(self):
  v=torch.ones(1,1,2,requires_grad=True);mask=torch.ones_like(v)
  base=fm_weighted_mean(v,torch.tensor([2.]),mask,torch.tensor([[1.,1.]]));g0=torch.autograd.grad(base,v)[0]
  changed=fm_weighted_mean(v,torch.tensor([2.]),mask,torch.tensor([[1.5,1.]]));g1=torch.autograd.grad(changed,v)[0]
  self.assertEqual(g0[0,0,1],g1[0,0,1]);self.assertEqual(g1[0,0,0],g0[0,0,0]*1.5)
 def test_shared_visual_loss_remains_sample2(self):
  v=torch.ones(1,3,4);self.assertEqual(weighted_mean(v,torch.tensor([2.])).item(),2)
 def test_no_action_weights_is_identical(self):
  v=torch.arange(12,dtype=torch.float32).reshape(2,2,3);w=torch.tensor([1.,2.]);mask=torch.ones_like(v)
  self.assertTrue(torch.equal(weighted_mean(v,w,mask),fm_weighted_mean(v,w,mask)))
 def test_frozen_policy_preserves_arm_semantics(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);i=p/'input';i.mkdir();cfg=self.config(i);out=p/'output';out.mkdir();registry=freeze_registry(p/'empty',out,cfg);w=SelectionWeights(registry)
   self.assertEqual(w.relative_action_weights('robotwin',2150,0,55)[28],1.5)
 def test_invalid_layout_and_dimensions_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);cfg=self.config(p);c=json.loads(cfg.read_text());c['arm_action_dimensions']['right'].append(0);cfg.write_text(json.dumps(c))
   with self.assertRaises(ValueError):SelectionWeights(cfg)
  with self.assertRaises(ValueError):fm_weighted_mean(torch.ones(2,3,4),torch.ones(2),None,torch.ones(2,5))

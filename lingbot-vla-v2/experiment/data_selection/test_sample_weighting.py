"""CPU contracts: real autograd reducers and extracted production loss methods."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
from experiment.data_selection.weights import SelectionWeights
from experiment.data_selection.grasp_sequence import detect_arm, Settings

ROOT=Path(__file__).resolve().parents[2]

class ManifestTests(unittest.TestCase):
    def test_matching_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); rows=[dict(episode_index=1450,frame_index=3)]
            data=json.dumps(rows).encode();(p/'selected.json').write_bytes(data)
            c=dict(version=1,normalization='sample_count',dataset_root=str(p),dataset_name='robotwin',multiplier=2,selected_frames='selected.json',selected_frames_sha256=hashlib.sha256(data).hexdigest())
            (p/'config.json').write_text(json.dumps(c)); w=SelectionWeights(p/'config.json')
            self.assertEqual(w.weight('robotwin',1450,3),2)
            self.assertEqual(w.weight('robotwin',1450,4),1)
            self.assertEqual(w.weight('other',1450,3),1)
            (p/'selected.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'fingerprint'):SelectionWeights(p/'config.json')

class SequenceTests(unittest.TestCase):
    def trajectory(self):
        n=120;p=np.zeros((n,3));p[:,2]=.9
        p[:20,0]=np.linspace(.12,0,20)
        p[40:51,2]=np.linspace(.9,.8,11);p[51:,2]=.8
        p[80:96,2]=np.linspace(.8,.9,16);p[96:,2]=.9
        g=np.ones(n);g[60:71]=np.linspace(1,0,11);g[71:]=0
        return p,np.tile([0.,0.,-1.],(n,1)),g
    def test_aim_includes_approach_before_alignment(self):
        p,d,g=self.trajectory();events=detect_arm(p,d,g,15)
        self.assertEqual(len(events),1);e=events[0];self.assertTrue(e['accepted'],e)
        self.assertLess(e['stages']['aim'][0],e['stages']['align'][0])
        self.assertEqual(e['stages']['close'],[61,68])
    def test_no_lift_rejected(self):
        p,d,g=self.trajectory();p[80:,2]=.8
        self.assertEqual(detect_arm(p,d,g,15)[0]['rejection'],'no_lift_after_close')
    def test_downward_without_descent_rejected(self):
        p,d,g=self.trajectory();p[:,2]=.8
        self.assertFalse(detect_arm(p,d,g,15)[0]['accepted'])

try:
    import torch
    from lingbotvla.utils.sample_loss_weight import weighted_mean, fm_weighted_mean
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'requires CPU PyTorch')
class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile actual loss helper methods without importing transformer/CUDA models.
        path=ROOT/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py'
        tree=ast.parse(path.read_text());methods={}
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name in ('_emb_loss','_video_emb_loss','_video_cls_loss'):
                methods[node.name]=node
        env=dict(torch=torch,F=torch.nn.functional,weighted_mean=weighted_mean)
        exec(compile(ast.Module(body=list(methods.values()),type_ignores=[]),str(path),'exec'),env)
        cls.helpers=type('ActualHelpers',(),{name:env[name] for name in methods})
    def test_mask_and_unit_weight_parity(self):
        x=torch.arange(24,dtype=torch.float32).reshape(2,3,4).requires_grad_()
        mask=torch.ones_like(x);mask[1,:,2:]=0
        for m in [None,mask,mask*0]:
            a=weighted_mean(x,None,m);ga=torch.autograd.grad(a,x,retain_graph=True)[0]
            b=weighted_mean(x,torch.ones(2),m);gb=torch.autograd.grad(b,x,retain_graph=True)[0]
            torch.testing.assert_close(a,b);torch.testing.assert_close(ga,gb)
    def test_twofold_gradient_and_normalization(self):
        x=torch.ones(2,3,4,requires_grad=True);w=torch.tensor([2.,1.],requires_grad=True)
        loss=weighted_mean(x,w);loss.backward()
        torch.testing.assert_close(x.grad[0],2*x.grad[1]);self.assertIsNone(w.grad)
        self.assertEqual(loss.item(),1.5)
    def test_threefold_absolute_gradient(self):
        x=torch.ones(2,3,4,requires_grad=True)
        y=weighted_mean(x,torch.tensor([3.,1.]));y.backward()
        self.assertEqual(y.item(),2.)
        torch.testing.assert_close(x.grad[0],3*x.grad[1])
        baseline=torch.full_like(x.grad[1],1./24)
        torch.testing.assert_close(x.grad[1],baseline)

    def test_depth_video_and_cls_losses(self):
        h=self.helpers();h.config=types.SimpleNamespace(align_params={'video':dict(use_smooth_l1_loss=True,use_mse_loss=True,use_cosine_loss=True,cls_loss_type='mse_cosine')})
        p=torch.tensor([[[.3,.5],[.2,.7]],[[.3,.5],[.2,.7]]],requires_grad=True)
        t=torch.ones_like(p);w=torch.tensor([2.,1.])
        for f in [h._emb_loss,h._video_emb_loss,h._video_cls_loss]:
            target=t.mean(1) if f.__name__=='_video_cls_loss' else t
            pred=p.mean(1) if f.__name__=='_video_cls_loss' else p
            base=f(pred,target);base=base[0] if isinstance(base,tuple) else base
            unit=f(pred,target,sample_loss_weight=torch.ones(2));unit=unit[0] if isinstance(unit,tuple) else unit
            torch.testing.assert_close(base,unit)
            out=f(pred,target,sample_loss_weight=w);out=out[0] if isinstance(out,tuple) else out
            grad=torch.autograd.grad(out,p,retain_graph=True)[0]
            torch.testing.assert_close(grad[0],2*grad[1])
    def test_dataset_to_collation_weight_contract(self):
        path=ROOT/'lingbotvla/data/vla_data/base_dataset.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VLADataset')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='getitem')
        env=dict(torch=torch);exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
        rows=[dict(episode_index=torch.tensor(1450),frame_index=torch.tensor(i)) for i in [3,4]]
        class Matcher:
            stage_tables={}
            def weight(self,name,ep,frame):return 2. if frame==3 else 1.
            def relative_action_weights(self,*args):return None
        obj=types.SimpleNamespace(check_lerobot_item=lambda x:x,dataset=rows,use_future_image=False,
            feature_transform=types.SimpleNamespace(apply=lambda x:dict(state=torch.zeros(14),actions=torch.zeros(50,55))),transform=None,
            sample_selection_weights=Matcher(),data_name='robotwin')
        batch=torch.utils.data.default_collate([env['getitem'](obj,i) for i in range(2)])
        torch.testing.assert_close(batch['sample_loss_weight'],torch.tensor([2.,1.]))
        obj.sample_selection_weights=None
        self.assertNotIn('sample_loss_weight',env['getitem'](obj,0))

    def test_four_losses_and_regularizers_policy_contract(self):
        path=ROOT/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py'
        tree=ast.parse(path.read_text());policy=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LingbotVlaV2Policy')
        method=next(n for n in policy.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        method.returns=None
        env=dict(torch=torch,weighted_mean=weighted_mean,fm_weighted_mean=fm_weighted_mean)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
        class Fake:
            def forward(self,*a,**kw):
                weights=kw['sample_loss_weight'];v=kw['depth_targets']
                return (v,weighted_mean(v,weights),weighted_mean(v,weights),weighted_mean(v,weights),None,torch.tensor(3.),torch.tensor(5.),{},None,None,None)
        obj=types.SimpleNamespace(config=types.SimpleNamespace(loss_type='fm',action_dim=2),model=Fake())
        x=torch.ones(2,3,2,requires_grad=True)
        out=env['forward'](obj,None,None,x,None,None,x,depth_targets=x,sample_loss_weight=torch.tensor([2.,1.]))
        self.assertEqual(out[0].item(),14.) # four amplified data terms + unchanged 3 + 5
        grad=torch.autograd.grad(out[0],x)[0];torch.testing.assert_close(grad[0],2*grad[1])
        torch.testing.assert_close(out[6]['batch_mean_losses'],torch.ones(2))
        self.assertEqual(out[6]['router_z_loss'].item(),5.)
        # Exercise the real policy method with FM-only arm corrections.
        y=torch.ones(1,3,2,requires_grad=True)
        split=env['forward'](obj,None,None,y,None,None,y,depth_targets=y,sample_loss_weight=torch.tensor([2.]),action_loss_weight=torch.tensor([[1.5,1.]]))
        self.assertAlmostEqual(split[0].item(),16.5) # FM2.5 + three shared terms2 + regularizers8


if __name__=='__main__':unittest.main()

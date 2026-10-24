import json,hashlib,tempfile,types,unittest
from pathlib import Path
import numpy as np
import torch
from lingbotvla.losses.tcp_geometry import TCPGeometryRuntime,tcp_pose,relative_pose
from lingbotvla.losses.tcp_geometry_config import load_settings,freeze_settings
from lingbotvla.losses.tcp_geometry_data import PoseCache
from lingbotvla.losses.ee_fk import _unnormalize_action,build_action_layout
ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/'configs/kinematics/robotwin_aloha_agilex_ee.json'
LAYOUT={'left_arm':[0,6],'right_arm':[6,12],'gripper':[28,30]}
def settings():
    return dict(version=1,spec_path=str(SPEC),spec_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),tcp_offset_m=.12,huber_delta=.01,time_max=1.,absolute=dict(enabled=True,position_weight=1.,rotation_weight=.1,stages=[]),relative=dict(enabled=True,position_weight=.3,rotation_weight=.1,stages=['pre_contact']))
def fixture():
    runtime=TCPGeometryRuntime(settings(),LAYOUT,[]);q=torch.linspace(-.3,.6,110).reshape(1,2,55)
    truth=tcp_pose(runtime.fk.solver('cpu'),q[...,:6],q[...,6:12],.12)
    fields=dict(tcp_target_pose=truth.detach(),tcp_target_relative=relative_pose(truth).detach(),tcp_target_valid=torch.ones(1,2,2,dtype=torch.bool),tcp_geometry_gate=torch.ones(1,2,2,2),joint_mask=torch.ones_like(q,dtype=torch.bool),action_is_pad=torch.zeros(1,2,dtype=torch.bool),time=torch.tensor([.5]))
    return runtime,q,fields
class GeometryTests(unittest.TestCase):
    def test_zero_truth_loss_and_no_gripper_gradient(self):
        runtime,q,fields=fixture();pred=q.clone().requires_grad_();loss,metrics=runtime.compute(pred,**fields)
        self.assertLess(abs(loss.item()),1e-6);loss.backward();self.assertTrue(torch.isfinite(pred.grad).all());self.assertFalse(pred.grad[...,12:].any())
    def test_relative_gradient_reaches_both_arms_and_fm_velocity(self):
        runtime,q,fields=fixture();runtime.settings['absolute']['enabled']=False
        noise=torch.ones_like(q);v=(noise-q+.05).requires_grad_();loss,_=runtime.from_flow(q,noise,fields['time'],v,**{k:v for k,v in fields.items()if k!='time'});loss.backward()
        self.assertGreater(v.grad[...,:6].abs().sum().item(),0);self.assertGreater(v.grad[...,6:12].abs().sum().item(),0);self.assertFalse(v.grad[...,12:].any());self.assertTrue(torch.isfinite(v.grad).all())
    def test_padding_nan_and_empty_masks_have_zero_gradient(self):
        runtime,q,fields=fixture();q=q+.03;fields['action_is_pad'][:,-1]=True;q[:,-1]=float('nan');fields['tcp_target_pose'][:,-1]=float('nan');fields['tcp_target_relative'][:,-1]=float('nan');pred=q.requires_grad_();loss,_=runtime.compute(pred,**fields);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(pred.grad).all());self.assertFalse(pred.grad[:,-1].any())
        runtime,q,fields=fixture();fields['tcp_target_valid'].zero_();pred=q.requires_grad_();loss,_=runtime.compute(pred,**fields);loss.backward();self.assertEqual(loss.item(),0);self.assertFalse(pred.grad.any())
    def test_one_arm_invalid_suppresses_pair_and_that_arm(self):
        runtime,q,fields=fixture();fields['joint_mask'][...,6:12]=False;pred=(q+.05).requires_grad_();loss,metrics=runtime.compute(pred,**fields);loss.backward();self.assertEqual(metrics['tcp_geometry/paired_steps'].item(),0);self.assertFalse(pred.grad[...,6:12].any());self.assertGreater(pred.grad[...,:6].abs().sum().item(),0)
    def test_stage_gate_is_both_arms_and_not_fm20(self):
        runtime,q,fields=fixture();runtime.settings['absolute']['enabled']=False;fields['tcp_geometry_gate'][...,1,1]=0;pred=(q+.02).requires_grad_();loss,_=runtime.compute(pred,**fields);loss.backward();self.assertEqual(loss.item(),0);self.assertFalse(pred.grad.any())
    def test_relative_identity_global_transform_and_tcp_offset(self):
        runtime,q,fields=fixture();tcp=fields['tcp_target_pose'];rel=relative_pose(tcp);torch.testing.assert_close(tcp[...,0,:,:]@rel,tcp[...,1,:,:],atol=2e-6,rtol=2e-6)
        W=torch.eye(4);W[:3,3]=torch.tensor([1.,2.,3.]);theta=torch.tensor(.4);W[:2,:2]=torch.tensor([[theta.cos(),-theta.sin()],[theta.sin(),theta.cos()]])
        torch.testing.assert_close(relative_pose(W@tcp),rel,atol=2e-6,rtol=2e-6)
        ee=tcp_pose(runtime.fk.solver('cpu'),q[...,:6],q[...,6:12],0)
        torch.testing.assert_close(tcp[...,:3,3]-ee[...,:3,3],ee[...,:3,0]*.12)
    def test_fp32_inside_bfloat_autocast_and_runtime_not_registered(self):
        runtime,q,fields=fixture()
        with torch.autocast('cpu',dtype=torch.bfloat16):loss,_=runtime.compute((q+.03).bfloat16().requires_grad_(),**fields)
        self.assertEqual(loss.dtype,torch.float32);self.assertTrue(torch.isfinite(loss));self.assertEqual({b.dtype for b in runtime.fk.solver('cpu').buffers()},{torch.float32})
        model=torch.nn.Module();model.runtime=runtime;self.assertEqual(list(model.state_dict()),[])
    def test_inverse_normalization_has_grad_and_clipping_fails(self):
        value=torch.zeros(1,2,55,requires_grad=True);norm=[dict(start=0,end=12,norm_type='bounds_99_woclip',q01=[-2]*12,q99=[2]*12)];raw=_unnormalize_action(value,norm);raw[...,:12].sum().backward();self.assertTrue((value.grad[...,:12]>0).all())
        with self.assertRaises(ValueError):TCPGeometryRuntime(settings(),LAYOUT,[dict(norm_type='bounds_99')])
    def test_numpy_fk_rotation_position_parity(self):
        from tools.prepare_dual_tcp_pose_cache import fk
        runtime,q,fields=fixture();spec=json.loads(SPEC.read_text())
        for arm,columns in [('left',slice(0,6)),('right',slice(6,12))]:
            source=q[...,columns].reshape(-1,6);a=tcp_pose(runtime.fk.solver('cpu'),source,source,.12)[:,0 if arm=='left'else 1];reference=fk(source.numpy(),spec[arm]);np.testing.assert_allclose(a.detach().numpy(),reference,atol=5e-7,rtol=5e-6)

class CacheTests(unittest.TestCase):
    def fixture(self,root):
        runtime,q,fields=fixture();tcp=fields['tcp_target_pose'][0].numpy();state=tcp.copy();state[:,:,:3,3]+=.01
        np.savez(root/'poses.npz',episode_index=np.array([7,7]),frame_index=np.array([0,1]),valid=np.ones((2,2),bool),stage=np.array([[3,3],[4,6]]),action_tcp=tcp,action_rel=fields['tcp_target_relative'][0].numpy(),state_tcp=state,state_rel=relative_pose(torch.tensor(state)).numpy())
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();meta=dict(schema_version=1,relative_frame='left_tcp',rotation_format='matrix3x3',tcp_offset_m=.12,spec_sha256=sha(SPEC),arrays='poses.npz',arrays_sha256=sha(root/'poses.npz'));(root/'meta.json').write_text(json.dumps(meta))
        c=settings();c.update(dataset_name='robotwin',dataset_root=str(root),cache_metadata='meta.json',cache_metadata_sha256=sha(root/'meta.json'));(root/'config.json').write_text(json.dumps(c));return root/'config.json'
    def test_original_coordinates_offset_padding_and_gate(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);cache=PoseCache(self.fixture(root),'robotwin',root)
            a=cache.sample(7,0,2,[False,False]);b=cache.sample(7,0,2,[False,True],True)
            np.testing.assert_equal(a['tcp_target_pose'][1],cache.arrays['action_tcp'][1]);np.testing.assert_equal(b['tcp_target_pose'][0],cache.arrays['state_tcp'][1]);self.assertFalse(b['tcp_target_valid'][1].any());self.assertEqual(a['tcp_geometry_gate'][1,1,1],0)
            with self.assertRaises(ValueError):cache.sample(7,1,2,[False,False],True)
            self.assertFalse(cache.sample(100,0,2,[False,False])['tcp_target_valid'].any())
    def test_fingerprints_freeze_and_invalid_settings(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=self.fixture(root);frozen=freeze_settings(path,root/'frozen');(root/'poses.npz').write_bytes(b'broken');cache=PoseCache(frozen,'robotwin',root)
            with self.assertRaises(ValueError):PoseCache(path,'robotwin',root)
            c=json.loads(path.read_text());c['relative']['rotation_weight']=-1;path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):load_settings(path)

class PolicyIntegrationTests(unittest.TestCase):
    def test_actual_v2_policy_adds_geometry_separately_from_fm(self):
        from experiment.data_selection.test_stage_loss import StageLossTests
        method=StageLossTests().policy_method();seen={}
        class Core:
            def forward(self,*args,**kwargs):
                seen.update(kwargs);losses=args[5]**2
                return losses,0,0,0,None,0,torch.tensor(0.),{'_tcp_geometry_loss':args[5].sum()*3,'tcp_geometry/weighted_loss':torch.tensor(3.)},None,None,None
        x=torch.ones(1,2,2,requires_grad=True);obj=types.SimpleNamespace(config=types.SimpleNamespace(loss_type='fm',action_dim=2,tcp_geometry_config_path='enabled'),model=Core(),_tcp_geometry_configured=True,_tcp_geometry_runtime=object())
        fields=dict(tcp_target_pose=torch.eye(4).expand(1,2,2,4,4),tcp_target_relative=torch.eye(4).expand(1,2,4,4),tcp_target_valid=torch.ones(1,2,2,dtype=torch.bool),tcp_geometry_gate=torch.ones(1,2,2,2))
        result=method(obj,None,None,x,None,None,x,joint_mask=torch.ones_like(x),action_is_pad=torch.zeros(1,2,dtype=torch.bool),**fields)
        self.assertEqual(result[1].item(),1);self.assertEqual(result[0].item(),13);self.assertNotIn('_tcp_geometry_loss',result[6]);self.assertIn('geometry_context',seen);result[0].backward();self.assertTrue((x.grad>3).all())
    def test_disabled_policy_and_real_collator(self):
        from experiment.data_selection.test_stage_loss import StageCollatorTests,StageLossTests
        StageLossTests().test_actual_policy_uses_absolute_fm_and_routes_four_auxiliary_vectors()
        holder=StageCollatorTests();feature=holder.feature();h=2;feature.update(tcp_target_pose=torch.eye(4).expand(h,2,4,4),tcp_target_relative=torch.eye(4).expand(h,4,4),tcp_target_valid=torch.ones(h,2,dtype=torch.bool),tcp_geometry_gate=torch.ones(h,2,2))
        result=holder.method()(types.SimpleNamespace(state_features=[]),[feature,feature]);self.assertEqual(result['tcp_target_pose'].shape,(2,h,2,4,4))
        other=dict(feature);other.pop('tcp_geometry_gate')
        with self.assertRaises(ValueError):holder.method()(types.SimpleNamespace(state_features=[]),[feature,other])


class ActualFMForwardTests(unittest.TestCase):
    def method(self):
        import ast
        path=ROOT/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py';tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='FlowMatchingV2');node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='forward');node.returns=None
        env=dict(torch=torch,F=torch.nn.functional,make_att_2d_masks=lambda pad,att:pad[:,:,None]&pad[:,None,:]);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env);return env['forward']
    def test_actual_fm_velocity_backward_and_disabled_return_contract(self):
        runtime,q,fields=fixture();b,h,d=q.shape;projection=torch.nn.Linear(4,d)
        class Core:
            config=types.SimpleNamespace(align_params={},n_action_steps=h,action_fp32=False,use_cache=False)
            block_future_depth_to_action=False
            action_out_proj=projection
            qwenvl_with_expert=types.SimpleNamespace(forward=lambda **kw:((torch.zeros(b,1,4),torch.ones(b,h+1,4)),None,None))
            def embed_prefix(self,*args,**kw):return torch.zeros(b,1,4),torch.ones(b,1,dtype=torch.bool),torch.zeros(b,1,dtype=torch.bool),torch.zeros(3,b,1,dtype=torch.long),torch.zeros(b,1,dtype=torch.bool),[]
            def embed_suffix(self,*args):return None,torch.zeros(b,h+1,4),torch.ones(b,h+1,dtype=torch.bool),torch.zeros(b,h+1,dtype=torch.bool)
            def _block_suffix_to_future_video_if_enabled_(self,mask,**kw):return mask
            def _build_full_position_ids(self,*args):return torch.zeros(3,b,h+2,dtype=torch.long)
            def _moe_losses_and_metrics(self,*args):return torch.tensor(0.),torch.tensor(0.),{}
        method=self.method();core=Core();state=torch.zeros(b,55);noise=torch.ones_like(q);time=fields['time']
        result=method(core,None,None,None,None,state,q,noise,time,geometry_context=(runtime,{k:v for k,v in fields.items()if k!='time'}))
        self.assertEqual(len(result),11);value=result[7]['_tcp_geometry_loss'];self.assertTrue(value.requires_grad);value.backward();grad=projection.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad[:6].abs().sum().item(),0);self.assertGreater(grad[6:12].abs().sum().item(),0);self.assertFalse(grad[12:].any())
        plain=method(core,None,None,None,None,state,q,noise,time);self.assertEqual(len(plain),11);self.assertNotIn('_tcp_geometry_loss',plain[7]);torch.testing.assert_close(plain[0],result[0])


class CalibrationTests(unittest.TestCase):
    def test_archived_200_sapien_poses_with_explicit_tcp_offset(self):
        runtime,_,_=fixture();data=np.load(ROOT/'tests/ee_fk/evidence/2026-10-17/sapien_reference.npz',allow_pickle=False);q=torch.tensor(data['joints'],dtype=torch.float32);pred=tcp_pose(runtime.fk.solver('cpu'),q[:,:6],q[:,6:],.12).numpy();target=data['expected'].copy();target[...,:3,3]+=target[...,:3,0]*.12
        np.testing.assert_allclose(pred,target,atol=2e-6,rtol=2e-6)
    def test_finite_difference_relative_gradient(self):
        runtime,q,fields=fixture();runtime.settings['absolute']['enabled']=False;pred=(q+.03).requires_grad_();value,_=runtime.compute(pred,**fields);gradient=torch.autograd.grad(value,pred)[0]
        for index in [0,3,6,10]:
            plus=pred.detach().clone();minus=pred.detach().clone();plus[0,0,index]+=.001;minus[0,0,index]-=.001
            numerical=(runtime.compute(plus,**fields)[0]-runtime.compute(minus,**fields)[0])/.002
            self.assertAlmostEqual(gradient[0,0,index].item(),numerical.item(),delta=.002)


class WeightNormalizationTests(unittest.TestCase):
    def test_uniform_fractional_stage_gate_does_not_change_loss(self):
        runtime,q,fields=fixture();pred=q+.03;a=runtime.compute(pred,**fields)[0]
        fields['tcp_geometry_gate']*=.01;b=runtime.compute(pred,**fields)[0]
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

"""FP32 differentiable absolute TCP and dynamic right-in-left pose supervision."""
import json,math
import torch
import torch.nn.functional as F
from .ee_fk import EEFKRuntime,recover_clean_action,_unnormalize_action,_rotation_error,build_action_layout,build_action_norm_spec
from .tcp_geometry_config import load_settings,checked

def tcp_pose(solver,left,right,offset):
    result=solver(left,right);poses=[]
    for arm in ['left','right']:
        p,R=result[arm];p=p+R[..., :,0]*offset
        T=torch.eye(4,dtype=p.dtype,device=p.device).expand(p.shape[:-1]+(4,4)).clone();T[...,:3,:3]=R;T[...,:3,3]=p;poses.append(T)
    return torch.stack(poses,dim=-3)
def relative_pose(tcp):
    left,right=tcp.unbind(-3);R=left[...,:3,:3].transpose(-1,-2);out=torch.eye(4,dtype=tcp.dtype,device=tcp.device).expand(left.shape).clone();out[...,:3,:3]=R@right[...,:3,:3];out[...,:3,3]=(R@(right[...,:3,3]-left[...,:3,3]).unsqueeze(-1)).squeeze(-1);return out

def mean(value,weights):
    return (value*weights).sum()/weights.sum().clamp_min(1e-12)

class TCPGeometryRuntime:
    def __init__(self,settings,layout,norm):
        self.settings=settings;self.layout=layout;self.norm=norm
        if any(x.get('norm_type')in ['bounds_98','bounds_99','minmax']for x in norm):raise ValueError('geometry cache requires non-clipping absolute action normalization')
        self.fk=EEFKRuntime(json.loads(checked(settings['spec_path'],settings['spec_sha256']).read_text()))
    def from_flow(self,actions,noise,time,velocity,**fields):
        predicted=recover_clean_action(actions,noise,time,velocity)
        return self.compute(predicted,time=time,**fields)
    def compute(self,predicted,*,tcp_target_pose,tcp_target_relative,tcp_target_valid,tcp_geometry_gate,joint_mask,action_is_pad,time):
        with torch.autocast(device_type=predicted.device.type,enabled=False):
            b,h,d=predicted.shape
            if tcp_target_pose.shape!=(b,h,2,4,4)or tcp_target_relative.shape!=(b,h,4,4)or tcp_target_valid.shape!=(b,h,2)or tcp_geometry_gate.shape!=(b,h,2,2)or action_is_pad.shape!=(b,h)or joint_mask.shape!=predicted.shape:raise ValueError('TCP geometry batch shape mismatch')
            if time.shape!=(b,)or not torch.isfinite(time).all()or(time<0).any()or(time>1).any():raise ValueError('geometry FM time must be finite in [0,1]')
            valid=tcp_target_valid.bool() & ~action_is_pad.bool()[...,None] & (time.float()<=self.settings['time_max'])[:,None,None]
            for arm,key in enumerate(['left_arm','right_arm']):
                a,z=self.layout[key];valid[...,arm]&=joint_mask[...,a:z].bool().all(-1)
            gate=tcp_geometry_gate.float()
            if not torch.isfinite(gate).all()or (gate<0).any():raise ValueError('invalid geometry gate')
            # Invalid/padded values never enter nonlinear FK, including NaN padding.
            clean=predicted.float().clone()
            for arm,key in enumerate(['left_arm','right_arm']):
                a,z=self.layout[key];clean[...,a:z]=torch.where(valid[...,arm,None],clean[...,a:z],torch.zeros_like(clean[...,a:z]))
            raw=_unnormalize_action(clean,self.norm);l,z=self.layout['left_arm'];r,s=self.layout['right_arm'];poses=tcp_pose(self.fk.solver(predicted.device),raw[...,l:z],raw[...,r:s],self.settings['tcp_offset_m'])
            target=tcp_target_pose.detach().float();target=torch.where(valid[...,None,None],target,torch.eye(4,device=predicted.device))
            rel=relative_pose(poses);pair=valid.all(-1);truth=tcp_target_relative.detach().float();truth=torch.where(pair[...,None,None],truth,torch.eye(4,device=predicted.device))
            if not torch.isfinite(poses).all()or not torch.isfinite(target).all()or not torch.isfinite(truth).all():raise ValueError('nonfinite valid geometry')
            position=F.smooth_l1_loss(poses[...,:3,3],target[...,:3,3],beta=self.settings['huber_delta'],reduction='none').mean(-1)
            angle=_rotation_error(poses[...,:3,:3],target[...,:3,:3])
            rel_pos=F.smooth_l1_loss(rel[...,:3,3],truth[...,:3,3],beta=self.settings['huber_delta'],reduction='none').mean(-1);rel_angle=_rotation_error(rel[...,:3,:3],truth[...,:3,:3])
            ew=valid.float()*gate[...,0];rw=pair.float()*torch.minimum(gate[...,0,1],gate[...,1,1])
            ee_pos=sum(mean(position[...,a],ew[...,a])for a in [0,1])/2;ee_rot=sum(mean(angle[...,a],ew[...,a])for a in [0,1])/2
            rp,rr=mean(rel_pos,rw),mean(rel_angle,rw);total=poses.sum()*0.;metrics={}
            for name,values in [('absolute',(ee_pos,ee_rot)),('relative',(rp,rr))]:
                section=self.settings[name];value=section['position_weight']*values[0]+section['rotation_weight']*values[1]
                if section['enabled']:total=total+value
                metrics['tcp_geometry/'+name+'_position_loss']=values[0].detach();metrics['tcp_geometry/'+name+'_rotation_loss']=values[1].detach();metrics['tcp_geometry/'+name+'_weighted_loss']=(value if section['enabled']else value*0).detach()
            for arm,name in enumerate(['left','right']):
                metrics['tcp_geometry/'+name+'_position_cm']=mean((poses[...,arm,:3,3]-target[...,arm,:3,3]).norm(dim=-1)*100,valid[...,arm].float()).detach()
                metrics['tcp_geometry/'+name+'_rotation_deg']=mean(angle[...,arm]*180/math.pi,valid[...,arm].float()).detach()
            metrics['tcp_geometry/relative_position_cm']=mean((rel[...,:3,3]-truth[...,:3,3]).norm(dim=-1)*100,pair.float()).detach();metrics['tcp_geometry/relative_rotation_deg']=mean(rel_angle*180/math.pi,pair.float()).detach()
            metrics['tcp_geometry/paired_steps']=rw.gt(0).sum().detach();metrics['tcp_geometry/weighted_loss']=total.detach()
            return total,metrics

def configure_policy_geometry(policy,dataset,path):
    c=load_settings(path);policy._tcp_geometry_configured=True;policy._tcp_geometry_runtime=None
    if not any(c[name]['enabled']for name in ['absolute','relative']):return
    if getattr(policy.config,'ee_fk_loss_enabled',False):raise ValueError('new TCP geometry cannot be combined with the legacy EE objective')
    if getattr(policy.config,'gradient_accumulation_steps',1)!=1 or (torch.distributed.is_initialized() and torch.distributed.get_world_size()!=1):raise ValueError('TCP geometry currently requires single-device GAS=1 for exact normalization')
    if not hasattr(policy,'model')or type(policy).__name__!='LingbotVlaV2Policy':raise ValueError('TCP geometry requires VLA V2')
    transforms=list(getattr(dataset,'feature_transforms',{}).values())or[dataset.feature_transform]
    layouts=[build_action_layout(t)for t in transforms];norms=[build_action_norm_spec(t)for t in transforms]
    if any(v!=layouts[0]for v in layouts)or any(v!=norms[0]for v in norms):raise ValueError('geometry requires identical physical action layout/normalization across datasets')
    policy._tcp_geometry_runtime=TCPGeometryRuntime(c,layouts[0],norms[0]);policy.config.tcp_geometry_settings=c

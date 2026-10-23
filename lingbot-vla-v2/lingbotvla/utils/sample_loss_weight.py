"""Experimental sample-count loss weighting; no effect when weights are absent."""
import torch


def weighted_mean(values, weights=None, mask=None):
    if weights is None:
        return values.mean() if mask is None else (values * mask).sum() / mask.sum().clamp(min=1)
    if weights.ndim != 1 or len(weights) != len(values):
        raise ValueError("sample weights must match batch dimension")
    # Metadata weights are validated on CPU before collation. Never learn weights.
    w = weights.detach().to(device=values.device, dtype=torch.float32)
    w = w.reshape((-1,) + (1,) * (values.ndim - 1))
    if mask is None:
        per_sample = values.float().reshape(len(values), -1).mean(dim=1)
        return (per_sample * w.reshape(-1)).mean()
    valid = mask.float().expand_as(values)
    return (values.float() * valid * w).sum() / valid.sum().clamp(min=1)


def fm_weighted_mean(values, weights=None, mask=None, action_weights=None):
    """Apply per-action corrections without changing the unweighted denominator."""
    if action_weights is None:return weighted_mean(values,weights,mask)
    if values.ndim!=3 or action_weights.ndim!=2 or action_weights.shape!=(values.shape[0],values.shape[2]):
        raise ValueError("FM action weights must be batch by action dimension")
    correction=action_weights.detach().to(device=values.device,dtype=torch.float32)
    return weighted_mean(values.float()*correction[:,None,:],weights,mask)


def normalized_sample_weights(weights):
    """Adapt weight-sum normalization to fixed-size auxiliary feature reductions."""
    if weights.ndim!=1:raise ValueError("auxiliary stage weights must be batch vectors")
    w=weights.detach().float()
    return w*len(w)/w.sum().clamp(min=torch.finfo(torch.float32).tiny)


def stage_valid_mask(values,joint_mask,action_is_pad):
    if values.ndim!=3 or action_is_pad is None or action_is_pad.shape!=values.shape[:2]:
        raise ValueError("stage FM requires batch by horizon padding")
    joint=torch.ones_like(values,dtype=torch.bool)if joint_mask is None else joint_mask.bool().expand_as(values)
    return joint & ~action_is_pad.bool()[:,:,None]


def stage_fm_mean(values,stage_weights,joint_mask,action_is_pad):
    if stage_weights.shape!=values.shape:raise ValueError("stage FM weights must match every action element")
    valid=stage_valid_mask(values,joint_mask,action_is_pad)
    effective=stage_weights.detach().float()*valid
    return (values.float()*effective).sum()/effective.sum().clamp(min=torch.finfo(torch.float32).tiny)


def stage_fm_metrics(values,weights,joint_mask,action_is_pad,stage_ids=None,arm_index=None):
    from experiment.data_selection.full_stages import STAGES
    valid=stage_valid_mask(values,joint_mask,action_is_pad);v=values.float();w=weights.detach().float();count=valid.sum().clamp(min=1);den=(w*valid).sum().clamp(min=torch.finfo(torch.float32).tiny)
    result={'stage_weight/fm_unweighted':((v*valid).sum()/count).detach(),'stage_weight/fm_weighted':stage_fm_mean(values,weights,joint_mask,action_is_pad).detach(),'stage_weight/fm_valid_elements':valid.sum().detach(),'stage_weight/fm_padded_time_fraction':action_is_pad.float().mean().detach(),'stage_weight/fm_mean_weight':((w*valid).sum()/count).detach()}
    for value in (1.,2.,3.,5.):result['stage_weight/fm_weight_'+str(int(value))+'_fraction']=((valid & (w==value)).sum()/count).detach()
    if stage_ids is not None and arm_index is not None:
        if stage_ids.shape!=(*values.shape[:2],2)or arm_index.shape!=(values.shape[0],values.shape[2]):raise ValueError('stage metric identity shapes')
        for sid,name in enumerate(STAGES):
            phase=((stage_ids[:,:,0]==sid)[:,:,None] & (arm_index==0)[:,None,:]) | ((stage_ids[:,:,1]==sid)[:,:,None] & (arm_index==1)[:,None,:])
            mask=phase & valid;n=mask.sum()
            result['stage_weight/'+name+'/elements']=n.detach()
            result['stage_weight/'+name+'/fm_unweighted']=((v*mask).sum()/n.clamp(min=1)).detach()
            result['stage_weight/'+name+'/fm_contribution']=((v*w*mask).sum()/den).detach()
    return result

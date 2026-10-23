"""Build tensor-ready stage metadata after the real transform; no Torch import."""
from .full_stages import derive_arm_layout

def build_stage_fields(policy,data_name,episode,frame,entry,feature_transform):
    if not policy.stage_tables:return None
    horizon,width=entry['actions'].shape
    pad=entry['action_is_pad'].bool().tolist()
    if len(pad)!=horizon:raise ValueError('action padding horizon mismatch')
    table=policy.stage_tables.get(episode)if data_name==policy.dataset_name else None
    offset=int(bool(feature_transform.actions_convert_from_state))
    if table:
        active=entry['action_joint_mask'].nonzero().flatten().tolist()
        layout=derive_arm_layout(feature_transform.feature_config,feature_transform.key_mapping,active,table.table['source_action_channels'])
        result=table.sample(episode,frame,horizon,width,layout,offset)
        if result['valid']!=[not v for v in pad]:raise ValueError('stage temporal mask differs from actual dataset action padding')
    else:
        scalar=policy.weight(data_name,episode,frame);correction=policy.relative_action_weights(data_name,episode,frame,width)or [1.]*width
        future_k=horizon-1-offset;future_valid=(not pad[future_k])if future_k>=0 else True
        result=dict(fm=[[scalar*v if not pad[k]else 0. for v in correction]for k in range(horizon)],ids=[[-1,-1]for _ in range(horizon)],arm_index=[-1]*width,current_depth=scalar,current_video=scalar,future_depth=scalar if future_valid else 0.,future_video=scalar if future_valid else 0.)
    return dict(stage_fm_weight=result['fm'],stage_ids=result['ids'],stage_arm_index=result['arm_index'],stage_current_depth_weight=result['current_depth'],stage_future_depth_weight=result['future_depth'],stage_current_video_weight=result['current_video'],stage_future_video_weight=result['future_video'])

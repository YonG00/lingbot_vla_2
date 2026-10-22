"""Phase-balanced calibration anchors; original action trajectories remain intact."""
from dataclasses import dataclass,field,asdict
from collections import Counter
import math
import numpy as np
from .grasp_sequence import Settings as GraspSettings,select_rows
from .bowl_targeted import TargetedSettings,select_targeted

STAGES=('aim','align','descend','settle','close')
@dataclass(frozen=True)
class CalibrationSettings:
    stage_quota:dict=field(default_factory=lambda:dict(aim=13,align=6,descend=10,settle=6,close=8))
    target_fraction:float=.30
    fraction_tolerance:float=.02
    targeted:dict=field(default_factory=lambda:asdict(TargetedSettings(include_far_grasp_core=False)))
    def validate(self):
        if set(self.stage_quota)!=set(STAGES)or any(type(v)is not int or v<2 for v in self.stage_quota.values()):raise ValueError('positive per-stage endpoint quotas required')
        for key in ['target_fraction','fraction_tolerance']:
            value=getattr(self,key)
            if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or not 0<value<1:raise ValueError('invalid '+key)
        TargetedSettings(**self.targeted).validate()
        if self.targeted.get('include_far_grasp_core',True):raise ValueError('far core must not override all-grasp uniform anchors')


def uniform_frames(start,end,quota):
    if end<start:return []
    return np.unique(np.rint(np.linspace(start,end,min(quota,end-start+1))).astype(int)).tolist()


def calibrate_events(grasp_events,extra_rows,settings=CalibrationSettings()):
    settings.validate();chosen={};diagnostics=[]
    def add(ep,frame,label):
        key=(int(ep),int(frame));target=chosen.setdefault(key,dict(episode_index=key[0],frame_index=key[1],labels=[]))
        if label not in target['labels']:target['labels'].append(label)
    for event in grasp_events:
        if not event['accepted']:continue
        anchors={}
        for stage in STAGES:
            frames=uniform_frames(*event['stages'][stage],settings.stage_quota[stage]);anchors[stage]=frames
            for frame in frames:add(event['episode_index'],frame,dict(stage='grasp_'+stage,arm=event['arm'],closure_frame=event['closure_crossing_frame']))
        diagnostics.append(dict(episode_index=event['episode_index'],arm=event['arm'],closure_frame=event['closure_crossing_frame'],anchors=anchors))
    for row in extra_rows:
        for label in row['labels']:
            if label['stage']in ('far_grasp_position','second_place_height_band'):add(row['episode_index'],row['frame_index'],label)
    return diagnostics,[chosen[key]for key in sorted(chosen)]


def select_calibration(rows,spec,fps,settings=CalibrationSettings()):
    settings.validate()
    grasp_summary,events,_=select_rows(rows,spec,fps,GraspSettings())
    narrow_summary,narrow_events,narrow_rows=select_targeted(rows,spec,fps,TargetedSettings(**settings.targeted))
    diagnostics,selected=calibrate_events(events,narrow_rows,settings)
    fraction=len(selected)/len(rows)
    if abs(fraction-settings.target_fraction)>settings.fraction_tolerance:
        raise ValueError('calibration coverage outside declared tolerance: '+str(fraction))
    summary=dict(task='stack_bowls_two',episodes=grasp_summary['episodes'],total_frames=len(rows),grasp_events=grasp_summary['accepted_sequences'],selected_frames=len(selected),selected_fraction=fraction,target_fraction=settings.target_fraction,fraction_tolerance=settings.fraction_tolerance,stage_quota=settings.stage_quota,stage_frames=dict(Counter(label['stage']for row in selected for label in row['labels'])),qualified_far_grasps=narrow_summary['qualified_far_grasps'],second_placements_with_height_samples=narrow_summary['second_placements_with_height_samples'],raw_trajectories_unchanged=True,kinematic_proxy_only=True,object_pose_verified=False,collision_verified=False)
    return summary,dict(grasp_anchors=diagnostics,targeted_diagnostics=narrow_events),selected

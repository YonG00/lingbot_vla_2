"""CPU kinematic proxy for grasp-confirmed stapler placement to episode end."""
from dataclasses import dataclass,asdict
from collections import Counter
import math
import numpy as np
from .grasp_sequence import forward_kinematics,_runs

@dataclass(frozen=True)
class StaplerPlaceSettings:
    open_threshold:float=.95
    closed_threshold:float=.2
    lift_min_m:float=.08
    preplace_distance_min_m:float=.08
    preplace_distance_max_m:float=.12
    preplace_height_min_m:float=.06
    stable_speed_mps:float=.025
    stable_frames:int=2
    aiming_context_seconds:float=1/3
    max_release_drift_m:float=.015
    min_descent_m:float=.05
    reverse_tolerance_m:float=.002
    forward_fraction:float=.7
    max_perpendicular_m:float=.025
    action_horizon:int=50
    def validate(self):
        for key,value in asdict(self).items():
            if key in ('stable_frames','action_horizon'):
                if type(value)is not int or value<1:raise ValueError('invalid '+key)
            elif isinstance(value,bool)or not isinstance(value,(float,int))or not math.isfinite(value)or value<=0:raise ValueError('invalid '+key)
        if not 0<self.closed_threshold<.5<self.open_threshold<=1 or self.preplace_distance_min_m>=self.preplace_distance_max_m or self.forward_fraction>1:raise ValueError('threshold order')

def detect_episode(position,gripper,fps,settings=StaplerPlaceSettings()):
    settings.validate();p=np.asarray(position,float);g=np.asarray(gripper,float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<3 or g.shape!=(len(p),)or not np.isfinite(p).all()or not np.isfinite(g).all()or np.any((g<0)|(g>1))or not math.isfinite(fps)or fps<=0:raise ValueError('trajectory/fps contract')
    closes=np.flatnonzero((g[1:]<=.5)&(g[:-1]>.5))+1;opens=np.flatnonzero((g[1:]>=.5)&(g[:-1]<.5))+1
    events=[]
    for crossing in closes:
        event=dict(closure_crossing_frame=int(crossing),accepted=False)
        release=opens[opens>crossing]
        if len(release)!=1 or len(closes)!=1:event['rejection']='ambiguous_close_release';events.append(event);continue
        op=int(release[0]);closed=np.flatnonzero(g[crossing:op]<=settings.closed_threshold);opened=np.flatnonzero(g[op:]>=settings.open_threshold)
        if not len(closed)or not len(opened):event['rejection']='incomplete_gripper_sequence';events.append(event);continue
        ce=int(crossing+closed[0]);rs=int(crossing+closed[-1]+1);re=int(op+opened[0]);target=np.median(p[rs:re+1],axis=0)
        lifted=np.flatnonzero(p[ce:rs,2]-p[ce,2]>=settings.lift_min_m)
        if not len(lifted):event['rejection']='no_confirmed_lift';events.append(event);continue
        lift=ce+int(lifted[0]);dist=np.linalg.norm(p-target,axis=1);speed=np.linalg.norm(np.gradient(p,axis=0)*fps,axis=1)
        mask=(dist>=settings.preplace_distance_min_m)&(dist<=settings.preplace_distance_max_m)&(p[:,2]-target[2]>=settings.preplace_height_min_m)&(speed<=settings.stable_speed_mps)&(g<=settings.closed_threshold)
        runs=[(a+lift,b+lift)for a,b in _runs(mask[lift:rs])if b-a+1>=settings.stable_frames]
        if not runs:event['rejection']='no_stable_preplace';events.append(event);continue
        ready_start,ready_end=runs[-1];axis=target-p[ready_end];axis/=np.linalg.norm(axis)
        delta=p[ready_end:rs+1]-target;perp=np.linalg.norm(delta-np.outer(delta@axis,axis),axis=1);progress=np.diff(p[ready_end:rs+1],axis=0)@axis
        forward=float(np.mean(progress>=-settings.reverse_tolerance_m));drop=float(p[ready_end,2]-target[2]);drift=float(np.linalg.norm(p[rs:re+1]-target,axis=1).max())
        reasons=[]
        if np.max(g[ce:rs])>settings.closed_threshold:reasons.append('opened_before_place')
        if forward<settings.forward_fraction or perp.max()>settings.max_perpendicular_m:reasons.append('invalid_final_approach')
        if drop<settings.min_descent_m:reasons.append('insufficient_placement_descent')
        if drift>settings.max_release_drift_m:reasons.append('moving_while_releasing')
        if np.min(g[re:])<settings.open_threshold:reasons.append('reclosed_after_release')
        start=max(lift,ready_start-math.ceil(settings.aiming_context_seconds*fps));end=len(g)-1
        event.update(accepted=not reasons,rejection=reasons,close_end_frame=ce,lift_confirm_frame=lift,preplace_start=int(ready_start),preplace_end=int(ready_end),release_start_frame=rs,release_end_frame=re,start_frame=int(start),end_frame=end,broad_postgrasp_start_frame=lift,max_lift_m=float(p[ce:rs,2].max()-p[ce,2]),preplace_height_above_release_m=float(p[ready_start,2]-target[2]),preplace_distance_m=float(dist[ready_start]),release_drift_m=drift,descent_m=drop,forward_fraction=forward,perpendicular_max_m=float(perp.max()))
        events.append(event)
    return events

def select_stapler_placement(rows,spec,fps,settings=StaplerPlaceSettings()):
    grouped={}
    for row in rows:grouped.setdefault(int(row['episode_index']),[]).append(row)
    all_events=[];selected={};broad={};remaining=[]
    for ep,items in sorted(grouped.items()):
        items.sort(key=lambda r:r['frame_index']);n=len(items)
        if [r['frame_index']for r in items]!=list(range(n)):raise ValueError('missing/duplicate frames')
        if any(abs(r['timestamp']-r['frame_index']/fps)>.01/fps for r in items):raise ValueError('timestamp mismatch')
        state=np.asarray([r['observation.state']for r in items],float)
        if state.shape!=(n,14)or not np.isfinite(state).all():raise ValueError('expected finite 14D state')
        episode_events=[]
        for arm,joints,j in [('left',slice(0,6),6),('right',slice(7,13),13)]:
            position,_=forward_kinematics(state[:,joints],spec[arm])
            for event in detect_episode(position,state[:,j],fps,settings):event.update(episode_index=ep,arm=arm);episode_events.append(event)
        accepted=[e for e in episode_events if e['accepted']]
        if len(accepted)!=1:
            for e in accepted:e.update(accepted=False,rejection=['episode_requires_one_accepted_grasp'])
        all_events.extend(episode_events)
        for event in episode_events:
            if not event['accepted']:continue
            for frame in range(event['start_frame'],event['end_frame']+1):
                if frame<event['preplace_start']:stage='place_aim_context'
                elif frame<=event['preplace_end']:stage='preplace_hold'
                elif frame<event['release_start_frame']:stage='place_descend_settle'
                elif frame<=event['release_end_frame']:stage='place_release'
                else:stage='episode_tail'
                selected[(ep,frame)]=dict(episode_index=ep,frame_index=frame,labels=[dict(arm=event['arm'],stage=stage)])
                remaining.append(min(settings.action_horizon,n-frame))
            for frame in range(event['broad_postgrasp_start_frame'],n):broad[(ep,frame)]=dict(episode_index=ep,frame_index=frame)
    accepted=[e for e in all_events if e['accepted']]
    summary=dict(task='move_stapler_pad',episodes=len(grouped),total_frames=len(rows),candidate_grasps=len(all_events),accepted_sequences=len(accepted),arm_counts=dict(Counter(e['arm']for e in accepted)),rejections=dict(Counter(reason for e in all_events if not e['accepted']for reason in (e.get('rejection')if isinstance(e.get('rejection'),list)else[e.get('rejection')]))),selected_frames=len(selected),selected_fraction=len(selected)/len(rows),broad_postgrasp_frames=len(broad),broad_postgrasp_fraction=len(broad)/len(rows),stage_frames=dict(Counter(row['labels'][0]['stage']for row in selected.values())),anchors_reaching_episode_end=sum(v<settings.action_horizon for v in remaining),mean_real_future_frames=float(np.mean(remaining))if remaining else 0.,kinematic_proxy_only=True,object_pose_verified=False,contact_verified=False,subsampled=False)
    return summary,dict(events=all_events,broad_reference_rows=[broad[k]for k in sorted(broad)]),[selected[k]for k in sorted(selected)]

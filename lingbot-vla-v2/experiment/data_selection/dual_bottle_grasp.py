"""Task-specific horizontal reach/grasp/lift selector; CPU kinematic proxy."""
from dataclasses import dataclass,asdict
from collections import Counter
import math
import numpy as np
from .grasp_sequence import forward_kinematics,_runs

@dataclass(frozen=True)
class Settings:
    open_threshold:float=.95
    closed_threshold:float=.2
    pregrasp_min_m:float=.06
    pregrasp_max_m:float=.10
    horizontal_degrees:float=15.
    stable_speed_mps:float=.03
    stable_frames:int=2
    perpendicular_m:float=.025
    arrival_m:float=.015
    closing_drift_m:float=.015
    forward_fraction:float=.7
    reverse_m:float=.002
    aim_search_seconds:float=4.
    aim_min_progress_m:float=.02
    aim_reverse_fraction:float=.3
    lift_min_m:float=.08
    lift_timeout_seconds:float=3.
    def validate(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<=0:raise ValueError('invalid '+key)
        if type(self.stable_frames)is not int or self.stable_frames<2 or not 0<self.closed_threshold<.5<self.open_threshold<=1 or self.pregrasp_min_m>=self.pregrasp_max_m or self.horizontal_degrees>=90 or self.forward_fraction>1 or self.aim_reverse_fraction>1:raise ValueError('threshold order')

def detect_arm(p,d,g,fps,s=Settings()):
    s.validate();p=np.asarray(p,float);d=np.asarray(d,float);g=np.asarray(g,float)
    if p.ndim!=2 or p.shape[1]!=3 or d.shape!=p.shape or g.shape!=(len(p),)or len(p)<3 or not all(np.isfinite(v).all()for v in [p,d,g])or np.any((g<0)|(g>1))or not math.isfinite(fps)or fps<=0:raise ValueError('trajectory contract')
    norm=np.linalg.norm(d,axis=1)
    if np.any(norm<1e-8):raise ValueError('invalid direction')
    d=d/norm[:,None];angle=np.degrees(np.arcsin(np.clip(abs(d[:,2]),0,1)))
    crossings=np.flatnonzero((g[1:]<=.5)&(g[:-1]>.5))+1
    if len(crossings)!=1:return [dict(accepted=False,rejection='requires_one_closure')]
    center=int(crossings[0]);e=dict(accepted=False,closure_crossing_frame=center)
    def reject(reason):e['rejection']=reason;return [e]
    opened=np.flatnonzero(g[:center]>=s.open_threshold);closed=np.flatnonzero(g[center:]<=s.closed_threshold)
    if not len(opened)or not len(closed):return reject('incomplete_gripper_sequence')
    cs=int(opened[-1])+1;ce=center+int(closed[0]);target=np.median(p[cs:ce+1],axis=0);axis=d[center]
    delta=target-p;remaining=delta@axis;perp=np.linalg.norm(delta-remaining[:,None]*axis,axis=1);speed=np.linalg.norm(np.gradient(p,axis=0)*fps,axis=1)
    aligned=(remaining>=s.pregrasp_min_m)&(remaining<=s.pregrasp_max_m)&(perp<=s.perpendicular_m)&(angle<=s.horizontal_degrees)&(speed<=s.stable_speed_mps)&(g>=s.open_threshold)
    candidates=[(int(a),int(b))for a,b in _runs(aligned[:cs])if b-a+1>=s.stable_frames]
    if not candidates:return reject('no_horizontal_pregrasp')
    align_start,align_end=candidates[-1]
    arrival=np.flatnonzero(np.linalg.norm(p[align_end+1:cs]-target,axis=1)<=s.arrival_m)
    if not len(arrival):return reject('no_arrival_before_closure')
    arrival=align_end+1+int(arrival[0]);forward=float(np.mean(np.diff(p[align_end:arrival+1],axis=0)@axis>=-s.reverse_m))
    if forward<s.forward_fraction or perp[align_end:arrival+1].max()>s.perpendicular_m:return reject('invalid_approach_path')
    if angle[align_end:ce+1].max()>s.horizontal_degrees:return reject('nonhorizontal_grasp')
    if g[align_end:arrival+1].min()<s.open_threshold:return reject('premature_closing')
    if np.linalg.norm(p[cs:ce+1]-target,axis=1).max()>s.closing_drift_m:return reject('moving_during_closure')
    stop=min(len(g),ce+math.ceil(s.lift_timeout_seconds*fps)+1);lift=np.flatnonzero(p[ce:stop,2]-p[ce,2]>=s.lift_min_m)
    if not len(lift):return reject('no_confirmed_lift')
    end=ce+int(lift[0])
    if g[ce:end+1].max()>.5:return reject('reopened_before_lift')
    lower=max(0,align_start-math.ceil(s.aim_search_seconds*fps));not_open=np.flatnonzero(g[lower:align_start]<s.open_threshold)
    if len(not_open):lower+=int(not_open[-1])+1
    distance=np.linalg.norm(p-p[align_end],axis=1);start=align_start
    for candidate in range(lower,align_start):
        if distance[candidate]-distance[align_end]>=s.aim_min_progress_m and np.mean(np.diff(distance[candidate:align_end+1])>s.reverse_m)<=s.aim_reverse_fraction:start=candidate;break
    stages=dict(reach_aim=[start,align_start-1],horizontal_align=[align_start,align_end],forward_approach=[align_end+1,arrival],settle=[arrival+1,cs-1],close=[cs,ce],lift=[ce+1,end])
    e.update(accepted=True,rejection=None,start_frame=start,end_frame=end,close_start_frame=cs,close_end_frame=ce,stages=stages,horizontal_deviation_degrees=float(angle[center]),approach_forward_fraction=forward,approach_perpendicular_max_m=float(perp[align_end:arrival+1].max()),lift_m=float(p[end,2]-p[ce,2]));return [e]

def select_rows(rows,spec,fps,s=Settings()):
    grouped={}
    for row in rows:grouped.setdefault(row['episode_index'],[]).append(row)
    events=[];selected={}
    for ep,items in sorted(grouped.items()):
        items.sort(key=lambda x:x['frame_index']);n=len(items)
        if [v['frame_index']for v in items]!=list(range(n))or any(abs(v['timestamp']-v['frame_index']/fps)>.01/fps for v in items):raise ValueError('episode frame/timestamp contract')
        state=np.asarray([v['observation.state']for v in items],float)
        if state.shape!=(n,14):raise ValueError('14D states required')
        episode_events=[]
        for arm,j in [('left',0),('right',7)]:
            p,d=forward_kinematics(state[:,j:j+6],spec[arm])
            for event in detect_arm(p,d,state[:,j+6],fps,s):event.update(episode_index=ep,arm=arm);episode_events.append(event)
        # Keep only paired successes, so one arm cannot silently disappear.
        accepted=[e for e in episode_events if e['accepted']]
        if len(accepted)!=2 or {e['arm']for e in accepted}!={'left','right'}:
            for event in accepted:event.update(accepted=False,rejection='missing_other_arm')
        events.extend(episode_events)
        for event in episode_events:
            if not event['accepted']:continue
            for stage,(a,b)in event['stages'].items():
                for frame in range(a,b+1):
                    v=selected.setdefault((ep,frame),dict(episode_index=ep,frame_index=frame,labels=[]));v['labels'].append(dict(arm=event['arm'],stage=stage))
    result=[selected[k]for k in sorted(selected)];accepted=[e for e in events if e['accepted']]
    summary=dict(task='pick_dual_bottles',episodes=len(grouped),total_frames=len(rows),accepted_sequences=len(accepted),arm_sequences=dict(Counter(e['arm']for e in accepted)),paired_episodes=len({e['episode_index']for e in accepted}),rejections=dict(Counter(e['rejection']for e in events if not e['accepted'])),selected_frames=len(result),selected_fraction=len(result)/len(rows),both_arm_frames=sum({v['arm']for v in row['labels']}=={'left','right'}for row in result),stage_arm_frames=dict(Counter(v['stage']for row in result for v in row['labels'])),subsampled=False,kinematic_proxy_only=True,contact_verified=False)
    return summary,events,result

"""Offline motion-sequence proxy: align, descend, close, then lift; NumPy only."""
from dataclasses import dataclass,asdict
import math
import numpy as np

@dataclass(frozen=True)
class Settings:
    downward_degrees:float=30.
    open_threshold:float=.95
    closed_threshold:float=.2
    closure_crossing:float=.5
    search_before_seconds:float=4.
    closing_timeout_seconds:float=2.
    alignment_hold_seconds:float=.13
    alignment_context_seconds:float=.5
    aiming_search_seconds:float=4.
    aiming_min_progress_m:float=.02
    aiming_reverse_fraction:float=.3
    alignment_speed_mps:float=.03
    pregrasp_distance_min_m:float=.06
    pregrasp_distance_max_m:float=.16
    perpendicular_tolerance_m:float=.025
    target_tolerance_m:float=.015
    minimum_descent_m:float=.05
    reverse_tolerance_m:float=.002
    forward_fraction:float=.7
    close_position_tolerance_m:float=.015
    lift_timeout_seconds:float=3.
    lift_confirmation_m:float=.03
    selected_stages:tuple=('aim','align','descend','settle','close')

    def validate(self):
        for key,value in asdict(self).items():
            if key=='selected_stages':continue
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:raise ValueError('invalid '+key)
        if not 0<self.closed_threshold<self.closure_crossing<self.open_threshold<=1:raise ValueError('gripper threshold order')
        if not self.pregrasp_distance_min_m<self.pregrasp_distance_max_m:raise ValueError('pregrasp range')
        if self.forward_fraction>1 or self.aiming_reverse_fraction>1 or self.downward_degrees>=90:raise ValueError('invalid fraction or angle')
        if not self.selected_stages or set(self.selected_stages)-{'aim','align','descend','settle','close','lift'}:raise ValueError('invalid selected stages')


def _rpy(v):
    r,p,y=v;cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])@np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])

def _T(r,p):
    t=np.broadcast_to(np.eye(4),r.shape[:-2]+(4,4)).copy();t[...,:3,:3]=r;t[...,:3,3]=p;return t

def forward_kinematics(q,spec,tcp_offset_m=.12):
    q=np.asarray(q,float)
    if q.ndim!=2 or q.shape[1]!=6 or not np.isfinite(q).all():raise ValueError('expected finite Nx6 joints')
    if len(spec['joints'])!=6:raise ValueError('expected six-joint chain')
    t=np.broadcast_to(_T(_rpy(spec['root']['rpy']),spec['root']['translation']),(len(q),4,4)).copy()
    for i,j in enumerate(spec['joints']):
        axis=np.asarray(j['axis'],float);norm=np.linalg.norm(axis)
        if not np.isfinite(norm)or norm<=0:raise ValueError('invalid axis')
        axis/=norm;x,y,z=axis;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        c=np.cos(q[:,i])[:,None,None];sn=np.sin(q[:,i])[:,None,None]
        rotation=c*np.eye(3)+(1-c)*np.outer(axis,axis)+sn*k
        t=t@_T(_rpy(j['origin_rpy']),j['origin_xyz'])@_T(rotation,np.zeros((len(q),3)))
    t=t@_T(_rpy(spec['tip_rpy']),spec['tip_xyz']);direction=t[:,:3,0]
    # Verified spec is get_*_ee_pose; RoboTwin TCP is +0.12m along its +X.
    return t[:,:3,3]+tcp_offset_m*direction,direction


def _runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1))


def detect_arm(position,direction,gripper,fps,settings=Settings()):
    settings.validate()
    if not isinstance(fps,(float,int)) or not math.isfinite(fps) or fps<=0:raise ValueError('invalid fps')
    p=np.asarray(position,float);d=np.asarray(direction,float);g=np.asarray(gripper,float)
    if p.ndim!=2 or p.shape[1]!=3 or d.shape!=p.shape or g.shape!=(len(p),)or len(p)<2:raise ValueError('trajectory shapes')
    if not all(np.isfinite(v).all()for v in [p,d,g])or np.any((g<-.001)|(g>1.001)):raise ValueError('invalid trajectory values')
    norms=np.linalg.norm(d,axis=1)
    if np.any(norms<1e-8):raise ValueError('zero direction')
    d=d/norms[:,None];angle=np.rad2deg(np.arccos(np.clip(-d[:,2],-1,1)))
    speed=np.linalg.norm(np.gradient(p,axis=0)*fps,axis=1)
    crossings=np.flatnonzero((g[1:]<=settings.closure_crossing)&(g[:-1]>settings.closure_crossing))+1
    events=[]
    for center in crossings:
        result=dict(closure_crossing_frame=int(center),accepted=False,rejection=None)
        def reject(reason):result['rejection']=reason;events.append(result)
        lo=max(0,center-math.ceil(settings.closing_timeout_seconds*fps))
        open_frames=np.flatnonzero(g[lo:center]>=settings.open_threshold)
        if not len(open_frames):reject('no_open_before_close');continue
        close_start=lo+int(open_frames[-1])+1
        hi=min(len(g),close_start+math.ceil(settings.closing_timeout_seconds*fps)+1)
        closed_frames=np.flatnonzero(g[close_start:hi]<=settings.closed_threshold)
        if not len(closed_frames):reject('no_complete_close');continue
        close_end=close_start+int(closed_frames[0]);target=np.median(p[close_start:close_end+1],axis=0);axis=d[center]
        lower=max(0,close_start-math.ceil(settings.search_before_seconds*fps))
        delta=target-p;remaining=delta@axis;perp=np.linalg.norm(delta-remaining[:,None]*axis,axis=1)
        aligned=(remaining>=settings.pregrasp_distance_min_m)&(remaining<=settings.pregrasp_distance_max_m)&(perp<=settings.perpendicular_tolerance_m)&(angle<=settings.downward_degrees)&(speed<=settings.alignment_speed_mps)&(g>=settings.open_threshold)
        candidates=[(a+lower,b+lower)for a,b in _runs(aligned[lower:close_start])if b-a+1>=max(2,math.ceil(settings.alignment_hold_seconds*fps))]
        if not candidates:reject('no_stable_pregrasp_alignment');continue
        _,align_end=candidates[-1]
        align_start=max(lower,align_end-math.ceil(settings.alignment_context_seconds*fps)+1)
        arrival_frames=np.flatnonzero(np.linalg.norm(p[align_end+1:close_start]-target,axis=1)<=settings.target_tolerance_m)
        if not len(arrival_frames):reject('no_arrival_before_close');continue
        arrival=align_end+1+int(arrival_frames[0]);descent=slice(align_end,arrival+1)
        drop=float(p[align_end,2]-p[arrival,2]);progress=np.diff(p[descent],axis=0)@axis
        forward=float(np.mean(progress>=-settings.reverse_tolerance_m))
        result.update(alignment_frame=int(align_end),close_start_frame=int(close_start),close_end_frame=int(close_end),arrival_frame=int(arrival),descent_m=drop,forward_fraction=forward,downward_angle_at_close_deg=float(angle[center]),max_descent_perpendicular_error_m=float(perp[descent].max()))
        if drop<settings.minimum_descent_m:reject('insufficient_downward_descent');continue
        if forward<settings.forward_fraction or perp[descent].max()>settings.perpendicular_tolerance_m:reject('descent_not_aligned');continue
        if angle[descent].max()>settings.downward_degrees:reject('descent_not_downward');continue
        if np.min(g[align_end:arrival+1])<settings.open_threshold:reject('premature_closing_during_descent');continue
        if np.max(np.linalg.norm(p[close_start:close_end+1]-target,axis=1))>settings.close_position_tolerance_m:reject('moving_while_closing');continue
        stop=min(len(g),close_end+math.ceil(settings.lift_timeout_seconds*fps)+1)
        lifted=np.flatnonzero(p[close_end:stop,2]-target[2]>=settings.lift_confirmation_m)
        if not len(lifted):reject('no_lift_after_close');continue
        lift_end=close_end+int(lifted[0])
        if np.max(g[close_end:lift_end+1])>settings.closure_crossing:reject('reopened_before_lift');continue
        # Offline aiming reference is the demonstrated stable pregrasp pose.
        # Search only this closure's open segment; never bridge a previous grasp.
        aim_lower=max(0,align_start-math.ceil(settings.aiming_search_seconds*fps))
        not_open=np.flatnonzero(g[aim_lower:align_start]<settings.open_threshold)
        if len(not_open): aim_lower+=int(not_open[-1])+1
        distance=np.linalg.norm(p-p[align_end],axis=1)
        aim_start=align_start
        for candidate in range(aim_lower,align_start):
            changes=np.diff(distance[candidate:align_end+1])
            if distance[candidate]-distance[align_end]<settings.aiming_min_progress_m:continue
            if np.mean(changes>settings.reverse_tolerance_m)>settings.aiming_reverse_fraction:continue
            aim_start=candidate
            break
        stages=dict(aim=[int(aim_start),int(align_start-1)],align=[int(align_start),int(align_end)],descend=[int(align_end+1),int(arrival)],settle=[int(arrival+1),int(close_start-1)],close=[int(close_start),int(close_end)],lift=[int(close_end+1),int(lift_end)])
        result.update(accepted=True,rejection=None,lift_confirmation_frame=int(lift_end),stages=stages)
        events.append(result)
    return events


def select_rows(rows,spec,fps,settings=Settings()):
    from collections import defaultdict,Counter
    grouped=defaultdict(list)
    for row in rows:grouped[int(row['episode_index'])].append(row)
    events=[];selected={};total=0
    for episode,items in sorted(grouped.items()):
        items=sorted(items,key=lambda r:r['frame_index']);n=len(items);total+=n
        if [r['frame_index']for r in items]!=list(range(n)):raise ValueError('missing/duplicate frame indices')
        if any(abs(r['timestamp']-r['frame_index']/fps)>.01/fps for r in items):raise ValueError('timestamp/fps mismatch')
        states=np.asarray([r['observation.state']for r in items],float)
        if states.shape!=(n,14):raise ValueError('expected RoboTwin 14D states')
        for arm,joints,gripper in [('left',slice(0,6),6),('right',slice(7,13),13)]:
            position,direction=forward_kinematics(states[:,joints],spec[arm])
            for event in detect_arm(position,direction,states[:,gripper],fps,settings):
                event.update(episode_index=episode,arm=arm);events.append(event)
                if not event['accepted']:continue
                for stage in settings.selected_stages:
                    a,b=event['stages'][stage]
                    for frame in range(a,b+1):
                        key=(episode,frame)
                        if key not in selected:selected[key]=dict(episode_index=episode,frame_index=frame,source_index=items[frame].get('index'),labels=[])
                        selected[key]['labels'].append(dict(arm=arm,stage=stage,closure_frame=event['closure_crossing_frame']))
    accepted=[e for e in events if e['accepted']]
    summary=dict(episodes=len(grouped),frames=total,candidate_closures=len(events),accepted_sequences=len(accepted),rejections=dict(Counter(e['rejection']for e in events if not e['accepted'])),selected_frames=len(selected),selected_fraction=len(selected)/total if total else 0,kinematic_proxy_only=True,object_rim_alignment_verified=False,contact_verified=False)
    return summary,events,[selected[k]for k in sorted(selected)]

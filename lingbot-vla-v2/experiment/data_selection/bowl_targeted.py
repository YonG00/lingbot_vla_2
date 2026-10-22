"""Narrow far-grasp and pre-contact placement proxies for stack_bowls_two."""
from dataclasses import dataclass,asdict
from collections import Counter
import math
import numpy as np
from .grasp_sequence import select_rows,forward_kinematics,Settings as GraspSettings
from .placement_sequence import select_placements,PlacementSettings

@dataclass(frozen=True)
class TargetedSettings:
    far_initial_xy_min_m:float=.25
    far_position_xy_min_m:float=.05
    far_position_xy_max_m:float=.20
    far_position_lookback_seconds:float=2.
    progress_window_seconds:float=.2
    minimum_xy_progress_m:float=.002
    minimum_direction_progress_degrees:float=2.
    place_height_min_m:float=.03
    place_height_max_m:float=.08
    open_threshold:float=.95
    include_far_grasp_core:bool=True

    def validate(self):
        for key,value in asdict(self).items():
            if key=='include_far_grasp_core':
                if type(value)is not bool:raise ValueError('invalid include_far_grasp_core')
                continue
            if isinstance(value,bool)or not isinstance(value,(float,int))or not math.isfinite(value)or value<=0:raise ValueError('invalid '+key)
        if not self.far_position_xy_min_m<self.far_position_xy_max_m:raise ValueError('far position range')
        if not self.place_height_min_m<self.place_height_max_m:raise ValueError('height range')
        if self.open_threshold>1:raise ValueError('invalid gripper threshold')


def select_arrays(x,grasp_events,placement_events,fps,settings=TargetedSettings()):
    settings.validate()
    if not math.isfinite(fps)or fps<=0:raise ValueError('invalid fps')
    selected={};diagnostics=[];far_count=0;placement_count=0
    def add(ep,frame,label):
        row=selected.setdefault((int(ep),int(frame)),dict(episode_index=int(ep),frame_index=int(frame),labels=[]))
        if label not in row['labels']:row['labels'].append(label)
    for event in grasp_events:
        if not event['accepted']:continue
        ep=event['episode_index'];arm=event['arm'];ids=np.flatnonzero(x['episode']==ep)
        p=x[arm+'_tcp'][ids];d=x[arm+'_direction'][ids];g=x['state'][ids,6 if arm=='left' else 13]
        a,b=event['stages']['close'];target=np.median(p[a:b+1],axis=0)
        xy=np.linalg.norm(p[:,:2]-target[:2],axis=1)
        # Explicit definition: robot TCP at episode frame0, not after an arbitrary lookback.
        initial=float(xy[0]);qualified=initial+1e-9>=settings.far_initial_xy_min_m
        diag=dict(kind='grasp',episode_index=int(ep),arm=arm,closure_frame=event['closure_crossing_frame'],initial_xy_distance_m=initial,qualified_far=qualified)
        if not qualified:diagnostics.append(diag);continue
        far_count+=1
        align_start=event['stages']['align'][0]
        lower=max(0,align_start-math.ceil(settings.far_position_lookback_seconds*fps))
        closed=np.flatnonzero(g[:align_start]<settings.open_threshold)
        if len(closed):lower=max(lower,int(closed[-1])+1)
        target_direction=d[event['closure_crossing_frame']]
        angle=np.rad2deg(np.arccos(np.clip(d@target_direction,-1,1)))
        window=max(1,math.ceil(settings.progress_window_seconds*fps));far_frames=[]
        for i in range(lower,align_start):
            if not settings.far_position_xy_min_m-1e-9<=xy[i]<=settings.far_position_xy_max_m+1e-9 or g[i]<settings.open_threshold:continue
            previous=max(lower,i-window);following=min(align_start,i+window)
            moving_closer=xy[previous]-xy[following]>=settings.minimum_xy_progress_m
            direction_improves=angle[previous]-angle[following]>=settings.minimum_direction_progress_degrees
            if not(moving_closer or direction_improves):continue
            add(ep,i,dict(stage='far_grasp_position',arm=arm,closure_frame=event['closure_crossing_frame']));far_frames.append(i)
        core_frames=0
        if settings.include_far_grasp_core:
            for stage in ['align','descend','settle','close']:
                start,stop=event['stages'][stage]
                for i in range(start,stop+1):
                    add(ep,i,dict(stage='far_grasp_'+stage,arm=arm,closure_frame=event['closure_crossing_frame']));core_frames+=1
        diag.update(far_position_frames=far_frames,core_frames=core_frames);diagnostics.append(diag)
    for event in placement_events:
        if not event['accepted']or event['placement_ordinal']!=2:continue
        ep=event['episode_index'];arm=event['arm'];ids=np.flatnonzero(x['episode']==ep);p=x[arm+'_tcp'][ids]
        a,b=event['stages']['place_release'];target=np.median(p[a:b+1],axis=0)
        start=event['approach_start'];stop=event['stages']['place_settle'][0];frames=[]
        for i in range(start,stop):
            height=float(p[i,2]-target[2])
            if settings.place_height_min_m-1e-9<=height<=settings.place_height_max_m+1e-9:
                add(ep,i,dict(stage='second_place_height_band',arm=arm,placement_ordinal=2));frames.append(i)
        placement_count+=bool(frames)
        diagnostics.append(dict(kind='second_placement',episode_index=int(ep),arm=arm,selected_frames=frames))
    counts=Counter(label['stage']for row in selected.values()for label in row['labels'])
    summary=dict(task='stack_bowls_two',total_frames=len(x['frame']),episodes=len(np.unique(x['episode'])),qualified_far_grasps=far_count,second_placements_with_height_samples=placement_count,selected_frames=len(selected),selected_fraction=len(selected)/len(x['frame']),stage_frames=dict(counts),initial_reference='episode_frame0_tcp_to_demonstrated_closure_tcp_xy',height_reference='tcp_height_above_demonstrated_release_tcp',kinematic_proxy_only=True,object_pose_verified=False,collision_verified=False)
    return summary,diagnostics,[selected[k]for k in sorted(selected)]


def select_targeted(rows,spec,fps,settings=TargetedSettings()):
    _,grasp_events,_=select_rows(rows,spec,fps,GraspSettings())
    _,placement_events,_=select_placements(rows,spec,fps,PlacementSettings())
    rows=sorted(rows,key=lambda r:(r['episode_index'],r['frame_index']));state=np.asarray([r['observation.state']for r in rows],float)
    x=dict(episode=np.array([r['episode_index']for r in rows]),frame=np.array([r['frame_index']for r in rows]),state=state)
    for arm,joints in [('left',slice(0,6)),('right',slice(7,13))]:x[arm+'_tcp'],x[arm+'_direction']=forward_kinematics(state[:,joints],spec[arm])
    return select_arrays(x,grasp_events,placement_events,fps,settings)

"""Contiguous target segments for far-left grasp and second-bowl placement."""
from dataclasses import dataclass,asdict
from collections import Counter
import math
import numpy as np
from .grasp_sequence import select_rows,forward_kinematics,Settings as GraspSettings
from .placement_sequence import select_placements,PlacementSettings

@dataclass(frozen=True)
class FailureSegmentSettings:
    far_initial_xy_min_m:float=.25
    action_horizon:int=50
    lift_min_height_m:float=.06
    lift_plateau_speed_mps:float=.015
    lift_plateau_frames:int=3
    lift_timeout_seconds:float=3.
    def validate(self):
        if isinstance(self.far_initial_xy_min_m,bool) or not math.isfinite(self.far_initial_xy_min_m) or self.far_initial_xy_min_m<=0:raise ValueError('invalid distance')
        for key in ('lift_min_height_m','lift_plateau_speed_mps','lift_timeout_seconds'):
            value=getattr(self,key)
            if isinstance(value,bool) or not math.isfinite(value) or value<=0:raise ValueError('invalid '+key)
        if type(self.lift_plateau_frames)is not int or self.lift_plateau_frames<2:raise ValueError('invalid plateau length')
        if type(self.action_horizon)is not int or self.action_horizon<=0:raise ValueError('invalid horizon')

def collect_segments(grasp_events,placement_events,far_distances,lengths,settings=FailureSegmentSettings()):
    settings.validate();segments=[]
    for event in grasp_events:
        key=(event['episode_index'],event['arm'],event['closure_crossing_frame'])
        if not event['accepted'] or event['arm']!='left':continue
        if far_distances[key]+1e-9<settings.far_initial_xy_min_m:continue
        stages=event['stages'];valid=[span for span in stages.values() if span[1]>=span[0]]
        segments.append(dict(kind='far_left_grasp',episode_index=event['episode_index'],arm='left',
            start_frame=min(a for a,b in valid),end_frame=stages['lift'][1],stages=stages,
            initial_xy_distance_m=far_distances[key],closure_frame=event['closure_crossing_frame'],
            endpoint='closed_gripper_6cm_lift_and_vertical_plateau'))
    for event in placement_events:
        if not event['accepted'] or event['placement_ordinal']!=2:continue
        stages=event['stages'];valid=[span for span in stages.values() if span[1]>=span[0]]
        segments.append(dict(kind='second_bowl_place',episode_index=event['episode_index'],arm=event['arm'],
            start_frame=min(a for a,b in valid),end_frame=max(b for a,b in valid),stages=stages,
            placement_ordinal=2,endpoint='release_complete_and_3cm_retreat'))
    selected={};crossing=Counter()
    for segment in segments:
        ep=segment['episode_index'];a=segment['start_frame'];b=segment['end_frame']
        if not 0<=a<=b<lengths[ep]:raise ValueError('invalid segment bounds')
        segment['frame_count']=b-a+1
        segment['future_action_cross_boundary_anchors']=sum(min(i+settings.action_horizon-1,lengths[ep]-1)>b for i in range(a,b+1))
        crossing[segment['kind']]+=segment['future_action_cross_boundary_anchors']
        for frame in range(a,b+1):
            row=selected.setdefault((ep,frame),dict(episode_index=ep,frame_index=frame,labels=[]))
            labels=[name for name,(lo,hi) in segment['stages'].items() if lo<=frame<=hi]
            row['labels'].append(dict(kind=segment['kind'],arm=segment['arm'],stages=labels))
    return segments,[selected[k] for k in sorted(selected)],dict(crossing)

def select_failure_segments(rows,spec,fps,settings=FailureSegmentSettings()):
    settings.validate()
    gs,ge,_=select_rows(rows,spec,fps,GraspSettings(selected_stages=('aim','align','descend','settle','close','lift')))
    ps,pe,_=select_placements(rows,spec,fps,PlacementSettings())
    episodes={}
    for row in rows:episodes.setdefault(int(row['episode_index']),[]).append(row)
    distances={};lengths={}
    for ep,items in episodes.items():
        items.sort(key=lambda x:x['frame_index']);lengths[ep]=len(items)
        states=np.asarray([x['observation.state'] for x in items],float)
        p,_=forward_kinematics(states[:,:6],spec['left'])
        for event in ge:
            if event['episode_index']!=ep or event['arm']!='left' or not event['accepted']:continue
            lo,hi=event['stages']['close'];target=np.median(p[lo:hi+1],axis=0)
            distance=float(np.linalg.norm(p[0,:2]-target[:2]))
            distances[(ep,'left',event['closure_crossing_frame'])]=distance
            if distance+1e-9<settings.far_initial_xy_min_m:continue
            close=event['close_end_frame'];start=event['lift_confirmation_frame'];finish=None
            for end in range(start+settings.lift_plateau_frames-1,min(len(p),close+math.ceil(settings.lift_timeout_seconds*fps)+1)):
                z=p[end-settings.lift_plateau_frames+1:end+1,2]
                if min(z)-p[close,2]<settings.lift_min_height_m:continue
                if np.max(np.abs(np.diff(z)*fps))>settings.lift_plateau_speed_mps:continue
                if np.max(states[close:end+1,6])>.5:continue
                finish=end;break
            if finish is None:raise ValueError('far-left lift has no confirmed plateau: '+str(ep))
            event['stages']=dict(event['stages'],lift=[close+1,finish])
    segments,selected,crossing=collect_segments(ge,pe,distances,lengths,settings)
    summary=dict(task='stack_bowls_two',episodes=len(episodes),total_frames=len(rows),
        segment_counts=dict(Counter(s['kind'] for s in segments)),
        frames_by_kind={kind:len({(s['episode_index'],i) for s in segments if s['kind']==kind for i in range(s['start_frame'],s['end_frame']+1)}) for kind in ('far_left_grasp','second_bowl_place')},
        selected_frames=len(selected),selected_fraction=len(selected)/len(rows),
        future_action_cross_boundary_anchors=crossing,settings=asdict(settings),
        grasp_rejections=gs['rejections'],placement_rejections=ps['rejections'],
        initial_reference='episode_frame0_tcp_to_demonstrated_closure_tcp_xy',
        full_contiguous_frames=True,subsampled=False,kinematic_proxy_only=True,
        object_pose_verified=False,contact_verified=False,collision_verified=False)
    return summary,segments,selected

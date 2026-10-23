"""Full horizontal dual-bottle trajectory labels; stages and weight policy stay separate."""
from collections import Counter
import hashlib,json
import numpy as np
from .grasp_sequence import forward_kinematics,_runs
from .dual_bottle_grasp import select_rows,Settings
from .full_stages import validate_table,STAGES

def build(rows,spec,fps,task='pick_dual_bottles',objects=None):
    if task not in ('pick_dual_bottles','pick_diverse_bottles'):raise ValueError('unsupported bottle task')
    if task=='pick_diverse_bottles'and objects is None:raise ValueError('diverse bottles require per-episode object identities')
    summary,events,_=select_rows(rows,spec,fps)
    if summary['paired_episodes']!=summary['episodes']:raise ValueError('all episodes require both grasps')
    grouped={}
    for row in rows:grouped.setdefault(row['episode_index'],[]).append(row)
    episodes=[];stats={'left':Counter(),'right':Counter()};substats={'left':Counter(),'right':Counter()};warnings=[]
    for ep,items in sorted(grouped.items()):
        items.sort(key=lambda x:x['frame_index']);n=len(items);state=np.asarray([v['observation.state']for v in items]);arms={}
        for arm,j,obj in [('left',0,'001_bottle/base13'),('right',7,'001_bottle/base16')]:
            if objects is not None:
                identities=objects.get(str(ep),objects.get(ep))
                if not isinstance(identities,dict)or set(identities)!={'left','right'}:raise ValueError('missing object identities: '+str(ep))
                obj=identities[arm]
                if not isinstance(obj,str)or not obj.startswith('001_bottle/base')or not obj[len('001_bottle/base'):].isdigit():raise ValueError('invalid bottle identity')
            e=next(v for v in events if v['episode_index']==ep and v['arm']==arm);p,_=forward_kinematics(state[:,j:j+6],spec[arm]);g=state[:,j+6];speed=np.linalg.norm(np.gradient(p,axis=0)*fps,axis=1)
            cs=e['close_start_frame'];ce=e['close_end_frame'];align_a,align_b=e['stages']['horizontal_align'];arrival=e['stages']['forward_approach'][1]
            candidates=np.flatnonzero((np.arange(n)>=e['end_frame'])&(p[:,2]-p[ce,2]>=.095)&(np.linalg.norm(p[:,:2]-p[ce,:2],axis=1)<=.015)&(g<=.2))
            if not len(candidates):raise ValueError('no isolated 10cm lift: '+str((ep,arm)))
            lift_end=int(candidates[0]);target=np.median(p[-3:],axis=0);distance=np.linalg.norm(p-target,axis=1)
            holds=[(int(a),int(b))for a,b in _runs((distance<=.015)&(speed<=.025)&(g<=.2))if b==n-1 and b-a+1>=2 and a>lift_end]
            hold_start=holds[-1][0]if holds else n
            movement=np.flatnonzero(np.linalg.norm(p[:align_a]-p[0],axis=1)>.005);reach_start=int(movement[0])if len(movement)else align_a
            labels=[]
            for f in range(n):
                if f<reach_start:stage,sub='idle_or_prepare','initial_pose'
                elif f<align_a:
                    dist=np.linalg.norm(p[f]-p[align_b]);stage='reach_aim';sub='reach_very_far'if dist>.3 else 'reach_far'if dist>.2 else 'reach_middle'if dist>.1 else 'reach_near'
                elif f<=align_b:stage,sub='fine_align','stable_horizontal_pregrasp'
                elif f<cs:stage,sub='pre_contact',('forward_advance'if f<=arrival else 'open_settle')
                elif f<=ce:stage,sub='gripper_close','closing_transition'
                elif f<=lift_end:
                    rise=p[f,2]-p[ce,2];stage='grasp_lift';sub='closed_hold_proxy'if rise<.002 else 'initial_clearance'if rise<.03 else 'lift_clearance'
                elif f>=hold_start:stage,sub='target_hold','closed_target_hold_proxy'
                elif distance[f]<=.03:stage,sub='target_align','held_target_alignment'
                else:stage,sub='transport',('transport_far'if distance[f]>.2 else 'transport_middle'if distance[f]>.1 else 'transport_near')
                labels.append((stage,sub));stats[arm][stage]+=1;substats[arm][stage+'/'+sub]+=1
            segments=[];start=0
            for end in range(1,n+1):
                if end==n or labels[end]!=labels[start]:
                    stage,sub=labels[start];segments.append(dict(start=start,end=end,stage=stage,substage=sub,object_id=obj,event_id=arm+'_bottle_interaction_1',proxy_only=True,evidence='expert_success_joint_fk_gripper_command',confidence='kinematic_proxy'));start=end
            arms[arm]=segments
        episodes.append(dict(episode_index=ep,length=n,arms=arms))
    fingerprint=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    table=dict(schema_version=1,dataset_name='robotwin',task=task,fps=fps,coordinate_space='original_episode_frame',recognizer_version='dual_bottle_full_stages_v1',source_action_channels={'action':{'left':list(range(7)),'right':list(range(7,14))},'observation.state':{'left':list(range(7)),'right':list(range(7,14))}},numeric_source_sha256=fingerprint(sorted(rows,key=lambda x:(x['episode_index'],x['frame_index']))),kinematics_sha256=fingerprint(spec),episodes=episodes)
    validate_table(table);total=len(rows)
    for arm in ['left','right']:
        for name,count in substats[arm].items():
            if count/total>.1:warnings.append(dict(arm=arm,substage=name,frames=count,fraction=count/total,action='review_meaningful_boundaries_not_forced_split'))
    report=dict(episodes=len(episodes),frames_per_arm=total,total_arm_frames=2*total,main_stage_counts={a:dict(v)for a,v in stats.items()},substage_counts={a:dict(v)for a,v in substats.items()},substage_over_10pct=warnings,full_coverage=True,contact_verified=False,release_stage_present=False,subsampled=False)
    return table,report

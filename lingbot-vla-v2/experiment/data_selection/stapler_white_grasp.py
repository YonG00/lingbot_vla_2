"""Retain broad placement coordinates and add model-tagged white-stapler grasps."""
from collections import Counter
import numpy as np
from .grasp_sequence import forward_kinematics,detect_arm,Settings
WHITE_MODEL='048_stapler/base3'

def merge_coordinates(broad,extra):
    merged={}
    for kind,rows in [('broad_postgrasp',broad),('white_grasp',extra)]:
        seen=set()
        for row in rows:
            key=(row['episode_index'],row['frame_index'])
            if any(type(v)is not int or v<0 for v in key)or key in seen:raise ValueError('invalid/duplicate coordinates')
            seen.add(key)
            item=merged.setdefault(key,dict(episode_index=key[0],frame_index=key[1],labels=[]))
            item['labels'].append(dict(source=kind,**row.get('label',{})))
    return [merged[k]for k in sorted(merged)]

def select_white_grasps(rows,catalog,spec,fps,broad,placement_events):
    grouped={}
    for row in rows:grouped.setdefault(row['episode_index'],[]).append(row)
    if len(catalog)!=len(grouped)or {c['episode_index']for c in catalog}!=set(grouped):raise ValueError('catalog episode mismatch')
    selected=[];events=[];white=[]
    for item in catalog:
        if item['scene']['{A}']!=WHITE_MODEL:continue
        ep=item['episode_index'];items=sorted(grouped[ep],key=lambda x:x['frame_index'])
        if [v['frame_index']for v in items]!=list(range(len(items)))or len(items)!=item['length']:raise ValueError('frame/length mismatch')
        arm=item['scene']['{a}']
        if arm not in ('left','right'):raise ValueError('invalid arm')
        state=np.asarray([v['observation.state']for v in items]);j=0 if arm=='left'else 7
        p,d=forward_kinematics(state[:,j:j+6],spec[arm]);found=[e for e in detect_arm(p,d,state[:,j+6],fps,Settings(selected_stages=('aim','align','descend','settle','close','lift')))if e['accepted']]
        place=[e for e in placement_events if e['episode_index']==ep and e['accepted']and e['arm']==arm]
        if len(found)!=1 or len(place)!=1:raise ValueError('requires one verified grasp/placement: '+str(ep))
        e=found[0];end=place[0]['broad_postgrasp_start_frame']
        if e['close_end_frame']>=end or e['lift_confirmation_frame']>end:raise ValueError('lift boundary mismatch')
        e['stages']['lift'][1]=end
        start=e['stages']['aim'][0]
        for frame in range(start,end+1):
            stages=[stage for stage,(a,b)in e['stages'].items()if a<=frame<=b]
            if len(stages)!=1:raise ValueError('grasp stage gap/overlap')
            selected.append(dict(episode_index=ep,frame_index=frame,label=dict(arm=arm,stage=stages[0],model=WHITE_MODEL)))
        events.append(dict(episode_index=ep,source_episode=item['source_episode'],model=WHITE_MODEL,arm=arm,start_frame=start,end_frame=end,grasp=e));white.append(ep)
    if not white:raise ValueError('no white stapler demos')
    merged=merge_coordinates(broad,selected)
    valid={(v['episode_index'],v['frame_index'])for v in rows}
    if not {(v['episode_index'],v['frame_index'])for v in merged}<=valid:raise ValueError('foreign selected coordinate')
    summary=dict(task='move_stapler_pad',white_model=WHITE_MODEL,white_episodes=white,white_sequences=len(events),white_grasp_frames=len(selected),broad_frames=len(broad),new_frames=len(merged)-len(broad),overlap_frames=len(broad)+len(selected)-len(merged),selected_frames=len(merged),total_frames=len(rows),selected_fraction=len(merged)/len(rows),arm_counts=dict(Counter(e['arm']for e in events)),stage_frames=dict(Counter(v['label']['stage']for v in selected)),subsampled=False,kinematic_proxy_only=True,loss_multiplier=2,normalization='sample_count',importance_sampling=False)
    return summary,events,selected,merged

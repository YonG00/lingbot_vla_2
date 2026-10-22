"""Preserve original grasp targets; add contiguous second-placement contact approach."""
from dataclasses import dataclass
import math
from .grasp_sequence import Settings,select_rows
from .placement_sequence import PlacementSettings,select_placements

@dataclass(frozen=True)
class AddPlaceSettings:
    approach_context_seconds:float=1/3
    def validate(self):
        if isinstance(self.approach_context_seconds,bool)or not math.isfinite(self.approach_context_seconds)or self.approach_context_seconds<0:raise ValueError('invalid approach context')

def placement_rows(events,fps,context):
    chosen={};segments=[]
    for event in events:
        if not event['accepted']or event['placement_ordinal']!=2:continue
        start=max(event['stages']['place_aim'][0],event['approach_start']-math.ceil(context*fps))
        end=event['release_end']
        if not 0<=start<=event['approach_start']<=end:raise ValueError('invalid placement bounds')
        segments.append(dict(episode_index=event['episode_index'],arm=event['arm'],start_frame=start,end_frame=end,approach_start=event['approach_start'],stages=event['stages']))
        for frame in range(start,end+1):
            key=(event['episode_index'],frame)
            chosen.setdefault(key,dict(episode_index=key[0],frame_index=key[1],labels=[]))['labels'].append(dict(stage='second_place_approach_settle_release',arm=event['arm']))
    return segments,[chosen[key]for key in sorted(chosen)]

def select_grasp_place10(rows,spec,fps,settings=AddPlaceSettings()):
    settings.validate()
    gs,ge,gr=select_rows(rows,spec,fps,Settings())
    ps,pe,_=select_placements(rows,spec,fps,PlacementSettings())
    segments,pr=placement_rows(pe,fps,settings.approach_context_seconds)
    chosen={(row['episode_index'],row['frame_index']):dict(row)for row in gr};old=set(chosen)
    for row in pr:
        key=(row['episode_index'],row['frame_index'])
        if key not in chosen:chosen[key]=row
        else:chosen[key]['labels']=chosen[key]['labels']+row['labels']
    summary=dict(task='stack_bowls_two',episodes=gs['episodes'],total_frames=len(rows),original_grasp_frames=len(gr),second_place_segments=len(segments),placement_frames=len(pr),new_placement_frames=len(set(chosen)-old),placement_fraction=len(pr)/len(rows),selected_frames=len(chosen),selected_fraction=len(chosen)/len(rows),original_grasp_preserved=old<=set(chosen),subsampled=False,kinematic_proxy_only=True,contact_verified=False)
    return summary,dict(grasp_events=ge,placement_segments=segments),[chosen[key]for key in sorted(chosen)]

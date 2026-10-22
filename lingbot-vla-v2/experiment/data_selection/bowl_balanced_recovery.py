"""Original all-grasp anchors plus second placement; no sparse subsampling."""
from collections import Counter
from .grasp_sequence import select_rows,Settings
from .placement_sequence import select_placements,PlacementSettings

def select_balanced(rows,spec,fps):
    gs,ge,gr=select_rows(rows,spec,fps,Settings())
    ps,pe,pr=select_placements(rows,spec,fps,PlacementSettings())
    selected={};segments=[]
    for row in gr:
        key=(row['episode_index'],row['frame_index']);weight=max(3 if label['arm']=='left' else 2 for label in row['labels'])
        selected[key]=dict(row,loss_weight=weight)
    for row in pr:
        key=(row['episode_index'],row['frame_index'])
        if key not in selected:selected[key]=dict(row,loss_weight=2)
        else:selected[key]['labels']=selected[key]['labels']+row['labels']
    for event in ge:
        if not event['accepted']:continue
        stages={name:event['stages'][name] for name in Settings().selected_stages}
        spans=[span for span in stages.values() if span[1]>=span[0]]
        segments.append(dict(kind=event['arm']+'_grasp',episode_index=event['episode_index'],arm=event['arm'],start_frame=min(a for a,b in spans),end_frame=max(b for a,b in spans),stages=stages))
    for event in pe:
        if not event['accepted']or event['placement_ordinal']!=2:continue
        spans=[span for span in event['stages'].values() if span[1]>=span[0]]
        segments.append(dict(kind='second_bowl_place',episode_index=event['episode_index'],arm=event['arm'],start_frame=min(a for a,b in spans),end_frame=max(b for a,b in spans),stages=event['stages']))
    union={(s['episode_index'],frame)for s in segments for frame in range(s['start_frame'],s['end_frame']+1)}
    if union!=set(selected):raise ValueError('segment union differs from original target anchors')
    summary=dict(task='stack_bowls_two',episodes=gs['episodes'],total_frames=len(rows),original_grasp_frames=len(gr),second_place_frames=len(pr),selected_frames=len(selected),background_frames=len(rows)-len(selected),selected_fraction=len(selected)/len(rows),segment_counts=dict(Counter(s['kind']for s in segments)),coordinate_weights=dict(Counter(v['loss_weight']for v in selected.values())),original_grasp_stages=Settings().selected_stages,grasp_rejections=gs['rejections'],placement_rejections=ps['rejections'],subsampled=False,kinematic_proxy_only=True,contact_verified=False)
    return summary,segments,[selected[key]for key in sorted(selected)]

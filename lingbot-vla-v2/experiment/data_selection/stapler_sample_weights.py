"""Task coordinate classes, not full-trajectory stages; overlapping classes use max."""
from collections import Counter
WHITE_MODEL='048_stapler/base3'
def build(catalog,placement):
 lengths={e['episode_index']:e['length']for e in catalog}
 if len(lengths)!=len(catalog)or not lengths:raise ValueError('duplicate/empty episodes')
 valid={(ep,f)for ep,n in lengths.items()for f in range(n)}
 keys=[(v['episode_index'],v['frame_index'])for v in placement]
 if len(keys)!=len(set(keys))or not set(keys)<=valid:raise ValueError('invalid placement coordinates')
 place=set(keys);white={e['episode_index']for e in catalog if e['scene']['{A}']==WHITE_MODEL}
 if not white or not place:raise ValueError('empty target class')
 rows=[]
 for ep,f in sorted(valid):
  kinds=[]
  if ep in white:kinds.append('white_stapler_demo')
  if (ep,f)in place:kinds.append('broad_placement')
  rows.append(dict(episode_index=ep,frame_index=f,loss_weight=20. if (ep,f)in place else 15. if ep in white else 1.,labels=[dict(source=k)for k in kinds]))
 counts=Counter(v['loss_weight']for v in rows)
 return rows,dict(task='move_stapler_pad',episodes=len(lengths),frames=len(valid),white_model=WHITE_MODEL,white_episodes=sorted(white),white_all_frames=sum(lengths[e]for e in white),placement_frames=len(place),overlap_frames=sum(ep in white for ep,f in place),effective_weight_frames={str(k):v for k,v in counts.items()},normalization='weight_sum',overlap_rule='max',importance_sampling=False,subsampled=False,training_started=False)


def build_early_late(catalog,placement,events):
    rows,report=build(catalog,placement)
    lengths={e['episode_index']:e['length']for e in catalog};accepted=[e for e in events if e['accepted']]
    mapped={e['episode_index']:e for e in accepted}
    if set(mapped)!=set(lengths)or len(mapped)!=len(accepted):raise ValueError('one accepted placement per episode required')
    broad=set();late=set();bounds=[]
    for ep,e in sorted(mapped.items()):
        start=e['broad_postgrasp_start_frame'];ready=e['preplace_end'];release=e['release_start_frame'];end=e['end_frame']
        if not 0<=start<=ready<release<=e['release_end_frame']<=end==lengths[ep]-1:raise ValueError('placement temporal order')
        broad.update((ep,f)for f in range(start,end+1));late.update((ep,f)for f in range(ready+1,end+1))
        bounds.append(dict(episode_index=ep,arm=e['arm'],early_start=start,early_end_exclusive=ready+1,late_start=ready+1,late_end_exclusive=end+1,evidence='stable_preplace_end_then_final_descent_release_tail',proxy_only=True))
    if broad!={(v['episode_index'],v['frame_index'])for v in placement}:raise ValueError('broad/event coordinates differ')
    white=set(report['white_episodes'])
    for row in rows:
        key=(row['episode_index'],row['frame_index']);iswhite=row['episode_index']in white
        kind='late_placement'if key in late else 'early_placement'if key in broad else None
        row['loss_weight']=20. if key in late else 15. if key in broad else 10. if iswhite else 1.
        row['labels']=([dict(source='white_stapler_demo')]if iswhite else [])+([dict(source=kind)]if kind else [])
    report.update(early_placement_frames=len(broad-late),late_placement_frames=len(late),white_other_multiplier=10,early_placement_multiplier=15,late_placement_multiplier=20,effective_weight_frames={str(k):v for k,v in Counter(v['loss_weight']for v in rows).items()},placement_boundaries=bounds,placement_rule='lift_confirm_to_preplace_end=early; final_descent_release_tail=late',contact_verified=False)
    return rows,report

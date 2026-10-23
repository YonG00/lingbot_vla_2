#!/usr/bin/env python3
"""Generate full dual-bottle stages and an independent weight-sum policy; never trains."""
import argparse,json,hashlib,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.dual_bottle_full_stages import build
from experiment.data_selection.full_stages import STAGES,StageTable

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames-json',type=Path);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fps',type=float,default=15)
 a=p.parse_args();manifest=json.loads((ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text());entry=manifest['tasks']['pick_dual_bottles'];ids=set(entry['train_ids']+entry['val_ids'])
 if a.frames_json:rows=json.loads(a.frames_json.read_text());fps=a.fps
 else:
  import pyarrow.dataset as ds
  fps=json.loads((a.dataset_root/'meta/info.json').read_text())['fps'];rows=ds.dataset(str(a.dataset_root/'data'),format='parquet').to_table(columns=['episode_index','frame_index','timestamp','index','observation.state'],filter=ds.field('episode_index').isin(sorted(ids)),use_threads=False).to_pylist()
 if {v['episode_index']for v in rows}!=ids:raise ValueError('manifest episode mismatch')
 spec=json.loads((ROOT/'configs/kinematics/robotwin_aloha_agilex_ee.json').read_text());table,summary=build(rows,spec,fps)
 values={s:1. for s in STAGES};values.update(fine_align=2.,pre_contact=5.,gripper_close=5.,grasp_lift=2.)
 policy=dict(version=1,normalization='weight_sum',fm=values,visual=dict(values),visual_branches={name:dict(values)for name in ['current_depth','future_depth','current_video','future_video']},future_video_rule='target_endpoint',unknown_high_weight=False,review_threshold_fraction=.1,overrides={})
 policy['overrides']={branch:{arm:{seg['stage']+'/'+seg['substage']:values[seg['stage']]for ep in table['episodes']for seg in ep['arms'][arm]}for arm in ['left','right']}for branch in ['fm','visual']}
 selected=[dict(episode_index=ep['episode_index'],frame_index=f)for ep in table['episodes']for f in range(ep['length'])]
 a.output_dir.mkdir(parents=True,exist_ok=False)
 for name,obj in [('segments',table),('stage_loss_weights',policy),('selected_frames',selected)]: (a.output_dir/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 sha=lambda name:hashlib.sha256((a.output_dir/name).read_bytes()).hexdigest()
 cfg=dict(version=1,task='pick_dual_bottles',dataset_name='robotwin',dataset_root=str(a.dataset_root.resolve()),normalization='weight_sum',multiplier=1,selected_frames='selected_frames.json',selected_frames_sha256=sha('selected_frames.json'),stage_table='segments.json',stage_table_sha256=sha('segments.json'),stage_policy='stage_loss_weights.json',stage_policy_sha256=sha('stage_loss_weights.json'))
 (a.output_dir/'weight_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 stage=StageTable(cfg,a.output_dir);raw={a:Counter()for a in ['left','right']};windows={a:Counter()for a in ['left','right']};current=Counter();future=Counter();video=Counter();pads=0
 for ep,e in stage.episodes.items():
  n=e['length']
  def visual(t):return max(stage.coefficients['visual'][stage.labels[ep][arm][t]]for arm in ['left','right'])
  for t in range(n):
   current[visual(t)]+=1;future[visual(t+49)if t+49<n else 0.]+=1;video[visual(t+49)if t+49<n else 0.]+=1;pads+=max(0,t+50-n)
   for arm in ['left','right']:
    raw[arm][stage.coefficients['fm'][stage.labels[ep][arm][t]]]+=1
    for k in range(min(50,n-t)):windows[arm][stage.coefficients['fm'][stage.labels[ep][arm][t+k]]]+=1
 summary.update(normalization='weight_sum',raw_fm_weight_frames={a:dict(c)for a,c in raw.items()},all_anchor_chunk_fm_arm_steps={a:dict(c)for a,c in windows.items()},all_anchor_padded_time_steps=pads,current_visual_weight_samples=dict(current),future_depth_weight_samples=dict(future),future_video_weight_samples=dict(video),future_video_rule='target_endpoint',training_started=False)
 (a.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 if 'torch'in sys.modules:raise RuntimeError('annotation must not import Torch')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Prepare target classes using the same per-action category contract as trajectory phases."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.stapler_sample_weights import build,build_early_late
from experiment.data_selection.category_conversion import from_targets,base_policy,write_snapshot
from experiment.data_selection.full_stages import policy_coverage

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--placement-mode',choices=['early_late','broad'],default='early_late');p.add_argument('--split-placement',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
 catalog=ROOT/'refs/data_selection/move_stapler_pad_broad_white_grasp_v1/episode_catalog.json';broad=ROOT/'refs/data_selection/move_stapler_pad_postgrasp_place_v1/broad_reference_frames.json';cat=json.loads(catalog.read_text());place=json.loads(broad.read_text())
 if a.split_placement or a.placement_mode=='early_late':
  events=json.loads((ROOT/'refs/data_selection/move_stapler_pad_postgrasp_place_v1/events.json').read_text())['events'];rows,report=build_early_late(cat,place,events);weights={'background':1,'early_placement':15,'late_placement':20,'white_stapler_demo':10}
 else:rows,report=build(cat,place);weights={'background':1,'broad_placement':20,'white_stapler_demo':15}
 cfg=dict(version=1,task='move_stapler_pad',dataset_name='robotwin',dataset_root='/root/autodl-tmp/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30')
 table,policy=from_targets(cfg,rows,cat,base_policy({name:dict(fm=value,visual=value)for name,value in weights.items()}));compiled=write_snapshot(cfg,table,policy,a.output_dir)
 report.update(weighting_unit='action_timestep',catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),placement_sha256=hashlib.sha256(broad.read_bytes()).hexdigest());(a.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output_dir/'coverage.json').write_text(json.dumps(policy_coverage(compiled),indent=2)+'\n');print(json.dumps({k:report[k]for k in ['task','frames','normalization','weighting_unit','effective_weight_frames']},indent=2))
 if 'torch'in sys.modules:raise RuntimeError('offline category preparation imported torch')
if __name__=='__main__':main()

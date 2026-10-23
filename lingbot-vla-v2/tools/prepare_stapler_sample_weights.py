#!/usr/bin/env python3
"""Freeze sample classes; --split-placement uses early15/late20/white10; never trains."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.stapler_sample_weights import build,build_early_late
from experiment.data_selection.weights import SelectionWeights
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--split-placement',action='store_true');a=p.parse_args()
 catalog=ROOT/'refs/data_selection/move_stapler_pad_broad_white_grasp_v1/episode_catalog.json';broad=ROOT/'refs/data_selection/move_stapler_pad_postgrasp_place_v1/broad_reference_frames.json'
 if a.split_placement:
  events=json.loads((ROOT/'refs/data_selection/move_stapler_pad_postgrasp_place_v1/events.json').read_text())['events'];rows,report=build_early_late(json.loads(catalog.read_text()),json.loads(broad.read_text()),events)
 else:rows,report=build(json.loads(catalog.read_text()),json.loads(broad.read_text()))
 a.output_dir.mkdir(parents=True,exist_ok=False)
 (a.output_dir/'selected_frames.json').write_text(json.dumps(rows,indent=2)+'\n')
 cfg=dict(version=1,task='move_stapler_pad',dataset_name='robotwin',dataset_root='/root/autodl-tmp/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30',normalization='weight_sum',multiplier=1,use_selected_frame_weights=True,selected_frames='selected_frames.json',selected_frames_sha256=hashlib.sha256((a.output_dir/'selected_frames.json').read_bytes()).hexdigest())
 (a.output_dir/'weight_config.json').write_text(json.dumps(cfg,indent=2)+'\n');SelectionWeights(a.output_dir/'weight_config.json')
 report.update(catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),placement_sha256=hashlib.sha256(broad.read_bytes()).hexdigest());(a.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

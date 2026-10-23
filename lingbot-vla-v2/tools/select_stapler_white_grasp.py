#!/usr/bin/env python3
"""CPU-only: append white-base3 aiming through confirmed 8cm lift to frozen broad targets."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.stapler_white_grasp import select_white_grasps

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['frames-json','catalog','broad-dir','output-dir','training-dataset-root']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();load=lambda p:json.loads(p.read_text())
 spec=load(ROOT/'configs/kinematics/robotwin_aloha_agilex_ee.json');broad=load(a.broad_dir/'selected_frames.json');old=load(a.broad_dir/'summary.json')
 if old.get('selection_scope')!='postgrasp' or len(broad)!=3595:raise ValueError('requires frozen 3595 broad selection')
 summary,events,white,selected=select_white_grasps(load(a.frames_json),load(a.catalog),spec,15,broad,load(a.broad_dir/'events.json')['events'])
 summary.update(rule_version='stapler_broad_plus_white_grasp_v1',inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [a.frames_json,a.catalog,a.broad_dir/'selected_frames.json']})
 a.output_dir.mkdir(exist_ok=False,parents=True)
 for name,v in [('summary',summary),('events',events),('white_grasp_frames',white),('selected_frames',selected)]:
  (a.output_dir/(name+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 cfg=dict(version=1,task='move_stapler_pad',rule_version=summary['rule_version'],dataset_name='robotwin',dataset_root=str(a.training_dataset_root.resolve()),normalization='sample_count',multiplier=2,selected_frames='selected_frames.json',selected_frames_sha256=hashlib.sha256((a.output_dir/'selected_frames.json').read_bytes()).hexdigest())
 (a.output_dir/'weight_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 if 'torch'in sys.modules:raise RuntimeError('CPU selector must not import torch')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

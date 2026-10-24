#!/usr/bin/env python3
"""Freeze new class weights against unchanged frame labels, for either labeling method."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.full_stages import checked_json,policy_coverage
from experiment.data_selection.category_conversion import write_snapshot
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels-dir',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();cfg=json.loads((a.labels_dir/'weight_config.json').read_text());table=checked_json(cfg,'category_table',a.labels_dir);stage=write_snapshot(cfg,table,json.loads(a.weights.read_text()),a.output_dir);(a.output_dir/'coverage.json').write_text(json.dumps(policy_coverage(stage),indent=2)+'\n');print('CATEGORY_WEIGHTS_FROZEN_NO_TRAINING')

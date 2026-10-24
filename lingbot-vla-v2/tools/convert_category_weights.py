#!/usr/bin/env python3
"""Convert full-trajectory stages or target classes to the same per-action contract."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.category_conversion import from_stages,from_targets,write_snapshot
from experiment.data_selection.full_stages import checked_json,policy_coverage
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-config',type=Path,required=True);p.add_argument('--target-catalog',type=Path);p.add_argument('--class-weights',type=Path);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.source_config.read_text())
if 'category_table'in cfg:table=checked_json(cfg,'category_table',a.source_config.parent);policy=checked_json(cfg,'category_weights',a.source_config.parent)
elif 'stage_table'in cfg:table,policy=from_stages(cfg,a.source_config.parent)
else:
 if a.target_catalog is None or a.class_weights is None:p.error('target classes require --target-catalog and --class-weights')
 table,policy=from_targets(cfg,checked_json(cfg,'selected_frames',a.source_config.parent),json.loads(a.target_catalog.read_text()),json.loads(a.class_weights.read_text()))
result=write_snapshot(cfg,table,policy,a.output_dir);report=policy_coverage(result);report.update(task=cfg['task'],label_source=table['label_source'],weighting_unit='action_timestep',training_started=False);(a.output_dir/'coverage.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['task','label_source','weighting_unit','training_started']}))

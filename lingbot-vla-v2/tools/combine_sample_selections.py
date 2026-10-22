#!/usr/bin/env python3
"""Combine task-specific selectors with one configurable sample multiplier."""
import argparse,json,hashlib,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.combine import combine_directories

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,action='append',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--multiplier',type=float,required=True)
    p.add_argument('--dataset-name',required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    a=p.parse_args()
    if not math.isfinite(a.multiplier) or a.multiplier<=0:p.error('invalid multiplier')
    if a.output_dir.resolve().is_relative_to(a.dataset_root.resolve()):p.error('output must be outside dataset')
    summary,rows=combine_directories(a.selection)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    selected=a.output_dir/'selected_frames.json';selected.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    config=dict(version=1,task=summary['task'],rule_version=summary['rule_version'],normalization='sample_count',combination='union_no_stacking',dataset_name=a.dataset_name,dataset_root=str(a.dataset_root.resolve()),multiplier=a.multiplier,selected_frames='selected_frames.json',selected_frames_sha256=hashlib.sha256(selected.read_bytes()).hexdigest())
    (a.output_dir/'weight_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (a.output_dir/'summary.json').write_text(json.dumps(dict(summary,multiplier=a.multiplier,normalization='sample_count'),indent=2)+'\n')
    print(json.dumps(dict(summary,multiplier=a.multiplier),indent=2))
if __name__=='__main__':main()

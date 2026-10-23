#!/usr/bin/env python3
"""Freeze a new policy against unchanged full-stage labels; no relabeling or training."""
import argparse,json,hashlib,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.full_stages import StageTable,policy_coverage
from experiment.data_selection.weights import SelectionWeights

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels-dir',type=Path,required=True);p.add_argument('--weights','--policy',dest='policy',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 cfg=json.loads((a.labels_dir/'weight_config.json').read_text());StageTable(cfg,a.labels_dir)
 if a.output_dir.exists():raise ValueError('refuse overwrite policy snapshot')
 a.output_dir.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.stage_policy_',dir=a.output_dir.parent))
 try:
  for field,name in [('selected_frames','selected_frames.json'),('stage_table','segments.json')]:
   source=Path(cfg[field]);source=source if source.is_absolute()else a.labels_dir/source;shutil.copyfile(source,tmp/name);cfg[field]=name;cfg[field+'_sha256']=hashlib.sha256((tmp/name).read_bytes()).hexdigest()
  shutil.copyfile(a.policy,tmp/'stage_loss_weights.json');cfg['stage_policy']='stage_loss_weights.json';cfg['stage_policy_sha256']=hashlib.sha256((tmp/'stage_loss_weights.json').read_bytes()).hexdigest()
  stage=StageTable(cfg,tmp);(tmp/'weight_config.json').write_text(json.dumps(cfg,indent=2)+'\n');SelectionWeights(tmp/'weight_config.json')
  report=policy_coverage(stage);report.update(training_started=False,normalization='weight_sum',labels_unchanged=True);(tmp/'policy_coverage.json').write_text(json.dumps(report,indent=2)+'\n');tmp.rename(a.output_dir)
  print(json.dumps(report,indent=2))
 finally:
  if tmp.exists():shutil.rmtree(tmp)
 if 'torch'in sys.modules:raise RuntimeError('policy snapshot must not import torch')
if __name__=='__main__':main()

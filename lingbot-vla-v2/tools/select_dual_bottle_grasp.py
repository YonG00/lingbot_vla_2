#!/usr/bin/env python3
"""双手水平瞄准、前伸、闭爪及提瓶连续片段筛选；不启动训练。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.data_selection.dual_bottle_grasp import Settings,select_rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True)
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--frames-json',type=Path,help='已提取的只读数值轨迹缓存；和dataset-root二选一')
    p.add_argument('--fps',type=float)
    p.add_argument('--settings',type=Path)
    p.add_argument('--multiplier',type=float,default=2.)
    p.add_argument('--training-dataset-root',type=Path,help='缓存模式生成权重配置时必须指定真实训练数据集根目录')
    p.add_argument('--dataset-name',help='训练配置的data_name；指定后同时生成可启用的权重配置')
    p.add_argument('--spec',type=Path,default=ROOT/'configs/kinematics/robotwin_aloha_agilex_ee.json')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    import math
    if not math.isfinite(a.multiplier)or a.multiplier<=0:p.error('invalid multiplier')
    if a.dataset_name and not (a.dataset_root or a.training_dataset_root):
        p.error('weight config requires dataset-root or training-dataset-root')
    if bool(a.dataset_root)==bool(a.frames_json):p.error('choose dataset-root or frames-json')
    manifest=json.loads((ROOT/'refs/al_artifacts/al/task_splits_50/manifest.json').read_text())
    task=manifest['tasks'][a.task];ids=set(task['train_ids']+task['val_ids'])
    if a.frames_json:
        if a.fps is None:p.error('cached numeric rows require explicit fps')
        rows=json.loads(a.frames_json.read_text());fps=a.fps
    else:
        if a.output_dir.resolve().is_relative_to(a.dataset_root.resolve()):p.error('output must be outside source dataset')
        import pyarrow.dataset as ds
        info=json.loads((a.dataset_root/'meta/info.json').read_text());fps=info['fps']
        table=ds.dataset(str(a.dataset_root/'data'),format='parquet').to_table(columns=['episode_index','frame_index','timestamp','index','observation.state'],filter=ds.field('episode_index').isin(sorted(ids)),use_threads=False)
        rows=table.to_pylist()
    if {r['episode_index']for r in rows}!=ids:raise ValueError('target episode set differs from manifest')
    rule = json.loads(a.settings.read_text()) if a.settings else dict(task='pick_dual_bottles',rule_version='dual_bottle_horizontal_grasp_v1',settings={})
    if rule.get('task') != a.task:
        p.error('selection rule task must match --task; provide a task-specific --settings')
    settings=Settings(**rule['settings']);settings.validate()
    spec=json.loads(a.spec.read_text());summary,events,selected=select_rows(rows,spec,fps,settings)
    digest=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    summary.update(rule_version=rule['rule_version'],task=a.task,fps=fps,settings=asdict(settings),numeric_source_sha256=digest(sorted(rows,key=lambda r:(r['episode_index'],r['frame_index']))),spec_sha256=digest(spec),training_modified=False,automatic_selection=True)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for name,value in [('summary',summary),('events',events),('selected_frames',selected)]:
        (a.output_dir/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    if a.dataset_name:
        selected_file=a.output_dir/'selected_frames.json'
        weight_config=dict(version=1,normalization='sample_count',dataset_root=str((a.dataset_root or a.training_dataset_root).resolve()),task=a.task,rule_version=rule['rule_version'],dataset_name=a.dataset_name,multiplier=a.multiplier,selected_frames='selected_frames.json',selected_frames_sha256=hashlib.sha256(selected_file.read_bytes()).hexdigest())
        (a.output_dir/'weight_config.json').write_text(json.dumps(weight_config,ensure_ascii=False,indent=2)+'\n')
    if 'torch'in sys.modules:raise RuntimeError('selector must not import torch')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

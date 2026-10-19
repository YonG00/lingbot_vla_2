#!/usr/bin/env python3
"""鼠标辅助配方真实资产CPU验收：不加载模型、不启动训练或关机。"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from lingbotvla.auto_learning.catalog import TaskCatalog
from lingbotvla.auto_learning.config import AutoLearningConfig
from lingbotvla.auto_learning.ports import TrainRequest
from lingbotvla.auto_learning.sampling.auxiliary import AuxiliaryPool,AuxiliaryBatchSampler,full_catalog_for_auxiliary,preflight_pool
from lingbotvla.auto_learning.sampling.target_change_window import TargetChangeScanner
from experiment.robotwin.prepare_target_change_windows import prepare


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--check-dataset-chunks',action='store_true',help='另验证真实LeRobot读取器的episode末尾action chunk，仅CPU')
    p.add_argument('--check-runtime-assembly',action='store_true',help='独立进程验证真实Parquet列接入训练装配，避免与LeRobot依赖同时驻留')
    args,extra=p.parse_known_args(argv)
    if args.check_dataset_chunks and args.check_runtime_assembly:
        raise ValueError('run assembly and dataset chunk checks separately on low-memory instances')
    if args.output_dir.exists():raise FileExistsError('use a new audit output directory')
    spec=importlib.util.spec_from_file_location('mouse_all50',ROOT/'experiment/robotwin/recipes/20261018_place_mouse_pad_all50_9680_ee_lnfm.py')
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    recipe=wrapper.recipe
    parsed=recipe.parser().parse_args([
        '--model',str(wrapper.SOURCE_RUN/'checkpoints/global_step_500/hf_ckpt'),
        '--train-config',str(wrapper.SOURCE_RUN/'lingbotvla_cli.yaml'),
        '--train-all-episodes','--ee-position-weight','10',
        '--auxiliary-manifest',str(ROOT/'refs/data_manifests/mouse/20261019_train_v1/approved.json'),*extra])
    if parsed.execute or parsed.shutdown:raise ValueError('CPU preflight cannot execute training or shutdown')
    job,plan,_=recipe.build_job(parsed)
    cfg=AutoLearningConfig(**plan['config'])
    cfg.target_change_cache_dir=str(args.output_dir.resolve()/'primary_cache')
    full=full_catalog_for_auxiliary(parsed.manifest)
    primary=TaskCatalog({t:full.entry(t) for t in cfg.task_names},meta=full.meta,task_order=full.task_order).for_training(train_all_episodes=True)
    pool=AuxiliaryPool(cfg,full,primary,dataset_root=parsed.dataset_root,manifest_path=parsed.manifest)
    summary=preflight_pool(pool,parsed.dataset_root)
    primary=primary.attach_samples(pool.resolver)
    args.output_dir.mkdir(parents=True)
    prepare(cfg,parsed.dataset_root,json.loads(parsed.manifest.read_text()),cfg.task_names)
    entry=primary.entry(cfg.task_names[0])
    scanned=TargetChangeScanner(cfg,pool.resolver).scan(SimpleNamespace(task_name=entry.name,sample_ids=entry.train_sample_ids),entry)
    sampler=AuxiliaryBatchSampler(cfg,pool.resolver,primary,random.Random(cfg.seed),pool)
    req=TrainRequest(entry.name,probs=scanned.probs,batch_size=cfg.batch_size,new_slots=cfg.batch_size)
    prepared=sampler.prepare(req)
    primary_count=aux_count=weighted=0;eps=Counter();tasks=Counter();arms=Counter()
    for step in range(parsed.steps):
        comp=sampler.build(prepared,logical_step=step)
        primary_count+=len(comp.new);aux_count+=len(comp.auxiliary)
        if comp.old:raise AssertionError('unexpected Replay')
        for ref in comp.auxiliary:
            eps[ref.episode_id]+=1;tasks[ref.task]+=1
            arms[pool.records[ref.episode_id]['review']['mouse_arm']]+=1
            weighted+=ref.sample_id in pool.grasp_ids
    chunk_checks=0
    chunk_size=int(plan['training_config'].get('train',{}).get('chunk_size',50))
    if chunk_size < 1:raise ValueError('invalid source training chunk_size')
    if args.check_dataset_chunks:
        from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
        dataset=LeRobotDataset(repo_id=parsed.dataset_root.name,root=parsed.dataset_root,
             episodes=plan['train_ids'],load_image=False,
             delta_timestamps={'action':[i/pool.fps for i in range(chunk_size)]},download_videos=False)
        # 真实读取器必须按绝对帧坐标裁到episode末尾；不把下一episode混入chunk。
        for ep in plan['train_ids']:
            sid=pool.resolver.local_indices_of_episode(ep)[-1]
            item=dataset[sid]
            expected=dataset.hf_dataset[sid]['action']
            import torch
            if not torch.allclose(item['action'],expected.expand_as(item['action'])):
                raise AssertionError(f'action chunk crosses episode end: {ep}')
            if not item['action_is_pad'][1:].all():
                raise AssertionError(f'episode boundary padding missing: {ep}')
            chunk_checks+=1
    if args.check_runtime_assembly:
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
        from lingbotvla.auto_learning.real.build import build_auto_learning_parts
        rows=ds.dataset([str(p) for p in sorted((parsed.dataset_root/'data').rglob('*.parquet'))],format='parquet').to_table(
            columns=['index','episode_index','task_index'],
            filter=ds.field('episode_index').isin(plan['train_ids']),use_threads=False).to_pylist()
        rows.sort(key=lambda r:r['index'])
        metadata={r['episode_index']:r for p in sorted((parsed.dataset_root/'meta/episodes').rglob('*.parquet'))
                  for r in pq.read_table(p,columns=['episode_index','length','tasks','dataset_from_index','dataset_to_index']).to_pylist()
                  if r['episode_index'] in plan['train_ids']}
        labels=pq.read_table(parsed.dataset_root/'meta/tasks.parquet').to_pylist()
        labels.sort(key=lambda r:r['task_index'])
        dataset=SimpleNamespace(root=parsed.dataset_root,
            hf_dataset={k:[r[k] for r in rows] for k in ('index','episode_index','task_index')},
            meta=SimpleNamespace(episodes=metadata,tasks=[r['__index_level_0__'] for r in labels]))
        cfg.data_audit_file=str(args.output_dir.resolve()/'runtime_data_audit.json')
        config_file=args.output_dir/'runtime_config.json'
        config_file.write_text(json.dumps(cfg.to_dict(),ensure_ascii=False,indent=2)+'\n')
        parts=build_auto_learning_parts(config_path=str(config_file),
            manifest_path=str(parsed.manifest),train_dataset=dataset)
        if (parts.catalog.task_names()!=cfg.task_names or parts.auxiliary_pool is None
                or len(parts.resolver)!=summary['joint_frames']):
            raise AssertionError('actual training assembly lost auxiliary pool or joint resolver')
    summary.update(source_model=job['source_model'],steps=parsed.steps,
        primary_anchors=primary_count,auxiliary_anchors=aux_count,
        auxiliary_by_episode=dict(eps),auxiliary_by_task=dict(tasks),auxiliary_by_mouse_arm=dict(arms),
        auxiliary_grasp_candidate_anchors=weighted,primary_window_frames=len(scanned.probs),
        dataset_boundary_chunk_checks=chunk_checks,
        dataset_action_chunk_size=chunk_size,
        runtime_assembly_checked=args.check_runtime_assembly,
        note='CPU sampler/real dataset only; no model forward/backward; no GPU or training')
    (args.output_dir/'preflight.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('MOUSE_AUXILIARY_CPU_PREFLIGHT_COMPLETE',flush=True)


if __name__=='__main__':main()

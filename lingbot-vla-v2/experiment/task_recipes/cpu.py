"""真实数据与固定步预算DataLoader装配检查；只在GPU空闲时运行。"""
import json
from pathlib import Path
from experiment.task_recipes.runtime import load_run,pools_for,StreamDataset
from experiment.task_recipes.planning import PlannedSampler
from experiment.task_recipes.schema import save_interval,milestones


def check(out):
    from experiment.official_control.prepare_adjust_bottle import cpu_check
    c,p,groups=load_run(out)
    plan=json.loads((out/'plan.json').read_text())
    cfg=json.loads((Path(out)/'trainer_config.json').read_text())
    if c['evaluation'].get('reuse_training_model'):
        from experiment.task_recipes.resident_evaluation import validate_preprocessing
        validate_preprocessing(cfg)
    expected=milestones(c) if c['checkpoint'].get('mode','steps')=='task' else None
    assert cfg['train'].get('save_at_steps')==expected,'checkpoint schedule/config mismatch'
    def dataset_check(dataset,cfg):
        import torch
        from types import SimpleNamespace
        from lingbotvla import data
        from lingbotvla.data import data_loader
        data_loader.get_parallel_state=lambda:SimpleNamespace(dp_size=1,dp_rank=0,sp_enabled=False,sp_size=1)
        pools=pools_for(dataset,groups)
        sampler=PlannedSampler(p,pools,c['train']['seed'])
        stream=StreamDataset(dataset,len(sampler))
        loader=data.build_dataloader(dataset=stream,micro_batch_size=c['train']['micro_batch_size'],global_batch_size=p['global_batch_size'],
            dataloader_batch_size=p['global_batch_size'],max_seq_len=2048,train_steps=len(p['steps']),rmpad=False,rmpad_with_pos_ids=False,
            collate_fn=[data.VLADataCollatorWithPacking()],num_workers=0,prefetch_factor=None,pin_memory=False,drop_last=True,seed=c['train']['seed'],sampler=sampler)
        assert len(loader)==len(p['steps'])
        batch=next(iter(loader))
        assert len(batch)==c['train']['gradient_accumulation_steps']
        assert all(x['actions'].shape==(c['train']['micro_batch_size'],50,55) and torch.isfinite(x['actions']).all() for x in batch)
        assert isinstance(loader.state_dict(),dict)
        return dict(recipe_sampler=True,virtual_epoch_steps=len(loader),loader_microbatches=len(batch),real_frames=sum(map(len,pools.values())))
    cpu_check(Path(out),expected_steps=c['train']['steps'],expected_save_steps=save_interval(c),expected_fp32=False,
        expected_mixed=False,expected_export='bf16',expected_episodes=plan['episodes'],expected_gas=c['train']['gradient_accumulation_steps'],dataset_check=dataset_check)

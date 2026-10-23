"""真实数据与固定步预算DataLoader装配检查；只在GPU空闲时运行。"""
import json
from pathlib import Path
from experiment.task_recipes.runtime import load_run,pools_for,StreamDataset,sampling_weights_for,make_sampler
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
        sampler=make_sampler(dataset,c,p,groups,out)
        segment_audit={}
        if 'target_segments' in c['train'] or 'role_segments' in c['train']:
            import random
            from collections import Counter
            counters=Counter();roles=Counter()
            for row in p['steps']:
                indices,labels=sampler.draw_row(row,random.Random(f'{sampler.seed}:{row["step"]}'),counters)
                assert len(indices)==p['global_batch_size']
                roles.update(labels)
            segment_audit=dict(target_segment_sample_counts=dict(roles))
            print('TARGET_SEGMENT_QUOTA_PASS',json.dumps(segment_audit),flush=True)
        stream=StreamDataset(dataset,len(sampler))
        loader=data.build_dataloader(dataset=stream,micro_batch_size=c['train']['micro_batch_size'],global_batch_size=p['global_batch_size'],
            dataloader_batch_size=p['global_batch_size'],max_seq_len=2048,train_steps=len(p['steps']),rmpad=False,rmpad_with_pos_ids=False,
            collate_fn=[data.VLADataCollatorWithPacking()],num_workers=0,prefetch_factor=None,pin_memory=False,drop_last=True,seed=c['train']['seed'],sampler=sampler)
        assert len(loader)==len(p['steps'])
        batch=next(iter(loader))
        assert len(batch)==c['train']['gradient_accumulation_steps']
        assert all(x['actions'].shape==(c['train']['micro_batch_size'],50,55) and torch.isfinite(x['actions']).all() for x in batch)
        if 'role_segments' in c['train']:
            import random
            from collections import Counter
            tokens,_=sampler.draw_row(p['steps'][0],random.Random(f'{sampler.seed}:1'),Counter())
            expected=Counter(token[1] for token in tokens)
            actual=Counter(batch[0]['sample_loss_weight'].tolist())
            assert actual==expected,('role weights changed through loader/collator',actual,expected)
            segment_audit['first_batch_role_weight_counts']={str(k):v for k,v in actual.items()}
        import os
        sample_weight_audit={}
        if os.environ.get('LINGBOT_SAMPLE_WEIGHT_CONFIG'):
            from experiment.data_selection.weights import SelectionWeights
            policy=SelectionWeights(os.environ['LINGBOT_SAMPLE_WEIGHT_CONFIG'])
            multipliers=policy.multipliers
            weights=batch[0]['sample_loss_weight']
            assert weights.shape==(c['train']['micro_batch_size'],)
            assert set(weights.tolist())<=({1.}|multipliers),'sample weights absent or invalid'
            sample_weight_audit=dict(sample_weight_count=len(weights),sample_weight_selected=int((weights!=1).sum()),sample_weight_normalization='sample_count',sample_weight_multipliers=sorted(multipliers))
            if policy.action_overrides:
                from experiment.data_selection.inheritance import read_policies
                layouts=[json.loads(path.read_text())['arm_action_dimensions'] for path in read_policies(os.environ['LINGBOT_SAMPLE_WEIGHT_CONFIG']).values() if 'arm_action_dimensions' in json.loads(path.read_text())]
                assert layouts and all(v==layouts[0]for v in layouts)
                correction=batch[0]['action_loss_weight'];assert correction.shape==(len(weights),55)
                absolute=correction*weights[:,None]
                left=absolute[:,layouts[0]['left']];right=absolute[:,layouts[0]['right']]
                assert set(left.flatten().tolist())<={1.,2.,3.} and set(right.flatten().tolist())<={1.,2.}
                assert (left==3).any(),'first batch must exercise left3 action path'
                sample_weight_audit.update(action_weight_shape=list(correction.shape),left_fm_multipliers=sorted(set(left.flatten().tolist())),right_fm_multipliers=sorted(set(right.flatten().tolist())),shared_visual_multipliers=sorted(set(weights.tolist())))
            print('SAMPLE_WEIGHT_CPU_PASS',json.dumps(sample_weight_audit),flush=True)
        assert isinstance(loader.state_dict(),dict)
        return dict(recipe_sampler=True,virtual_epoch_steps=len(loader),loader_microbatches=len(batch),real_frames=sum(map(len,pools.values())),**sample_weight_audit,**segment_audit)
    cpu_check(Path(out),expected_steps=c['train']['steps'],expected_save_steps=save_interval(c),expected_fp32=False,
        expected_mixed=False,expected_export='bf16',expected_episodes=plan['episodes'],expected_gas=c['train']['gradient_accumulation_steps'],dataset_check=dataset_check)

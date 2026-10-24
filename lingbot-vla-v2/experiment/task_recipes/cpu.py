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
            sample_weight_audit=dict(sample_weight_count=len(weights),sample_weight_selected=int((weights!=1).sum()),sample_weight_normalization='weight_sum' if (policy.stage_tables or getattr(policy,'weight_sum',False)) else 'sample_count',sample_weight_multipliers=sorted(multipliers))
            if policy.action_overrides:
                from experiment.data_selection.inheritance import read_policies
                layouts=[json.loads(path.read_text())['arm_action_dimensions'] for path in read_policies(os.environ['LINGBOT_SAMPLE_WEIGHT_CONFIG']).values() if 'arm_action_dimensions' in json.loads(path.read_text())]
                assert layouts and all(v==layouts[0]for v in layouts)
                correction=batch[0]['action_loss_weight'];assert correction.shape==(len(weights),55)
                absolute=correction*weights[:,None]
                left=absolute[:,layouts[0]['left']];right=absolute[:,layouts[0]['right']]
                assert set(left.flatten().tolist())<={1.,2.,3.} and set(right.flatten().tolist())<={1.,2.}
                # Coverage may be zero when a staged/other task is the current main task.
                sample_weight_audit.update(action_weight_shape=list(correction.shape),left_fm_multipliers=sorted(set(left.flatten().tolist())),right_fm_multipliers=sorted(set(right.flatten().tolist())),shared_visual_multipliers=sorted(set(weights.tolist())))
            if policy.stage_tables or getattr(policy,'weight_sum',False):
                stage=batch[0]['stage_fm_weight'];pads=batch[0]['action_is_pad'].bool();mask=batch[0]['joint_mask'].bool() & ~pads[:,:,None]
                assert stage.shape==batch[0]['actions'].shape and torch.isfinite(stage).all()and (stage>=0).all()
                assert (stage[~mask & pads[:,:,None].expand_as(stage)]==0).all()
                for name in ['stage_current_depth_weight','stage_future_depth_weight','stage_current_video_weight','stage_future_video_weight']:
                    assert batch[0][name].shape==weights.shape and torch.isfinite(batch[0][name]).all()and (batch[0][name]>=0).all()
                assert batch[0]['stage_ids'].shape==(len(weights),50,2)and batch[0]['stage_arm_index'].shape==(len(weights),55)
                raw=dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index','frame_index'])
                terminal=[];stage_index=None;legacy_index=None;active_staged=set();scalar_classes={};category_boundaries=[]
                offset=int(bool(dataset._datasets[0].feature_transform.actions_convert_from_state))
                for i,row in enumerate(raw):
                    ep=int(row['episode_index']);frame=int(row['frame_index']);table=policy.stage_tables.get(ep)
                    if table:active_staged.add(ep)
                    if table is None and getattr(policy,'weight_sum',False):
                        scalar=float(policy.weight(policy.dataset_name,ep,frame))
                        if str(scalar)not in scalar_classes:
                            sample=dataset[i];pad=sample['action_is_pad'].bool()
                            expected=torch.tensor([[policy.weight(policy.dataset_name,ep,frame+offset+k)*v if not bool(pad[k])else 0. for v in (policy.relative_action_weights(policy.dataset_name,ep,frame+offset+k,55)or [1.]*55)]for k in range(50)])
                            assert torch.equal(sample['stage_fm_weight'],expected),'legacy importance must be queried at every action time'
                            assert float(sample['stage_current_depth_weight'])==scalar and float(sample['stage_current_video_weight'])==scalar
                            scalar_classes[str(scalar)]=dict(episode=ep,frame=frame,fm=scalar,current_visual=scalar)

                    if table and hasattr(table,'category_labels')and frame in {seg['start']for segs in table.episodes[ep]['arms'].values()for seg in segs}:
                        from experiment.data_selection.full_stages import derive_arm_layout
                        sample=dataset[i];pad=sample['action_is_pad'].bool();ft=dataset._datasets[0].feature_transform
                        layout=derive_arm_layout(ft.feature_config,ft.key_mapping,sample['action_joint_mask'].nonzero().flatten().tolist(),table.table['source_action_channels'])
                        for k in range(50):
                            for arm,dims in layout.items():
                                expected=table.frame_weights['fm'][ep][arm][frame+offset+k]if not bool(pad[k])else 0.
                                assert (sample['stage_fm_weight'][k,dims]==expected).all(),'category boundary/time weight mismatch'
                        for branch in ['current_depth','future_depth','current_video','future_video']:
                            target=frame+49 if branch.startswith('future')else frame
                            expected=max(table.frame_weights[branch][ep][arm][target]for arm in ['left','right'])if target<table.episodes[ep]['length']else 0.
                            assert float(sample['stage_'+branch+'_weight'])==expected,'visual target clock mismatch'
                        category_boundaries.append(dict(episode=ep,frame=frame))
                    if table and stage_index is None and table.labels[ep]['left'][frame]>=0 and table.frame_weights['fm'][ep]['left'][frame]==5:stage_index=i
                    if table is None and legacy_index is None and policy.weight(policy.dataset_name,ep,frame)>1:legacy_index=i
                    if table and frame==table.episodes[ep]['length']-1:
                        item=dataset[i];assert item['stage_fm_weight'].shape==(50,55)
                        assert item['action_is_pad'][1:].all()and (item['stage_fm_weight'][1:]==0).all()
                        assert float(item['stage_future_depth_weight'])==0 and float(item['stage_future_video_weight'])==0
                        terminal.append(ep)
                assert set(terminal)==active_staged,'every staged episode terminal must be checked'
                mixed_report={}
                if stage_index is not None and legacy_index is not None:
                    mixed=data.VLADataCollatorWithPacking()([dataset[stage_index],dataset[legacy_index]])
                    assert mixed['stage_fm_weight'].shape==(2,50,55)
                    assert (mixed['stage_ids'][0]>=0).all()and (mixed['stage_ids'][1]==-1).all()
                    scalar=float(mixed['sample_loss_weight'][1]);valid=mixed['joint_mask'][1].bool() & ~mixed['action_is_pad'][1].bool()[:,None]
                    legacy_row=raw[legacy_index];legacy_ep=int(legacy_row['episode_index']);legacy_frame=int(legacy_row['frame_index'])
                    expected=torch.tensor([[policy.weight(policy.dataset_name,legacy_ep,legacy_frame+offset+k)*v for v in (policy.relative_action_weights(policy.dataset_name,legacy_ep,legacy_frame+offset+k,55)or [1.]*55)]for k in range(50)])
                    assert torch.equal(mixed['stage_fm_weight'][1][valid],expected[valid]),'legacy replay per-action importance changed'
                    mixed_report=dict(samples=2,stage_current_depth=float(mixed['stage_current_depth_weight'][0]),stage_future_depth=float(mixed['stage_future_depth_weight'][0]),legacy_scalar=scalar,legacy_fm_importance_preserved=True)

                sample_weight_audit.update(category_boundary_samples_verified=len(category_boundaries),category_boundary_examples=category_boundaries[:8],weighting_unit='action_timestep',scalar_weight_classes_verified=scalar_classes,stage_normalization='weight_sum',stage_mixed_batch=mixed_report,stage_fm_shape=list(stage.shape),stage_active_weight_values=sorted(set(stage[mask].tolist())),stage_terminal_episodes=terminal,stage_future_depth_values=sorted(set(batch[0]['stage_future_depth_weight'].tolist())),stage_future_video_values=sorted(set(batch[0]['stage_future_video_weight'].tolist())))
            print('SAMPLE_WEIGHT_CPU_PASS',json.dumps(sample_weight_audit),flush=True)
        assert isinstance(loader.state_dict(),dict)
        return dict(recipe_sampler=True,virtual_epoch_steps=len(loader),loader_microbatches=len(batch),real_frames=sum(map(len,pools.values())),**sample_weight_audit,**segment_audit)
    cpu_check(Path(out),expected_steps=c['train']['steps'],expected_save_steps=save_interval(c),expected_fp32=False,
        expected_mixed=False,expected_export='bf16',expected_episodes=plan['episodes'],expected_gas=c['train']['gradient_accumulation_steps'],dataset_check=dataset_check)

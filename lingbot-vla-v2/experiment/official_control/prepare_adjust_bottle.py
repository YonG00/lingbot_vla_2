"""从仓库初始训练配置准备标准方法对照；不启动训练、不删除存档。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def validate_identity():
    identity = json.loads((HERE / 'snapshot_identity.json').read_text())
    for relative, expected in identity['restored_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    return identity


def prepare(args):
    import yaml
    identity = validate_identity()
    cfg = yaml.safe_load((HERE / 'robotwin_snapshot.yaml').read_text())
    source = yaml.safe_load((args.source_run / 'lingbotvla_cli.yaml').read_text())
    model = args.source_run / 'checkpoints/global_step_9680/hf_ckpt'
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    # Validate actual shard headers, without loading weights or importing torch.
    seen = {}
    for name in set(index['weight_map'].values()):
        path = model / name
        assert path.resolve().parent == model.resolve()
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            assert 0 < size < min(path.stat().st_size, 32 * 1024 * 1024)
            header = json.loads(stream.read(size))
        for key, value in header.items():
            if key == '__metadata__':
                continue
            assert key not in seen
            assert value['dtype'] in ('BF16', 'F32', 'I64', 'I32', 'BOOL')
            assert 0 <= value['data_offsets'][0] <= value['data_offsets'][1] <= path.stat().st_size - 8 - size
            seen[key] = name
    assert seen == index['weight_map']
    manifest = json.loads(args.manifest.read_text())
    task = manifest['tasks']['adjust_bottle']
    ids = sorted(task['train_ids'] + task['val_ids'])
    assert len(ids) == len(set(ids)) == task['n_total'] == 50
    assert args.micro > 0 and not args.out.exists(), 'output must be fresh'
    cfg['model'].update(model_path=str(model), config_path=str(model / 'config.json'),
                        tokenizer_path=source['model']['tokenizer_path'], incremental_training=False)
    cfg['data'].update(train_path=str(args.out / 'datasets.txt'),
                       norm_stats_file=source['data']['norm_stats_file'],
                       episode_ids_file=str(args.out / 'train_ids.json'), image_augment=False,
                       robot_config_root=str(ROOT / 'configs/robot_configs'), num_workers=8)
    align = cfg['train']['align_params']
    old_align = source['train']['align_params']
    for key in ('moge_path', 'morgbd_path'):
        align['depth'][key] = old_align['depth'][key]
    for key in ('ckpt_path', 'config_path'):
        align['video'][key] = old_align['video'][key]
    # Keep original optimization/loss defaults; explicitly clear all experimental routing.
    cfg['train'].update(output_dir=str(args.out), micro_batch_size=args.micro,
                        global_batch_size=args.micro, gradient_accumulation_steps=1,
                        max_steps=500, num_train_epochs=501, save_steps=250, save_epochs=0,
                        enable_resume=False, load_checkpoint_path=None, step_offset=0,
                        reset_step_counter=True, train_expert_only=False, use_lora=False,
                        freeze_vision_encoder=False, enable_gradient_checkpointing=True,
                        enable_mixed_precision=True, enable_fp32=True, action_fp32=True,
                        use_compile=True, auto_learning=None, auto_learning_manifest=None,
                        auto_learning_baseline='', low_noise_fm_enabled=False,
                        ee_fk_loss_enabled=False, ee_fk_eval_enabled=False,
                        ee_fk_position_loss_weight=0., ee_fk_orientation_loss_weight=0.,
                        ee_fk_gripper_loss_weight=0., open_loop_eval_steps=0,
                        data_parallel_replicate_size=1, data_parallel_shard_size=1,
                        save_hf_weights=True, hf_export_dtype='fp32', dcp_save_mode='hf_only',
                        async_save_hf_weights=False, hf_pass_interval=0, seed=3,
                        disk_guard=True, disk_guard_margin=1.1, disk_check_interval=50)
    resources = [Path(cfg['data']['norm_stats_file']), Path(cfg['model']['tokenizer_path']),
                 args.dataset / 'meta', *[Path(align['depth'][k]) for k in ('moge_path', 'morgbd_path')],
                 *[Path(align['video'][k]) for k in ('ckpt_path', 'config_path')]]
    assert all(path.exists() for path in resources), [str(p) for p in resources if not p.exists()]
    # Budget two FP32 HF checkpoints with 10% margin and 3 GiB operational space.
    import shutil
    required = int(index['metadata']['total_size']) * 2 * 2 * 1.1 + 3 * 2**30
    assert shutil.disk_usage(args.out.parent).free >= required, 'insufficient disk for 250/500 FP32 HF'
    args.out.mkdir(parents=True)
    dump(args.out / 'trainer_config.json', cfg)
    dump(args.out / 'train_ids.json', ids)
    dump(args.out / 'evaluation_seeds.json', list(range(100000, 100010)))
    (args.out / 'datasets.txt').write_text(f'robotwin {args.dataset}\n')
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1',
               '--master_port=29531', 'tasks/vla/train_lingbotvla.py', str(args.out / 'trainer_config.json')]
    dump(args.out / 'plan.json', dict(task='adjust_bottle', episodes=50, steps=500,
         hf_steps=[250, 500], micro=args.micro, gas=1, batch_gpu_verified=False,
         train_command=command, cwd=str(ROOT), snapshot=identity, source_model=str(model),
         training_started=False, expected_frames=task['train_frames'] + task['val_frames'],
         normalization_sha256=hashlib.sha256(resources[0].read_bytes()).hexdigest(),
         required_checkpoint_budget_GiB=required / 2**30,
         evaluation=dict(labels=['9680', '250', '500'], seeds=list(range(100000, 100010)),
                         task_config='demo_clean', action_chunk=50, dtype='fp32'),
         limitations=['9680 is previously trained', 'initial repository snapshot is not verified upstream',
                      'data and checkpoint compatibility fixes retained', 'single-GPU batch differs from 32-GPU original']))
    # Explicit launch file. It is never executed by this preparation tool.
    script = '#!/usr/bin/env bash\nset -euo pipefail\n'
    script += 'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
    script += 'export QWEN3VL_PATH=' + shlex.quote(cfg['model']['tokenizer_path']) + '\n'
    script += 'export AL_DYNAMO_CACHE_SIZE_LIMIT=0\n'  # disable our cache-limit override
    script += 'unset AL_VLM_ATTENTION LINGBOT_PRECISION_AUDIT LINGBOTVLA_TRAINING_CONFIG\n'
    script += 'cd ' + shlex.quote(str(ROOT)) + '\n'
    script += 'exec ' + shlex.join(command) + '\n'
    (args.out / 'train.sh').write_text(script)
    print('PREPARED', args.out, 'NO_TRAINING_STARTED', flush=True)


def cpu_check(out, *, expected_steps=500, expected_fp32=True, expected_mixed=True, expected_export='fp32', expected_episodes=50, expected_save_steps=250, dataset_check=None):
    import ast
    import types
    import typing
    import dataclasses
    import torch
    # AutoDL no-GPU mode can have only 0.5 CPU quota. Hundreds of default
    # OpenMP threads make tiny image transforms slower rather than faster.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.data import build_vla_dataset
    from lingbotvla.utils import arguments as argument_utils
    # Importing the full trainer initializes CUDA-specific group-GEMM. Compile only
    # its actual dataclass definitions; execute the same parser on CPU, no stubs.
    trainer_tree = ast.parse((ROOT / 'tasks/vla/train_lingbotvla.py').read_text())
    selected = [n for n in trainer_tree.body if isinstance(n, ast.ClassDef)
                and n.name in ('MyTrainingArguments', 'MyDataArguments', 'Arguments')]
    assert len(selected) == 3
    module = types.ModuleType('_official_cpu_argument_contract')
    module.__dict__.update(vars(typing))
    module.__dict__.update(vars(argument_utils))
    module.__dict__.update(dataclass=dataclasses.dataclass, field=dataclasses.field)
    # dataclasses and get_type_hints look up the defining module by name.
    module.__dict__['__name__'] = '_official_cpu_argument_contract'
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<trainer-argument-definitions>', 'exec'), module.__dict__)
    for key, value in {'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1'}.items():
        os.environ.setdefault(key, value)
    old_argv = sys.argv
    try:
        sys.argv = [str(__file__), str(out / 'trainer_config.json')]
        parsed = argument_utils.parse_args(module.Arguments)
    finally:
        sys.argv = old_argv
    validate_identity()
    cfg = json.loads((out / 'trainer_config.json').read_text())
    t = cfg['train']
    assert not any(t[k] for k in ('auto_learning', 'low_noise_fm_enabled', 'ee_fk_loss_enabled', 'ee_fk_eval_enabled', 'use_lora', 'train_expert_only', 'freeze_vision_encoder'))
    assert t['max_steps'] == expected_steps and t['save_steps'] == expected_save_steps and t['gradient_accumulation_steps'] == 1
    assert t['enable_mixed_precision'] == expected_mixed and t['enable_fp32'] == expected_fp32
    assert t['hf_export_dtype'] == expected_export
    assert parsed.train.auto_learning is None and parsed.train.global_batch_size == t['micro_batch_size']
    config = LingbotVLAV2Config(**{**cfg['model'], **t})
    # Execute original sampling method and prove no replacement RNG path is called.
    torch.manual_seed(3)
    model_tree = ast.parse((ROOT / 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py').read_text())
    flow = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlowMatching')
    sample_time = next(n for n in flow.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_time')
    utils_tree = ast.parse((ROOT / 'lingbotvla/models/vla/lingbot_vla/utils.py').read_text())
    sample_beta_node = next(n for n in utils_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_beta')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[sample_beta_node, sample_time], type_ignores=[]), '<original-FM-functions>', 'exec'), scope)
    observed = scope['sample_time'](None, 128, 'cpu')
    torch.manual_seed(3)
    expected = (scope['sample_beta'](1.5, 1.0, 128, 'cpu') * .999 + .001).float()
    assert torch.equal(observed, expected)
    d = dict(vars(parsed.data), chunk_size=50)
    # Use parser output unchanged: FeatureInfo expects dictionary literals as
    # strings, matching the real trainer's List[str] contract.
    processor = build_processor(cfg['model']['tokenizer_path'])
    dataset = build_vla_dataset(types.SimpleNamespace(**d), types.SimpleNamespace(**cfg['model']),
                                config, processor, use_depth_align=True)
    dataset.strict_getitem = True
    ids = json.loads((out / 'train_ids.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    assert len(dataset) == plan['expected_frames']
    raw = dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index', 'frame_index'])
    counts = {}
    boundaries = []
    previous = None
    for idx, row in enumerate(raw):
        ep = int(row['episode_index'])
        counts[ep] = counts.get(ep, 0) + 1
        if ep != previous:
            boundaries.append(idx)
        previous = ep
    assert sorted(counts) == ids and len(counts) == expected_episodes
    checked = []
    for idx in boundaries:
        sample = dataset[idx]
        assert sample['actions'].shape == sample['joint_mask'].shape == (50, 55)
        assert torch.isfinite(sample['actions']).all() and torch.isfinite(sample['state']).all()
        checked.append(dict(index=idx, episode=int(raw[idx]['episode_index']),
                            frame=int(raw[idx]['frame_index'])))
        print('CPU_EPISODE_PASS', len(checked), '/', expected_episodes, checked[-1], flush=True)
    extra = dataset_check(dataset, cfg) if dataset_check else {}
    report = dict(state='cpu_pass', episodes=expected_episodes, frames=len(dataset), boundary_samples=checked,
                  original_beta_time_exact=True, al_enabled=False, ee_enabled=False,
                  model_initialized=False, optimizer_steps=0, gpu_tested=False, **extra)
    dump(out / 'cpu_report.json', report)
    print('CPU_CONTRACT_PASS', json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('/data/outputs/official_adjust_bottle_9680_add500_20261020'))
    p.add_argument('--source-run', type=Path, default=Path('/data/outputs/al_v79_poolfix_7300'))
    p.add_argument('--dataset', type=Path, default=Path('/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30'))
    p.add_argument('--manifest', type=Path, default=ROOT / 'refs/al_artifacts/al/task_splits_50/manifest.json')
    p.add_argument('--micro', type=int, default=16, help='待GPU验证的候选值；GAS固定1')
    p.add_argument('--cpu-check', action='store_true')
    args = p.parse_args()
    if args.cpu_check:
        cpu_check(args.out)
    else:
        prepare(args)


if __name__ == '__main__':
    main()

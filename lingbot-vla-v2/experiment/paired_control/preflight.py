"""CPU data/FK contracts for the enhanced run, executed only after training ends."""
import ast
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    out = Path(sys.argv[1])
    import torch
    torch.set_num_threads(1)
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.data import build_vla_dataset
    from lingbotvla.losses.ee_fk import build_action_layout, build_action_norm_spec, DifferentiableEEFKSolver, compute_ee_fk_losses
    from experiment.paired_control.methods import aligned_weights
    from lingbotvla.utils.flow_time import low_noise_time_mixture
    cfg = json.loads((out / 'trainer_config.json').read_text())
    methods = json.loads((out / 'methods.json').read_text())
    source = Path(methods['source_run'])
    import hashlib
    assert hashlib.sha256((source / 'trainer_config.json').read_bytes()).hexdigest() == methods['source_config_sha256']
    # Use real parser dataclasses extracted without importing CUDA-only trainer ops.
    from lingbotvla.utils import arguments as utils
    import dataclasses, typing
    module = types.ModuleType('_paired_cpu_args')
    module.__dict__.update(vars(typing));module.__dict__.update(vars(utils))
    module.__dict__.update(dataclass=dataclasses.dataclass, field=dataclasses.field)
    module.__dict__['__name__'] = '_paired_cpu_args';sys.modules[module.__name__] = module
    tree = ast.parse((ROOT / 'tasks/vla/train_lingbotvla.py').read_text())
    classes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name in ('Arguments', 'MyTrainingArguments', 'MyDataArguments')]
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<real trainer argument definitions>', 'exec'), module.__dict__)
    import os
    os.environ.update(LOCAL_RANK='0', RANK='0', WORLD_SIZE='1')
    sys.argv = [sys.argv[0], str(out / 'trainer_config.json')]
    parsed = utils.parse_args(module.Arguments)
    config = LingbotVLAV2Config(**{**cfg['model'], **cfg['train']})
    assert config.ee_fk_loss_enabled and config.low_noise_fm_enabled
    dataset = build_vla_dataset(types.SimpleNamespace(**dict(vars(parsed.data), chunk_size=50)),
          types.SimpleNamespace(**cfg['model']), config, build_processor(cfg['model']['tokenizer_path']), use_depth_align=True)
    dataset.strict_getitem = True
    raw = dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index', 'index', 'frame_index'])
    coords = [(int(r['episode_index']), int(r['index'])) for r in raw]
    cache = json.loads(Path(methods['window_cache']).read_text())
    weights = aligned_weights(cache, coords)
    assert len(weights) == len(dataset) == 7156 and set(weights) <= {1., 2.}
    episodes = sorted({x[0] for x in coords})
    assert episodes == json.loads((out / 'train_ids.json').read_text()) and len(episodes) == 50
    boundaries = [i for i, r in enumerate(raw) if int(r['frame_index']) == 0]
    first = None
    for index in boundaries:
        sample = dataset[index]
        assert sample['actions'].shape == sample['joint_mask'].shape == (50, 55)
        assert torch.isfinite(sample['actions']).all()
        first = sample if first is None else first
    ft = list(dataset.feature_transforms.values())[0]
    layout, norm = build_action_layout(ft), build_action_norm_spec(ft)
    solver = DifferentiableEEFKSolver(config.ee_fk_spec)
    targets = first['actions'][None]
    predicted = (targets + .01).requires_grad_(True)
    value, _ = compute_ee_fk_losses(solver, predicted, targets, action_layout=layout,
        action_norm_spec=norm, position_weight=3., orientation_weight=.8, gripper_weight=1.5,
        valid_time=~first['action_is_pad'][None].bool())
    value.backward()
    assert torch.isfinite(value) and predicted.grad is not None and torch.isfinite(predicted.grad).all()
    time = torch.ones(1024) * .8
    before = torch.random.get_rng_state();assert low_noise_time_mixture(time, False) is time
    assert torch.equal(before, torch.random.get_rng_state())
    assert (low_noise_time_mixture(time, True) < .2).any()
    # Exercise the real weighted sampler -> StatefulDataLoader -> VLA collator
    # on CPU before any model forward. Only distributed world metadata is local.
    from lingbotvla import data
    from lingbotvla.data import data_loader
    from experiment.paired_control.train import install_window_sampler
    data_loader.get_parallel_state = lambda: types.SimpleNamespace(
        dp_size=1, dp_rank=0, sp_enabled=False, sp_size=1)
    install_window_sampler(methods)
    loader = data.build_dataloader(dataset=dataset, micro_batch_size=28,
        global_batch_size=28, dataloader_batch_size=28, max_seq_len=2048,
        train_steps=255, rmpad=False, rmpad_with_pos_ids=False,
        collate_fn=[data.VLADataCollatorWithPacking()], num_workers=0,
        prefetch_factor=None, pin_memory=False, drop_last=True, seed=cfg['train']['seed'])
    loader.set_epoch(0)
    batch = next(iter(loader))
    assert len(batch) == 1 and batch[0]['actions'].shape == (28, 50, 55)
    assert torch.isfinite(batch[0]['actions']).all()
    assert isinstance(loader.state_dict(), dict)
    report = dict(state='cpu_pass', episodes=50, frames=len(dataset), boundary_samples=len(boundaries),
                  fk_backward_finite=True, ln_fm_disabled_rng_unchanged=True,
                  window_summary=cache['summary'], source_config_unchanged=True,
                  weighted_loader_batch_shape=[28, 50, 55], loader_state_dict=True)
    (out / 'cpu_report.json').write_text(json.dumps(report, indent=2));print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

"""Opt-in methods on the ordinary trainer; no AL task scheduler or early stop."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.paired_control.methods import aligned_weights, make_sampler, validate_flags


def install_window_sampler(methods):
    if not methods['window_sampling']:
        return  # preserve ordinary loader and RNG path exactly
    import torch
    from lingbotvla import data
    original = data.build_dataloader
    cache = json.loads(Path(methods['window_cache']).read_text())
    def build(*args, **kwargs):
        dataset = kwargs.get('dataset', args[0] if args else None)
        assert kwargs.get('sampler') is None and len(dataset._datasets) == 1
        raw = dataset._datasets[0].dataset.hf_dataset.select_columns(['episode_index', 'index'])
        coordinates = [(int(r['episode_index']), int(r['index'])) for r in raw]
        weights = aligned_weights(cache, coordinates)
        def factory(values, seed):
            generator = torch.Generator().manual_seed(seed)
            return torch.utils.data.WeightedRandomSampler(values, len(values), replacement=True, generator=generator)
        kwargs['sampler'] = make_sampler(True, factory, weights, kwargs.get('seed', 0))
        print('[paired_window] ' + json.dumps(dict(samples=len(weights), min_weight=min(weights),
              max_weight=max(weights), cache=methods['window_cache'], **cache['summary'])), flush=True)
        return original(*args, **kwargs)
    data.build_dataloader = build


def install_smoke_audit():
    import torch
    from lingbotvla.utils import precision_audit as audit
    from experiment.official_control.gpu_smoke import instrument_fp32_action_output
    class Audit(audit.PrecisionAudit):
        def __init__(self, model, optimizer, path):
            super().__init__(model, optimizer, path)
            self.restore = instrument_fp32_action_output(model, self.data['forward_samples'])
            self.data['feature_flags'] = dict(ee_loss=model.config.ee_fk_loss_enabled,
                                             ln_fm=model.config.low_noise_fm_enabled)
            def record(module, args, result):
                if isinstance(result, tuple) and len(result) > 6 and isinstance(result[6], dict):
                    self.data['ee_metrics'] = {k: float(v.detach().float()) if torch.is_tensor(v) else v
                                              for k, v in result[6].items() if k.startswith('ee_fk/')}
            self.handles.append(model.register_forward_hook(record))
        def before_step(self):
            super().before_step()
            if not self.done:
                assert all(torch.isfinite(p.grad).all() for p in self.parameters()
                           if p.grad is not None), 'nonfinite enhanced gradient'
        def after_step(self):
            if self.done:
                return
            self.data.update(parameter_numel_by_dtype=audit.tensor_inventory(self.parameters()),
                optimizer_state_numel_by_dtype=audit.tensor_inventory(audit.state_tensors(self.optimizer.state)),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            self.path.write_text(json.dumps(self.data, indent=2))
            assert set(self.data['parameter_numel_by_dtype']) == {'torch.float32'}
            assert set(self.data['gradient_numel_by_dtype']) == {'torch.float32'}
            for group in ('visual', 'language_model', 'qwen_expert', 'action_out'):
                values = [r for n, r in self.data['forward_samples'].items() if group in n]
                assert values and all(set(r['outputs']) == {'torch.float32'} for r in values)
            if self.data['feature_flags']['ee_loss']:
                assert self.data.get('ee_metrics'), 'EE enabled but no actual FK metrics produced'
                import math
                assert all(math.isfinite(v) for v in self.data['ee_metrics'].values())
                assert any(v > 0 for k, v in self.data['ee_metrics'].items() if k.endswith('_loss')), 'EE loss inactive'
            for handle in self.handles:
                handle.remove()
            self.restore()
            self.done = True
            print('PAIRED_FP32_AUDIT_PASS', json.dumps(self.data), flush=True)
    audit.PrecisionAudit = Audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--methods', type=Path, required=True)
    p.add_argument('--smoke-out', type=Path)
    for name in ('ee-loss', 'window-sampling', 'ln-fm'):
        p.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    flags = validate_flags(args.ee_loss, args.window_sampling, args.ln_fm)
    methods = json.loads(args.methods.read_text())
    assert all(methods[k] == v for k, v in flags.items()), 'CLI differs from frozen method flags'
    config = json.loads(args.config.read_text())
    t = config['train']
    assert t['ee_fk_loss_enabled'] == flags['ee_loss'] and t['low_noise_fm_enabled'] == flags['ln_fm']
    assert t['auto_learning'] is None and not t['enable_resume']
    path = args.config
    if args.smoke_out:
        args.smoke_out.mkdir(parents=True, exist_ok=False)
        t.update(output_dir=str(args.smoke_out), max_steps=3, save_steps=0, save_epochs=0,
                 smoke_no_checkpoint=True, save_hf_weights=False, dcp_save_mode='always', disk_guard=False)
        path = args.smoke_out / 'trainer_config.json'
        path.write_text(json.dumps(config, indent=2))
        os.environ['LINGBOT_PRECISION_AUDIT'] = str(args.smoke_out / 'precision_audit.json')
        install_smoke_audit()
    install_window_sampler(methods)
    os.environ['QWEN3VL_PATH'] = config['model']['tokenizer_path']
    os.environ['AL_DYNAMO_CACHE_SIZE_LIMIT'] = '0'
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    os.chdir(ROOT)
    sys.argv = ['tasks/vla/train_lingbotvla.py', str(path)]
    print('PAIRED_METHODS', json.dumps(flags), flush=True)
    runpy.run_path(sys.argv[0], run_name='__main__')
    if args.smoke_out:
        import torch
        report = dict(state='three_step_probe_complete', flags=flags, steps=3,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        (args.smoke_out / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

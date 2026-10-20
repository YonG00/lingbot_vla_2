"""显式启动三步FP32容量探针；不保存模型，不自动启动正式训练。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def instrument_fp32_action_output(policy, samples):
    """Observe the original functional linear, which bypasses nn.Linear hooks."""
    flow = policy.model
    original = flow._fp32_linear
    projection = flow.action_out_proj

    def observed(module, inputs):
        output = original(module, inputs)
        if module is projection:
            samples['model.action_out_proj.functional'] = dict(
                weight=str(module.weight.dtype), inputs=[str(inputs.dtype)],
                outputs=[str(output.dtype)])
        return output

    flow._fp32_linear = observed
    def restore():
        del flow._fp32_linear  # reveal the original class staticmethod again
    return restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='已通过CPU检查的正式输出目录')
    parser.add_argument('--smoke-out', type=Path, required=True, help='全新探针输出目录')
    args = parser.parse_args()
    report = json.loads((args.out / 'cpu_report.json').read_text())
    assert report['state'] == 'cpu_pass'
    from experiment.official_control.prepare_adjust_bottle import validate_identity
    validate_identity()
    assert not args.smoke_out.exists(), 'refuse to overwrite a probe'
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    assert not active.strip(), 'GPU is busy'
    cfg = json.loads((args.out / 'trainer_config.json').read_text())
    assert cfg['train']['enable_fp32'] and cfg['train']['enable_mixed_precision']
    assert cfg['train']['auto_learning'] is None
    args.smoke_out.mkdir(parents=True)
    cfg['train'].update(output_dir=str(args.smoke_out), max_steps=3,
                        save_steps=0, save_epochs=0, smoke_no_checkpoint=True,
                        save_hf_weights=False, dcp_save_mode='always', disk_guard=False)
    config = args.smoke_out / 'trainer_config.json'
    config.write_text(json.dumps(cfg, indent=2))
    os.environ['QWEN3VL_PATH'] = cfg['model']['tokenizer_path']
    os.environ['LINGBOT_PRECISION_AUDIT'] = str(args.smoke_out / 'precision_audit.json')
    for key in ('AL_VLM_ATTENTION', 'LINGBOTVLA_TRAINING_CONFIG'):
        os.environ.pop(key, None)
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'):
        os.environ.setdefault(key, '1' if key == 'WORLD_SIZE' else '0')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29532')
    import torch
    from lingbotvla.utils import precision_audit as audit

    class FP32Audit(audit.PrecisionAudit):
        # Existing audit requires BF16 expert outputs; this control explicitly
        # checks FP32, using the same non-invasive forward hooks.
        def __init__(self, model, optimizer, path):
            super().__init__(model, optimizer, path)
            self.restore_functional = instrument_fp32_action_output(model, self.data['forward_samples'])

        def before_step(self):
            super().before_step()
            if not self.done:
                assert all(torch.isfinite(p.grad).all() for p in self.parameters() if p.grad is not None), 'nonfinite gradient'

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
            for bucket in ('qwen_expert', 'visual', 'language_model', 'action_out'):
                matches = [r for n, r in self.data['forward_samples'].items() if bucket in n]
                assert matches and all(set(r['outputs']) == {'torch.float32'} for r in matches), (bucket, matches)
            for handle in self.handles:
                handle.remove()
            self.restore_functional()
            self.done = True
            print('FP32_AUDIT_PASS', json.dumps(self.data), flush=True)

    audit.PrecisionAudit = FP32Audit
    os.chdir(ROOT)
    sys.argv = ['tasks/vla/train_lingbotvla.py', str(config)]
    runpy.run_path(sys.argv[0], run_name='__main__')
    assert (args.smoke_out / 'precision_audit.json').is_file()
    result = dict(state='three_step_probe_complete', gpu=torch.cuda.get_device_name(),
                  micro=cfg['train']['micro_batch_size'], gas=1,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  model_saved=False, formal_training_started=False)
    (args.smoke_out / 'summary.json').write_text(json.dumps(result, indent=2))
    print('GPU_PROBE_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

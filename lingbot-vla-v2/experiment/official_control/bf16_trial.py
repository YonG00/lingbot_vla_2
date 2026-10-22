"""官方方法BF16参数/计算实验；准备、CPU检查、三步审计分别显式执行。"""
import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def observe_action_output(model, samples, disable):
    """Keep dtype observation outside compiled graphs without changing tensors."""
    projection = model.model.action_out_proj
    name = next(n for n, m in model.named_modules() if m is projection)
    def record(module, inputs, output):
        samples[name] = dict(weight=str(module.weight.dtype),
            inputs=[str(x.dtype) for x in inputs], outputs=[str(output.dtype)])
    return projection.register_forward_hook(disable(record))


def prepare(source, out):
    import shutil
    from experiment.official_control.prepare_adjust_bottle import validate_identity
    validate_identity()
    baseline = json.loads((Path(__file__).parent / 'baselines/adjust_bottle_bf16_200.json').read_text()) if source is None else None
    cfg = baseline['trainer_config'] if baseline else json.loads((source / 'trainer_config.json').read_text())
    assert cfg['train']['auto_learning'] is None
    assert 'global_step_9680' in cfg['model']['model_path']
    assert shutil.disk_usage(out.parent).free > 17 * 2**30
    out.mkdir(exist_ok=False)
    if source is None:
        (out / 'datasets.txt').write_text(baseline['datasets'])
        (out / 'train_ids.json').write_text(json.dumps(baseline['episode_ids']))
    else:
        for name in ('datasets.txt', 'train_ids.json'):
            shutil.copyfile(source / name, out / name)
    cfg['data'].update(train_path=str(out / 'datasets.txt'), episode_ids_file=str(out / 'train_ids.json'),
                       robot_config_root=str(ROOT / 'configs/robot_configs'))
    cfg['train'].update(output_dir=str(out), micro_batch_size=56, global_batch_size=56,
                        gradient_accumulation_steps=1, max_steps=200,
                        enable_mixed_precision=False, enable_fp32=False, action_fp32=False, hf_export_dtype='bf16')
    t = cfg['train']
    assert t['save_steps'] == 250 and t['hf_export_dtype'] == 'bf16'
    assert not t['ee_fk_loss_enabled'] and not t['low_noise_fm_enabled']
    (out / 'trainer_config.json').write_text(json.dumps(cfg, indent=2))
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1',
               '--master_port=29545', 'tasks/vla/train_lingbotvla.py', str(out / 'trainer_config.json')]
    (out / 'train.sh').write_text('#!/bin/bash\nset -euo pipefail\n'
        'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AL_DYNAMO_CACHE_SIZE_LIMIT=0\n'
        'unset LINGBOT_PRECISION_AUDIT LINGBOTVLA_TRAINING_CONFIG AL_VLM_ATTENTION\n'
        'export QWEN3VL_PATH=' + shlex.quote(cfg['model']['tokenizer_path']) + '\n'
        'cd ' + shlex.quote(str(ROOT)) + '\nexec ' + shlex.join(command) + '\n')
    (out / 'plan.json').write_text(json.dumps(dict(expected_frames=7156, source_model=cfg['model']['model_path'],
        micro=56, gas=1, steps=200, save_steps=250, final_checkpoint=200,
        parameter_dtype='bf16', compute_dtype='bf16', hf_export_dtype='bf16'), indent=2))


def smoke(out,steps=3):
    import subprocess
    if type(steps) is not int or steps<1:raise ValueError('positive probe steps required')
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip()
    assert json.loads((out / 'cpu_report.json').read_text())['state'] == 'cpu_pass'
    probe = out / 'probe'
    probe.mkdir(exist_ok=False)
    cfg = json.loads((out / 'trainer_config.json').read_text())
    cfg['train'].update(output_dir=str(probe), max_steps=steps, save_steps=0, save_epochs=0,
        smoke_no_checkpoint=True, save_hf_weights=False, dcp_save_mode='always', disk_guard=False)
    config = probe / 'trainer_config.json'
    config.write_text(json.dumps(cfg, indent=2))
    os.environ['LINGBOT_PRECISION_AUDIT'] = str(probe / 'precision_audit.json')
    os.environ['QWEN3VL_PATH'] = cfg['model']['tokenizer_path']
    import torch
    from lingbotvla.utils import precision_audit as audit
    class BF16Audit(audit.PrecisionAudit):
        def __init__(self, model, optimizer, path):
            super().__init__(model, optimizer, path)
            self.handles.append(observe_action_output(model, self.data['forward_samples'], torch.compiler.disable))

        def before_step(self):
            super().before_step()
            if not self.done:
                assert all(torch.isfinite(p.grad).all() for p in self.parameters() if p.grad is not None)
        def after_step(self):
            if self.done:
                return
            self.data.update(parameter_numel_by_dtype=audit.tensor_inventory(self.parameters()),
                optimizer_state_numel_by_dtype=audit.tensor_inventory(audit.state_tensors(self.optimizer.state)))
            assert set(self.data['parameter_numel_by_dtype']) == {'torch.bfloat16'}, self.data
            assert set(self.data['gradient_numel_by_dtype']) == {'torch.bfloat16'}, self.data
            for bucket in ('qwen_expert', 'visual', 'language_model', 'action_out'):
                rows = [v for k, v in self.data['forward_samples'].items() if bucket in k]
                assert rows and all(set(v['outputs']) == {'torch.bfloat16'} for v in rows), (bucket, rows)
            self.path.write_text(json.dumps(self.data, indent=2))
            for handle in self.handles:
                handle.remove()
            self.done = True
            print('BF16_AUDIT_PASS', json.dumps(self.data), flush=True)
    audit.PrecisionAudit = BF16Audit
    os.chdir(ROOT)
    sys.argv = ['tasks/vla/train_lingbotvla.py', str(config)]
    runpy.run_path(sys.argv[0], run_name='__main__')
    assert (probe / 'precision_audit.json').is_file()
    (probe / 'summary.json').write_text(json.dumps(dict(state='three_step_probe_complete' if steps==3 else 'probe_complete',optimizer_steps=steps,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, micro=56, gas=1), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'cpu', 'smoke'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.source, a.out)
    elif a.mode == 'cpu':
        from experiment.official_control.prepare_adjust_bottle import cpu_check
        cpu_check(a.out, expected_steps=200, expected_fp32=False, expected_mixed=False, expected_export='bf16')
    else:
        smoke(a.out)

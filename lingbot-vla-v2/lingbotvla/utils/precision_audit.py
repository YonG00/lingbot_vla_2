"""Optional first-update dtype audit. Importing this file does not import torch."""
import json
from collections import Counter
from pathlib import Path


def tensor_inventory(tensors):
    counts = Counter()
    seen = set()
    for tensor in tensors:
        if tensor is None or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        counts[str(tensor.dtype)] += tensor.numel()
    return dict(counts)


def state_tensors(value):
    if hasattr(value, 'dtype') and hasattr(value, 'numel'):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from state_tensors(item)


class PrecisionAudit:
    def __init__(self, model, optimizer, path):
        self.optimizer = optimizer
        self.path = Path(path)
        self.done = False
        self.data = {'forward_samples': {}}
        self.handles = []
        groups = set()
        for name, module in model.named_modules():
            if type(module).__name__ != 'Linear':
                continue
            bucket = next((k for k in ('qwen_expert', 'visual', 'language_model', 'action_out') if k in name), None)
            if bucket is None or bucket in groups:
                continue
            groups.add(bucket)
            def hook(mod, inputs, output, key=name):
                self.data['forward_samples'][key] = dict(
                    weight=str(mod.weight.dtype),
                    inputs=[str(t.dtype) for t in state_tensors(inputs)],
                    outputs=[str(t.dtype) for t in state_tensors(output)])
            self.handles.append(module.register_forward_hook(hook))

    def parameters(self):
        return [p for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad]

    def before_step(self):
        if not self.done:
            self.data['gradient_numel_by_dtype'] = tensor_inventory(p.grad for p in self.parameters())

    def after_step(self):
        if self.done:
            return
        import torch
        self.data.update(parameter_numel_by_dtype=tensor_inventory(self.parameters()),
                         optimizer_state_numel_by_dtype=tensor_inventory(state_tensors(self.optimizer.state)),
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                         peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.data, indent=2) + '\n')
        temp.replace(self.path)
        for handle in self.handles:
            handle.remove()
        self.done = True
        params = self.data['parameter_numel_by_dtype']
        if not params or set(params) != {'torch.float32'}:
            raise RuntimeError(f'FP32 master audit failed: {params}')
        samples = self.data['forward_samples']
        if not any('qwen_expert' in n and 'torch.bfloat16' in r['outputs'] for n, r in samples.items()):
            raise RuntimeError('BF16 action-expert forward audit failed')
        print('[precision_audit] ' + json.dumps(self.data), flush=True)

"""Process-local FP32 deployment dtype audit; no extra model forwards."""
import json
import os
from pathlib import Path
import runpy


def instrument(policy, path):
    from lingbotvla.utils.precision_audit import PrecisionAudit, tensor_inventory
    params = tensor_inventory(policy.vla.parameters())
    if not params or set(params) != {'torch.float32'}:
        raise RuntimeError(f'FP32 evaluation weights audit failed: {params}')
    audit = PrecisionAudit(policy.vla, None, path)
    original = policy._infer_batch
    def infer(*args, **kwargs):
        result = original(*args, **kwargs)
        samples = audit.data['forward_samples']
        if not any('qwen_expert' in n and 'torch.float32' in row['outputs'] for n, row in samples.items()):
            raise RuntimeError('FP32 evaluation action-expert output audit failed')
        target = Path(path)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(parameter_numel_by_dtype=params, forward_samples=samples), indent=2)+'\n')
        temporary.replace(target)
        return result
    policy._infer_batch = infer
    return policy


def main():
    from deploy import websocket_policy_server as ws
    original = ws.WebsocketPolicyServer.__init__
    def init(self, policy, *args, **kwargs):
        instrument(policy, os.environ['LINGBOT_EVAL_PRECISION_AUDIT'])
        return original(self, policy, *args, **kwargs)
    ws.WebsocketPolicyServer.__init__ = init
    runpy.run_module('deploy.lingbot_vla_v2_policy', run_name='__main__')


if __name__ == '__main__':
    main()

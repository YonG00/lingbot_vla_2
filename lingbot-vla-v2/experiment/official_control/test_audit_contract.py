"""CPU contract: observe the actual functional output without changing gradients."""
import ast
from pathlib import Path
import types
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiment.official_control.gpu_smoke import instrument_fp32_action_output


class FunctionalAuditContract(unittest.TestCase):
    def test_original_functional_linear_observed_and_restored(self):
        path = Path(__file__).resolve().parents[2] / 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py'
        tree = ast.parse(path.read_text())
        flow = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlowMatching')
        node = next(n for n in flow.body if isinstance(n, ast.FunctionDef) and n.name == '_fp32_linear')
        node.decorator_list = []
        scope = {'F': F}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        original = scope['_fp32_linear']
        class Flow(nn.Module):
            _fp32_linear = staticmethod(original)
            def __init__(self):
                super().__init__()
                self.action_out_proj = nn.Linear(4, 3).float()
        model = Flow()
        x = torch.randn(2, 4, requires_grad=True)
        before = original(model.action_out_proj, x)
        expected = torch.autograd.grad(before.sum(), (x, model.action_out_proj.weight))
        hits = []
        handle = model.action_out_proj.register_forward_hook(lambda *args: hits.append(True))
        samples = {}
        restore = instrument_fp32_action_output(types.SimpleNamespace(model=model), samples)
        after = model._fp32_linear(model.action_out_proj, x)
        observed = torch.autograd.grad(after.sum(), (x, model.action_out_proj.weight))
        self.assertEqual(hits, [])  # prove why the old module hook missed it
        self.assertTrue(torch.equal(before, after))
        for a, b in zip(expected, observed):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(samples['model.action_out_proj.functional']['outputs'], ['torch.float32'])
        restore()
        self.assertNotIn('_fp32_linear', model.__dict__)
        self.assertIs(model._fp32_linear, original)
        handle.remove()


if __name__ == '__main__':
    unittest.main()

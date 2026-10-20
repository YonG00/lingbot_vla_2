"""CPU precision contracts; run with unittest, no model/checkpoint/CUDA loading."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('precision_cpu_contract', Path(__file__).resolve().parents[2] / 'lingbotvla/utils/precision_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class BF16ContextTests(unittest.TestCase):
    def test_all_storage_compute_and_fsdp_combinations(self):
        from contextlib import nullcontext
        for mixed in (False, True):
            for fp32 in (False, True):
                for fsdp in (False, True):
                    with self.subTest(mixed=mixed, fp32=fp32, fsdp=fsdp):
                        calls=[]
                        def autocast(**kw):
                            calls.append(kw)
                            return nullcontext()
                        with audit.training_precision_context(mixed,fp32,fsdp, SimpleNamespace(autocast=autocast,bfloat16='BF16')):
                            pass
                        self.assertEqual(calls, [] if fp32 or fsdp else [dict(device_type='cuda',dtype='BF16')])

    def test_actual_bf16_linear_with_fp32_input_backward(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        torch.set_num_threads(1)
        layer=torch.nn.Linear(4,3).to(dtype=torch.bfloat16)
        x=torch.randn(2,4,dtype=torch.float32,requires_grad=True)
        def autocast(device_type,dtype):
            self.assertEqual(device_type,'cuda')
            return torch.autocast('cpu',dtype=dtype)
        adapter=SimpleNamespace(autocast=autocast,bfloat16=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            layer(x)
        with audit.training_precision_context(False,False,False,adapter):
            y=layer(x)
            loss=y.float().square().mean()
        loss.backward()
        self.assertEqual(y.dtype,torch.bfloat16)
        self.assertEqual(layer.weight.dtype,torch.bfloat16)
        self.assertEqual(layer.weight.grad.dtype,torch.bfloat16)
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_action_output_observer_retains_output_and_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        from experiment.official_control.bf16_trial import observe_action_output
        layer=torch.nn.Linear(4,3).to(dtype=torch.bfloat16)
        x=torch.randn(2,4,dtype=torch.bfloat16,requires_grad=True)
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model=torch.nn.Module()
                self.model.action_out_proj=layer
        expected=layer(x)
        samples={}
        handle=observe_action_output(Policy(),samples,torch.compiler.disable)
        output=layer(x)
        self.assertTrue(torch.equal(output,expected))
        output.float().sum().backward()
        self.assertEqual(samples['model.action_out_proj']['outputs'],['torch.bfloat16'])
        self.assertTrue(torch.isfinite(x.grad).all())
        handle.remove()

if __name__=='__main__': unittest.main()

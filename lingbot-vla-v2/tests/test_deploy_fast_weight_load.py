"""CPU tests for the opt-in deployment startup path (no 6B model / GPU)."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from deploy.fast_weight_load import (
    fast_load_enabled,
    load_safetensors_streaming,
    model_init_on_device,
    check_cuda_load_budget,
    require_model_dtype,
)


def _toy():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 2)
            self.register_buffer('scale', torch.arange(2, dtype=torch.float32))
            self.register_buffer('nonpersistent', torch.tensor([123.]), persistent=False)

        def forward(self, x):
            return self.fc(x) + self.scale

    return Tiny()


def _write_shards(path, state):
    path.mkdir(exist_ok=True)
    names = sorted(state)
    save_file({key: state[key].detach().contiguous() for key in names[:2]}, str(path / 'model-00001.safetensors'))
    save_file({key: state[key].detach().contiguous() for key in names[2:]}, str(path / 'model-00002.safetensors'))


def test_streaming_exact_vs_standard_and_preserves_nonpersistent_buffer(tmp_path):
    torch.manual_seed(123)
    origin = _toy()
    weights = {k: v.detach().clone() for k, v in origin.state_dict().items()}
    _write_shards(tmp_path, weights)

    with model_init_on_device('cpu', torch.bfloat16):
        fast = _toy()
    legacy = _toy().to(torch.bfloat16)
    legacy.load_state_dict(weights, strict=True)

    result = load_safetensors_streaming(fast, str(tmp_path))
    assert result['tensors'] == len(weights)
    assert result['shards'] == 2
    for key, value in fast.state_dict().items():
        assert torch.equal(value, legacy.state_dict()[key]), key
    assert float(fast.nonpersistent.item()) == 123.
    x = torch.randn(4, 3).to(torch.bfloat16)
    assert torch.equal(fast(x), legacy(x))


def test_fail_closed_missing_key_before_any_copy(tmp_path):
    model = _toy()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    save_file({'fc.weight': torch.zeros_like(initial['fc.weight'])}, str(tmp_path / 'part.safetensors'))
    with pytest.raises(RuntimeError, match='missing='):
        load_safetensors_streaming(model, str(tmp_path))
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial[k])


def test_fail_closed_wrong_shape_before_any_copy(tmp_path):
    model = _toy()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    bad = {k: v.clone() for k, v in initial.items()}
    bad['fc.weight'] = torch.zeros(5, 3)
    _write_shards(tmp_path, bad)
    with pytest.raises(RuntimeError, match='shape_mismatch'):
        load_safetensors_streaming(model, str(tmp_path))
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial[k])


def test_fail_closed_unexpected_key(tmp_path):
    model = _toy()
    src = {k: v.clone() for k, v in model.state_dict().items()}
    src['not_a_real_weight'] = torch.zeros(1)
    _write_shards(tmp_path, src)
    with pytest.raises(RuntimeError, match='unexpected='):
        load_safetensors_streaming(model, str(tmp_path))


def test_fail_closed_duplicate_shard_keys(tmp_path):
    save_file({'dup': torch.ones(2)}, str(tmp_path / 'a.safetensors'))
    save_file({'dup': torch.ones(2)}, str(tmp_path / 'b.safetensors'))
    with pytest.raises(RuntimeError, match='Duplicate checkpoint tensor'):
        load_safetensors_streaming(_toy(), str(tmp_path))


def test_no_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_safetensors_streaming(_toy(), str(tmp_path))


def test_init_context_restores_dtype_even_when_constructor_throws():
    original = torch.get_default_dtype()
    with pytest.raises(ValueError):
        with model_init_on_device('cpu', torch.bfloat16):
            assert torch.get_default_dtype() == torch.bfloat16
            t = torch.nn.Linear(4, 3)
            assert t.weight.dtype == torch.bfloat16
            raise ValueError('probe')
    assert torch.get_default_dtype() == original


def test_flag_defaults_off_and_rejects_typo(monkeypatch):
    monkeypatch.delenv('LINGBOT_DEPLOY_FAST_LOAD', raising=False)
    assert fast_load_enabled() is False
    monkeypatch.setenv('LINGBOT_DEPLOY_FAST_LOAD', '1')
    assert fast_load_enabled() is True
    monkeypatch.setenv('LINGBOT_DEPLOY_FAST_LOAD', 'true')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        fast_load_enabled()


def _extract_server_method(name):
    # Test production method without importing 6B/Qwen/torchvision runtime.
    policy = Path(__file__).resolve().parents[1] / 'deploy' / 'lingbot_vla_v2_policy.py'
    tree = ast.parse(policy.read_text())
    server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LingbotVLAv2Server')
    method = next(node for node in server.body if isinstance(node, ast.FunctionDef) and node.name == name)
    code = compile(ast.Module(body=[method], type_ignores=[]), str(policy), 'exec')
    ns = {'torch': torch, 'time': __import__('time'), 'fast_load_enabled': fast_load_enabled}
    exec(code, ns)
    return ns[name]


def test_server_default_load_path_retains_legacy_cast_and_move(monkeypatch):
    monkeypatch.delenv('LINGBOT_DEPLOY_FAST_LOAD', raising=False)
    calls = []

    class FakeModel:
        def to(self, dtype):
            calls.append(('cast', dtype))
            return self
        def cuda(self):
            calls.append(('cuda',))
            return self
        def eval(self):
            calls.append(('eval',))
            return self

    model = FakeModel()
    fake = SimpleNamespace(use_bf16=True, load_vla=lambda path, **opts: (calls.append(('load', path, opts)) or model))
    output = _extract_server_method('_load_and_place_vla')(fake, '/x')
    assert output is model
    assert calls == [('load', '/x', {'fast_load': False}), ('cast', torch.bfloat16), ('cuda',), ('eval',)]


def test_server_fast_path_no_extra_cast_or_cuda(monkeypatch):
    monkeypatch.setenv('LINGBOT_DEPLOY_FAST_LOAD', '1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: 1024)
    calls = []

    class FakeModel:
        def eval(self):
            calls.append('eval')
            return self

    model = FakeModel()
    fake = SimpleNamespace(use_bf16=True, load_vla=lambda path, **opts: (calls.append(opts) or model))
    assert _extract_server_method('_load_and_place_vla')(fake, '/x') is model
    assert calls == [{'fast_load': True}, 'eval']


def test_server_fast_path_requires_cuda_and_bf16(monkeypatch):
    monkeypatch.setenv('LINGBOT_DEPLOY_FAST_LOAD', '1')
    m = _extract_server_method('_load_and_place_vla')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    with pytest.raises(RuntimeError, match='requires use_bf16'):
        m(SimpleNamespace(use_bf16=False), '/unused')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requires a CUDA GPU'):
        m(SimpleNamespace(use_bf16=True), '/unused')


def test_cuda_budget_rejects_low_free_memory_before_model_init(tmp_path, monkeypatch):
    origin = _toy()
    _write_shards(tmp_path, origin.state_dict())
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (5 * 1024**3, 48 * 1024**3))
    with pytest.raises(RuntimeError, match='Insufficient free CUDA memory'):
        check_cuda_load_budget(str(tmp_path))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (48 * 1024**3, 48 * 1024**3))
    check_cuda_load_budget(str(tmp_path))


def test_require_model_dtype_detects_mixed_dtypes():
    model = _toy().to(torch.bfloat16)
    require_model_dtype(model, torch.bfloat16, 'cpu')
    model.fc.bias.data = model.fc.bias.data.float()
    with pytest.raises(RuntimeError, match='expected torch.bfloat16'):
        require_model_dtype(model, torch.bfloat16, 'cpu')
    with pytest.raises(RuntimeError, match='expected cuda'):
        require_model_dtype(model, torch.bfloat16, 'cuda')


def test_deploy_probe_plan_never_opens_checkpoint_or_writes_files(tmp_path):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / 'tools' / 'deploy_load_probe.py'
    output = tmp_path / 'uncreated' / 'result.json'
    result = subprocess.run(
        [sys.executable, str(script), '--checkpoint', '/nonexistent/checkpoint',
         '--mode', 'fast', '--output', str(output)],
        check=True, capture_output=True, text=True,
    )
    assert 'PLAN ONLY' in result.stdout
    assert not output.exists()
    assert not output.parent.exists()

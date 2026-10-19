"""CPU-only precision plumbing/audit contracts; no torch import or GPU work."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest
from mouse_pad_training import test_recipe as fixtures

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = load('audit_cpu', 'lingbotvla/utils/precision_audit.py')
probe = load('probe_cpu', 'experiment/robotwin/precision_batch_probe.py')
server = load('server_cpu', 'experiment/robotwin/precision_eval_server.py')
entry = load('fp32_cpu', 'experiment/robotwin/recipes/20261019_place_mouse_pad_primary50_fp32_master.py')


def test_fp32_flags_survive_recipe_snapshot_and_worker_rebuild():
    f = fixtures.MouseRecipeTests()
    f.setUp()
    try:
        args = f.args
        args.fp32_master = True
        args.micro, args.gas = 32, 1
        args.train_all_episodes = True
        job, plan, snapshot = fixtures.recipe.build_job(args)
        assert plan['training_config']['train']['enable_mixed_precision'] is True
        assert plan['training_config']['train']['enable_fp32'] is False
        assert '--fp32-master' in job['command']
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(plan['training_config']))
        frozen = snapshot.with_name('frozen.al.json')
        frozen.write_text(json.dumps(plan['config']))
        command = job['command'][2:]
        command[command.index('--al-config')+1] = str(frozen)
        command[command.index('--execute')] = '--prepare-only'
        with contextlib.redirect_stdout(io.StringIO()):
            fixtures.recipe.launcher.main(command)
        rebuilt = json.loads((Path(job['output_dir'])/'trainer_config.json').read_text())
        assert rebuilt['train']['enable_mixed_precision'] is True
        assert rebuilt['train']['enable_fp32'] is False
        cmd = plan['command']
        assert cmd[cmd.index('--train.global_batch_size')+1] == '32'
        assert cmd[cmd.index('--train.hf_export_dtype')+1] == 'bf16'
    finally:
        f.doCleanups()


def test_probe_does_not_write_checkpoint_or_mutate_formal_command():
    original = ['python', '--train.dcp_save_mode', 'hf_only', '--train.smoke_no_checkpoint', 'false',
                '--train.disk_guard', 'true', '--train.save_steps', '250', '--train.hf_export_dtype', 'bf16']
    changed = probe.probe_command(original)
    assert original[original.index('--train.dcp_save_mode')+1] == 'hf_only'
    assert changed[changed.index('--train.dcp_save_mode')+1] == 'always'
    assert changed[changed.index('--train.smoke_no_checkpoint')+1] == 'true'
    assert changed[changed.index('--train.save_steps')+1] == '250'


def test_probe_prepares_static_cache_and_uses_same_frozen_al_config(monkeypatch):
    f = fixtures.MouseRecipeTests()
    f.setUp()
    try:
        args = f.args
        args.fp32_master, args.train_all_episodes = True, True
        job, plan, snapshot = fixtures.recipe.build_job(args)
        events = []
        def cache(cfg, dataset, manifest, tasks):
            assert cfg.target_change_cache_dir == plan['config']['target_change_cache_dir']
            assert dataset == args.dataset_root
            assert tasks == ['place_mouse_pad']
            assert not Path(job['output_dir']).exists()
            events.append('cache')
        def materialize(command):
            assert events == ['cache']
            frozen = Path(command[command.index('--al-config')+1])
            assert json.loads(frozen.read_text()) == plan['config']
            assert json.loads(snapshot.read_text())['train']['enable_mixed_precision'] is True
            assert '--prepare-only' in command and '--execute' not in command
            events.append('worker')
        monkeypatch.setattr(probe.recipe.launcher, 'main', materialize)
        probe.prepare_probe(args, job, plan, snapshot, prepare_cache=cache)
        assert events == ['cache', 'worker']
    finally:
        f.doCleanups()


def test_new_recipe_defaults_to_requested_32_and_fp32_evaluation(monkeypatch):
    captured = []
    monkeypatch.setattr(entry.pipeline, 'main', lambda args: captured.extend(args))
    entry.main([])
    assert '--execute' not in captured and '--shutdown' not in captured
    assert captured[captured.index('--micro')+1] == '32'
    assert captured[captured.index('--gas')+1] == '1'
    assert captured[captured.index('--eval-precision')+1] == 'fp32'
    assert '--fp32-master' in captured


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_deployment_flags_match_selected_precision(precision):
    command = entry.pipeline.evaluation_server_command(Path('/checkpoint'), 12345, precision)
    assert command[command.index('--use_fp32')+1] == str(precision == 'fp32')
    assert command[command.index('--use_bf16')+1] == str(precision == 'bf16')
    if precision == 'fp32':
        assert any(x.endswith('precision_eval_server.py') for x in command)


class Tensor:
    def __init__(self, dtype, count=2): self.dtype, self.count = dtype, count
    def numel(self): return self.count


def test_fp32_eval_audit_does_not_add_forward_or_change_return_value(tmp_path):
    callbacks = []
    class Linear:
        weight = Tensor('torch.float32')
        def register_forward_hook(self, callback):
            callbacks.append(callback)
            return NS(remove=lambda: None)
    module = Linear()
    model = NS(parameters=lambda: [module.weight], named_modules=lambda: [('qwen_expert.linear', module)])
    calls = []
    returned = object()
    def infer(*args, **kwargs):
        calls.append(args)
        for callback in callbacks:
            callback(module, (Tensor('torch.float32'),), Tensor('torch.float32'))
        return returned
    policy = NS(vla=model, _infer_batch=infer)
    target = tmp_path/'audit.json'
    server.instrument(policy, target)
    assert policy._infer_batch('observation') is returned
    assert len(calls) == 1
    assert json.loads(target.read_text())['parameter_numel_by_dtype'] == {'torch.float32': 2}


def test_dtype_inventory_deduplicates_shared_parameters_and_nested_states():
    tensor = Tensor('torch.float32', 20)
    assert audit.tensor_inventory([tensor, tensor, None]) == {'torch.float32': 20}
    assert audit.tensor_inventory(audit.state_tensors({'a': [tensor, Tensor('torch.int64', 1)]})) == {'torch.float32': 20, 'torch.int64': 1}


@pytest.mark.parametrize('mixed,fp32,fsdp,active', [(True,False,False,True),
    (False,False,False,False), (True,True,False,False), (True,False,True,False)])
def test_only_single_card_mixed_training_needs_explicit_autocast(mixed, fp32, fsdp, active):
    from contextlib import nullcontext
    calls = []
    def autocast(**kwargs):
        calls.append(kwargs)
        return nullcontext()
    with audit.training_precision_context(mixed, fp32, fsdp,
            torch_module=NS(autocast=autocast, bfloat16='bf16')):
        pass
    assert calls == ([{'device_type':'cuda','dtype':'bf16'}] if active else [])


def test_cpu_operator_autocast_retains_fp32_optimizer_storage():
    torch = pytest.importorskip('torch')
    linear = torch.nn.Linear(4, 3).float()
    optimizer = torch.optim.SGD(linear.parameters(), lr=.01, momentum=.9)
    def autocast(device_type, dtype):
        assert device_type == 'cuda'
        return torch.autocast('cpu', dtype=dtype)
    adapter = NS(autocast=autocast, bfloat16=torch.bfloat16)
    with audit.training_precision_context(True, False, False, torch_module=adapter):
        output = linear(torch.ones(2, 4))
        assert output.dtype == torch.bfloat16
        loss = output.float().square().mean()
    loss.backward()
    assert all(p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in linear.parameters())
    optimizer.step()
    assert all(v['momentum_buffer'].dtype == torch.float32 for v in optimizer.state.values())


@pytest.mark.parametrize('dtype,raises', [('torch.float32', False), ('torch.bfloat16', True)])
def test_after_update_audits_real_optimizer_storage_and_preserves_parameters(tmp_path, monkeypatch, dtype, raises):
    tensor = Tensor(dtype, 20)
    tensor.requires_grad = True
    tensor.grad = Tensor('torch.float32', 20)
    optimizer = NS(param_groups=[{'params': [tensor]}], state={'p': {'momentum': Tensor(dtype, 20)}})
    obj = audit.PrecisionAudit(NS(named_modules=lambda: []), optimizer, tmp_path/'audit.json')
    obj.data['forward_samples'] = {'qwen_expert.test': {'weight': 'torch.bfloat16', 'outputs': ['torch.bfloat16']}}
    monkeypatch.setitem(sys.modules, 'torch', NS(cuda=NS(max_memory_allocated=lambda: 10, max_memory_reserved=lambda: 20)))
    obj.before_step()
    if raises:
        with pytest.raises(RuntimeError, match='master audit'):
            obj.after_step()
    else:
        obj.after_step()
        assert obj.done and tensor.dtype == dtype
        data = json.loads((tmp_path/'audit.json').read_text())
        assert data['parameter_numel_by_dtype'] == {'torch.float32': 20}
        assert data['gradient_numel_by_dtype'] == {'torch.float32': 20}
        assert data['optimizer_state_numel_by_dtype'] == {'torch.float32': 20}

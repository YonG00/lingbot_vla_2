"""CPU-only regression of the rebuilt shared inference core from production AST.

Do not infer real 6B BF16 batch parity from these isolated model tests.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).parents[1] / 'lingbotvla/utils/open_loop_validation.py'


def _production_class(cache_clear=lambda _m: None, cache_restore=lambda _m, _s: None):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'OpenLoopValidator')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                ('_infer_one', '_infer_batch', '_infer_core', '_noise_generator')]
    ast.fix_missing_locations(cls)
    namespace = {
        'torch': torch, 'np': np, 'os': __import__('os'), 'Dict': dict, 'Any': object,
        'List': list, 'Sequence': list, 'EVAL_SEED': 1234,
        '_visual_grid_cache_clear': cache_clear,
        '_visual_grid_cache_restore': cache_restore,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['OpenLoopValidator']


class RecordingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.calls = []

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       *, noise, image_grid_thw=None):
        self.calls.append({
            'images': images.detach().clone(),
            'img_masks': img_masks.detach().clone(),
            'lang_tokens': lang_tokens.detach().clone(),
            'lang_masks': lang_masks.detach().clone(),
            'state': state.detach().clone(),
            'noise': noise.detach().clone(),
            'grid': image_grid_thw.detach().clone() if image_grid_thw is not None else None,
        })
        # Per-sample independent policy; comparison is exact by construction.
        return noise + state[:, :1].reshape(-1, 1, 1) * 0.5


class Transform:
    def unapply(self, item):
        return {'actions': item['actions']}


def _item(value, *, grid=True):
    item = {
        'images': torch.full((2, 3, 4, 4), value),
        'img_masks': torch.ones(2),
        'lang_tokens': torch.tensor([13, 27, 9]),
        'lang_masks': torch.ones(3),
        'state': torch.tensor([value, value + 1]),
        'actions': torch.zeros(2, 3),
    }
    if grid:
        item['image_grid_thw'] = torch.tensor([[1, 4, 4], [1, 4, 4]])
    return item


def _validator(cls):
    v = object.__new__(cls)
    v.model = RecordingPolicy()
    v.device = 'cpu'
    v._model_config = SimpleNamespace(n_action_steps=2, max_action_dim=3, action_fp32=False)
    v.args = SimpleNamespace(train=SimpleNamespace(eval_inference_dtype='auto', use_bf16=False))
    v._noise_gen = None
    v._precision_logged = True
    v.dump_dir = None
    v._dump_prefix = None
    return v


def test_single_reference_input_and_rng_unchanged():
    cls = _production_class()
    v = _validator(cls)
    it = _item(1.5)
    expected_noise = torch.randn((1, 2, 3), generator=torch.Generator().manual_seed(1234))
    result = v._infer_one(it, Transform())
    call = v.model.calls[0]
    assert len(v.model.calls) == 1
    assert torch.equal(call['noise'], expected_noise)
    assert call['images'].shape == (1, 2, 3, 4, 4)
    assert call['img_masks'].shape == (1, 2)
    assert call['lang_tokens'].shape == (1, 3)
    assert call['lang_masks'].shape == (1, 3)
    assert call['state'].shape == (1, 2)
    assert call['grid'].shape == (2, 3)  # historical B1 shape (NOT stacked)
    assert torch.equal(result['actions'], (expected_noise + 0.75)[0])
    assert torch.equal(it['state'], torch.tensor([1.5, 2.5]))


def test_batch_is_one_forward_and_each_prepared_input_equals_serial():
    cls = _production_class()
    items = [_item(x) for x in (1., 2., 3.)]
    serial = _validator(cls)
    expected = [serial._infer_one(it, Transform()) for it in items]
    single_calls = list(serial.model.calls)
    batched = _validator(cls)
    outputs = batched._infer_batch(items, Transform())
    assert len(batched.model.calls) == 1
    batch_call = batched.model.calls[0]
    for key in ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'state', 'noise'):
        assert torch.equal(batch_call[key], torch.cat([c[key] for c in single_calls], dim=0)), key
    assert batch_call['grid'].shape == (3, 2, 3)
    for i, call in enumerate(single_calls):
        assert torch.equal(batch_call['grid'][i], call['grid'])
        assert torch.equal(outputs[i]['actions'], expected[i]['actions'])


def test_cache_scope_keeps_original_serial_and_restores_batch_on_error():
    record = []
    state = {'cached': 'training'}

    def clear(model):
        record.append('clear')
        old = state['cached']
        state['cached'] = None
        return old

    def restore(model, old):
        record.append('restore')
        state['cached'] = old

    cls = _production_class(clear, restore)
    v = _validator(cls)
    v._infer_one(_item(1), Transform())
    assert record == []  # original serial path did not clear here
    assert state['cached'] == 'training'

    def fail(*args, **kwargs):
        assert state['cached'] is None
        raise RuntimeError('injected model error')

    v.model.sample_actions = fail
    with pytest.raises(RuntimeError, match='injected'):
        v._infer_batch([_item(1), _item(2)], Transform())
    assert record == ['clear', 'restore']
    assert state['cached'] == 'training'


def test_unsafe_shape_rejected_before_model_or_rng_consumption():
    cls = _production_class()
    v = _validator(cls)
    one = _item(1)
    two = _item(2)
    two['state'] = torch.tensor([[2., 3.]])
    with pytest.raises(ValueError, match='homogeneous'):
        v._infer_batch([one, two], Transform())
    assert not v.model.calls
    assert v._noise_gen is None

    two = _item(2)
    two['image_grid_thw'] = None
    with pytest.raises(ValueError, match='homogeneous'):
        v._infer_batch([one, two], Transform())
    assert not v.model.calls


def test_bad_batch_grid_rejected_before_model_call():
    cls = _production_class()
    v = _validator(cls)
    one = _item(1)
    two = _item(2)
    one['image_grid_thw'] = torch.tensor([1, 4, 4])
    two['image_grid_thw'] = torch.tensor([1, 4, 4])
    with pytest.raises(ValueError, match='shape \\(N,3\\)'):
        v._infer_batch([one, two], Transform())
    assert not v.model.calls


def test_single_model_returning_2d_still_works():
    cls = _production_class()
    v = _validator(cls)
    def two_dim(*args, **kwargs):
        return torch.ones((2, 3))
    v.model.sample_actions = two_dim
    result = v._infer_one(_item(1), Transform())
    assert result['actions'].shape == (2, 3)
    assert torch.equal(result['actions'], torch.ones(2, 3))


def test_no_grid_is_valid_and_batch_grid_all_or_none():
    cls = _production_class()
    v = _validator(cls)
    items = [_item(1, grid=False), _item(2, grid=False)]
    v._infer_batch(items, Transform())
    assert v.model.calls[0]['grid'] is None

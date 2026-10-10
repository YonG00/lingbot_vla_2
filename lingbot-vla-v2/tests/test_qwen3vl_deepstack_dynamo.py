"""`_deepstack_process` 的 ROCm 绕行契约（2026-10-10）。

背景
----
上游 `transformers/models/qwen3_vl/modeling_qwen3_vl.py` 的 `_deepstack_process` 用
**布尔掩码索引写**：
```
local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
hidden_states[visual_pos_masks, :] = local_this
```
真机（ROCm 7.2.1 / gfx1100）实测：开 `torch.compile` 时这会让 Inductor 生成融合 kernel
`triton_tem_fused_slice_backward_transpose_view_zeros_2`，而 Triton AMD 后端的
`TritonAMDGPUOptimizeDotOperands` pass 对它确定性失败（`PassManager::run failed`，
7/7 rank）；同处 `aten.nonzero.default` 也无法编译。

因此我们在**仓库子类**里覆写该方法并加 `@torch.compiler.disable`，把这段移出编译图。

本测试守住三件事（防止后人误删/上游改名导致静默失效）：
  1. 子类**确实**在自己的 `__dict__` 里覆写了 `_deepstack_process`；
  2. 该函数带 PyTorch 的 dynamo-disable 标记（属性名跨 2.8/2.9 一致）；
  3. 语义与上游**逐位一致**（含全 False 掩码等边界）。
"""

from __future__ import annotations

import torch
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

DYNAMO_DISABLE_MARKERS = ("_torchdynamo_disable", "_torchdynamo_disable_msg")


def _cls():
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration


def test_subclass_overrides_deepstack_process():
    """必须是我们子类自己的实现（否则补丁没生效、崩溃会复现）。"""
    cls = _cls()
    assert "_deepstack_process" in cls.__dict__, (
        'Qwen3VLForConditionalGeneration 未覆写 _deepstack_process ⇒ ROCm 绕行补丁失效')


def test_override_is_dynamo_disabled():
    """必须带 dynamo 禁用标记；否则该段仍会进编译图、重新触发 Triton 缺陷。"""
    fn = _cls().__dict__["_deepstack_process"]
    present = [m for m in DYNAMO_DISABLE_MARKERS if hasattr(fn, m)]
    assert present, (
        f'_deepstack_process 缺少 dynamo-disable 标记（应为 {DYNAMO_DISABLE_MARKERS} 之一）'
        f'；现有以 _torch 开头的属性：{[a for a in dir(fn) if a.startswith("_torch")]}')
    assert getattr(fn, "_torchdynamo_disable", False) is True or hasattr(fn, "_torchdynamo_disable_msg")


class _Stub:
    """只为拿上游实现来对拍（不构造真模型）。"""

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        raise AssertionError('应由子类覆写覆盖')


@pytest.mark.parametrize('n_true', [0, 1, 3, 'all'])
def test_semantics_match_upstream_bitwise(n_true, monkeypatch):
    """我们的覆写与上游写法**逐位一致**（含全 False / 全 True 边界）。"""
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLForConditionalGeneration as C
    import transformers.models.qwen3_vl.modeling_qwen3_vl as hf

    torch.manual_seed(0)
    seq, dim, n_vis = 7, 4, 3
    base = torch.randn(seq, dim, dtype=torch.float32)
    vis = torch.randn(n_vis, dim, dtype=torch.float32) * 0.5
    if n_true == 0:
        mask = torch.zeros(seq, dtype=torch.bool)
    elif n_true == 'all':
        mask = torch.ones(seq, dtype=torch.bool)
    else:
        mask = torch.zeros(seq, dtype=torch.bool)
        mask[:n_true] = True
        if n_true > n_vis:
            pytest.skip('掩码为 True 的数量不应超过 visual_embeds 行数')

    # 上游实现（直接取库函数，绕过我们的覆写）
    upstream = hf.Qwen3VLForConditionalGeneration._deepstack_process
    ref = upstream(_Stub(), base.clone(), mask.clone(), vis.clone())
    got = C._deepstack_process(_Stub(), base.clone(), mask.clone(), vis.clone())

    assert torch.equal(ref, got), '覆写结果与上游不一致（必须逐位相同）'
    assert torch.equal(got[~mask], base[~mask]), '未命中掩码的位置不应被改动'


def test_no_op_when_not_compiled():
    """不开编译时装饰器应为空操作：连续调用两次结果一致。"""
    C = _cls()
    torch.manual_seed(1)
    base = torch.randn(5, 3)
    vis = torch.randn(2, 3)
    mask = torch.tensor([True, False, True, False, False])
    a = C._deepstack_process(_Stub(), base.clone(), mask, vis)
    b = C._deepstack_process(_Stub(), base.clone(), mask, vis)
    assert torch.equal(a, b)

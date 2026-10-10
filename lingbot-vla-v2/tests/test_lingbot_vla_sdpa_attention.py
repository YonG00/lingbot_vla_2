"""`sdpa_attention_forward` 契约与数值对拍（2026-10-10）。

背景：VLM 主干原本走 flex attention（`attention_implementation: flex_cached`），其**反向**
kernel 在 gfx1100 上令 Triton AMD 后端确定性失败（`PassManager::run failed`）。改用语义等价的
SDPA 实现（`sdpa_attention_forward`）。本测试守住：

  1. 输出契约与 `flex_attention_forward` 一致 —— `[B, L, H*D]`、dtype 跟随输入；
  2. **与 flex 数值一致**（前向），含 GQA（Hq≠Hkv）与含"全屏蔽行"的稠密 bool mask；
  3. 三种非法输入要**明确报错**（mask 为 None / 非 bool / 头数不整除），不静默算错。

⚠️ 这里**不能**用 `enable_gqa=True` 而省略 KV 展开：head 映射会变、误差≈输出量级（文档实测）。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lingbotvla.models.vla.lingbot_vla.flex_attention import (
    flex_attention_forward,
    sdpa_attention_forward,
)

ATOL = 5e-5


def _make_inputs(b=1, q=12, kv=12, hq=4, hkv=2, d=32, seed=0, dtype=torch.float32,
                 full_block_row=False):
    g = torch.Generator().manual_seed(seed)
    qs = torch.randn(b, q, hq, d, generator=g, dtype=dtype)
    ks = torch.randn(b, kv, hkv, d, generator=g, dtype=dtype)
    vs = torch.randn(b, kv, hkv, d, generator=g, dtype=dtype)
    # 因果 mask（True=允许）；可选在开头造一行"全屏蔽"
    mask = torch.tril(torch.ones(b, q, kv, dtype=torch.bool))
    if full_block_row:
        mask[:, 0, :] = False
    return qs, ks, vs, mask


def test_output_contract_matches_flex():
    """形状/dtype 必须与 flex 路径一致：[B, L, H*D]，dtype 跟随 Q。"""
    qs, ks, vs, mask = _make_inputs()
    out = sdpa_attention_forward(qs, ks, vs, mask)
    assert out.shape == (qs.shape[0], qs.shape[1], qs.shape[2] * qs.shape[3])
    assert out.dtype == qs.dtype

    qs16 = qs.to(torch.bfloat16)
    out16 = sdpa_attention_forward(qs16, ks.to(torch.bfloat16), vs.to(torch.bfloat16), mask)
    assert out16.dtype == torch.bfloat16


@pytest.mark.parametrize('full_block_row', [False, True])
@pytest.mark.parametrize('hq,hkv', [(4, 2), (4, 4), (4, 1)])
def test_numeric_parity_with_flex(hq, hkv, full_block_row):
    """与 flex 前向逐元素对比（GQA、以及含全屏蔽行的边界）。"""
    qs, ks, vs, mask = _make_inputs(hq=hq, hkv=hkv, full_block_row=full_block_row)
    ref = flex_attention_forward(qs.clone(), ks.clone(), vs.clone(), mask.clone())
    got = sdpa_attention_forward(qs.clone(), ks.clone(), vs.clone(), mask.clone())
    worst = (ref - got).abs().max().item()
    assert worst <= ATOL, f'max|Δ|={worst:.3e} > {ATOL:g}（hq={hq} hkv={hkv} 全屏蔽行={full_block_row}）'


def test_accepts_4d_mask_with_singleton_head_dim():
    """也接受 `[B,1,Q,KV]`（BlockMask 路径会给出这种形态）。"""
    qs, ks, vs, mask = _make_inputs()
    a = sdpa_attention_forward(qs, ks, vs, mask)
    b = sdpa_attention_forward(qs, ks, vs, mask[:, None])
    assert torch.allclose(a, b, atol=1e-6)


def test_full_block_row_outputs_zero():
    """全屏蔽行：softmax 全 -inf ⇒ 输出应为 0（而不是 NaN）。"""
    qs, ks, vs, mask = _make_inputs(full_block_row=True)
    out = sdpa_attention_forward(qs, ks, vs, mask)
    assert torch.isfinite(out).all(), '出现 NaN/Inf'
    assert out[:, 0, :].abs().max().item() < 1e-6


def test_none_mask_raises():
    qs, ks, vs, _ = _make_inputs()
    with pytest.raises(ValueError, match='attention_mask 为 None'):
        sdpa_attention_forward(qs, ks, vs, None)


def test_non_bool_mask_raises():
    qs, ks, vs, mask = _make_inputs()
    with pytest.raises(TypeError, match='bool'):
        sdpa_attention_forward(qs, ks, vs, mask.float())


def test_non_divisible_heads_raises():
    qs, ks, vs, mask = _make_inputs(hq=5, hkv=2)
    with pytest.raises(ValueError, match='不整除'):
        sdpa_attention_forward(qs, ks, vs, mask)

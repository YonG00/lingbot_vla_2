"""静态契约：SDPA 注意力路径必须存在且接线正确（任何环境可跑，无需 import 重依赖）。

本机 transformers 版本过旧（缺 `AutoModelForVision2Seq`）⇒ 依赖真模型的数值对拍
（`tests/test_lingbot_vla_sdpa_attention.py`）只能在训练机跑。这条用 AST 守住结构，
防止后人误删/漏接线导致 flex 反向 kernel 崩溃复现（gfx1100 上开 torch.compile 必崩）。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'lingbotvla/models/vla/lingbot_vla'
FLEX_SRC = ROOT / 'flex_attention.py'
MODEL_SRC = ROOT / 'modeling_lingbot_vla_v2.py'


def _trees():
    return (ast.parse(FLEX_SRC.read_text(encoding='utf-8'), filename=str(FLEX_SRC)),
            ast.parse(MODEL_SRC.read_text(encoding='utf-8'), filename=str(MODEL_SRC)))


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_sdpa_attention_forward_exists_with_same_signature():
    tree, _ = _trees()
    flex = _func(tree, 'flex_attention_forward')
    sdpa = _func(tree, 'sdpa_attention_forward')
    assert flex is not None, '缺少 flex_attention_forward（基准实现）'
    assert sdpa is not None, '缺少 sdpa_attention_forward ⇒ 崩溃绕行失效'
    a = [x.arg for x in flex.args.args]
    b = [x.arg for x in sdpa.args.args]
    assert a == b, f'签名必须一致（可互换注入）：flex={a} sdpa={b}'


def test_sdpa_uses_repeat_interleave_and_disables_gqa():
    """必须用 repeat_interleave 展开 KV 头并 enable_gqa=False（文档实测：否则误差≈输出量级）。"""
    tree, _ = _trees()
    src = ast.unparse(_func(tree, 'sdpa_attention_forward'))
    assert 'repeat_interleave' in src, 'GQA 未按仓库同序展开（应 repeat_interleave(g, dim=2)）'
    assert 'enable_gqa=False' in src, '必须 enable_gqa=False（KV 已展开）'
    assert 'scaled_dot_product_attention' in src
    assert 'is_causal=False' in src, '可见性应全部由 mask 编码'


def test_model_wires_sdpa_branch():
    _, tree = _trees()
    src = ast.unparse(tree)
    assert 'sdpa_attention_forward' in src, 'modeling_lingbot_vla_v2 未 import/使用 sdpa 实现'
    assert "'sdpa'" in src or '"sdpa"' in src, 'get_attention_interface 未加 "sdpa" 分支'


def test_config_uses_sdpa_not_flex_cached():
    cfg = (Path(__file__).resolve().parents[1]
           / 'configs/rocm/robotwin_official_paths_rocm.yaml').read_text(encoding='utf-8')
    lines = [l for l in cfg.splitlines() if l.strip().startswith('attention_implementation:')]
    assert lines, '配置里找不到 attention_implementation'
    val = lines[-1].split(':', 1)[1].strip()
    assert val == 'sdpa', f'ROCm 配置应为 sdpa（绕开 flex 反向崩溃），实际 {val!r}'

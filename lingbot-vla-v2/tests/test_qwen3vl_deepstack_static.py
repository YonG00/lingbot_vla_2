"""静态契约（无需 import transformers）：`_deepstack_process` 覆写必须存在且带禁用装饰器。

为什么单独写静态版：本机 transformers 版本过旧（`AutoModelForVision2Seq` 缺失）⇒ 依赖
真模型 import 的功能测试只能在训练机跑（见 `tests/test_qwen3vl_deepstack_dynamo.py`）。
这条用 AST 解析源码，**任何环境**都能守住"补丁被误删/装饰器被去掉"。
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1]
       / 'lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py')
TARGET_CLASS = 'Qwen3VLForConditionalGeneration'
TARGET_METHOD = '_deepstack_process'


def _find_method():
    tree = ast.parse(SRC.read_text(encoding='utf-8'), filename=str(SRC))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == TARGET_CLASS:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == TARGET_METHOD:
                    return node, item
    return None, None


def test_override_exists_in_repo_subclass():
    cls, fn = _find_method()
    assert cls is not None, f'找不到类 {TARGET_CLASS}'
    assert fn is not None, (
        f'{TARGET_CLASS} 未覆写 {TARGET_METHOD} ⇒ ROCm 绕行补丁失效，'
        '开 torch.compile 会重新触发 Triton AMD pass 崩溃（见 2026-10-10 记忆 §27）')


def test_override_is_decorated_with_compiler_disable():
    _, fn = _find_method()
    assert fn is not None
    deco = [ast.unparse(d) for d in fn.decorator_list]
    ok = any('compiler.disable' in d or '_dynamo.disable' in d or 'dynamo.disable' in d for d in deco)
    assert ok, f'{TARGET_METHOD} 缺少 dynamo-disable 装饰器；现有装饰器：{deco}'


def test_override_delegates_to_super():
    """必须是委托给上游实现（而不是复制一份逻辑），否则上游修复无法继承。"""
    _, fn = _find_method()
    assert fn is not None
    body = ast.unparse(fn)
    assert 'super()._deepstack_process' in body, (
        '覆写体应委托 super()._deepstack_process（保持与上游同步）；'
        f'实际：{body[:200]}')

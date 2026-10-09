"""`compile_tuning` 的 CPU 测试（② 评测禁编译 + ③ dynamo 重编译上限）。

两个开关都必须**默认安全**：默认开启调优；`=0` 能干净回退到改动前行为。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.utils import compile_tuning as ct  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ct.CACHE_LIMIT_ENV, raising=False)
    monkeypatch.delenv(ct.EVAL_DISABLE_ENV, raising=False)


# ---------------------------------------------------------------------------
# ③ cache_size_limit
# ---------------------------------------------------------------------------
def test_default_raises_cache_limit_and_is_idempotent():
    from torch._dynamo import config as dcfg
    before = dcfg.cache_size_limit
    try:
        info = ct.apply_dynamo_tuning()
        assert info["after"] == ct.DEFAULT_CACHE_LIMIT == 64
        assert info["before"] == before
        assert dcfg.cache_size_limit == 64
        again = ct.apply_dynamo_tuning()          # 幂等
        assert again["changed"] is False and again["after"] == 64
    finally:
        dcfg.cache_size_limit = before


def test_cache_limit_env_override_and_skip():
    from torch._dynamo import config as dcfg
    before = dcfg.cache_size_limit
    try:
        import os
        os.environ[ct.CACHE_LIMIT_ENV] = "128"
        assert ct.apply_dynamo_tuning()["after"] == 128
        os.environ[ct.CACHE_LIMIT_ENV] = "0"      # 0 = 跳过（保持原值）
        info = ct.apply_dynamo_tuning()
        assert info["skipped"] is True
        assert dcfg.cache_size_limit == 128       # 未被改动
        os.environ[ct.CACHE_LIMIT_ENV] = "-3"
        with pytest.raises(ValueError):
            ct.apply_dynamo_tuning()
        os.environ[ct.CACHE_LIMIT_ENV] = "abc"
        with pytest.raises(ValueError):
            ct.apply_dynamo_tuning()
    finally:
        dcfg.cache_size_limit = before


# ---------------------------------------------------------------------------
# ② 评测禁编译
# ---------------------------------------------------------------------------
def test_eval_compile_disabled_default_on_and_env_off():
    import os
    with ct.eval_compile_disabled() as active:
        assert active is True                     # 默认启用
    os.environ[ct.EVAL_DISABLE_ENV] = "0"
    with ct.eval_compile_disabled() as active:
        assert active is False                    # 可干净回退
    os.environ[ct.EVAL_DISABLE_ENV] = "1"
    with ct.eval_compile_disabled() as active:
        assert active is True


def test_eval_compile_disabled_actually_stops_tracing():
    """区间内 dynamo **不**产生编译图（这正是消除 guard 扰动的手段）。"""
    import torch
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        return x * 2 + 1

    compiled = torch.compile(fn, fullgraph=False)
    with ct.eval_compile_disabled():
        out = compiled(torch.ones(4))
    assert torch.allclose(out, torch.full((4,), 3.0))
    assert calls["n"] >= 1
    # dynamo 在该区间内没有新编译（frame 计数不增加）
    try:
        from torch._dynamo.utils import counters
        before = dict(counters.get("frames_total", {}))
        with ct.eval_compile_disabled():
            compiled(torch.ones(4))
        after = dict(counters.get("frames_total", {}))
        assert before == after, f"区间内仍在编译: {before} → {after}"
    except ImportError:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# 接线检查（AST，不改生产语义）
# ---------------------------------------------------------------------------
def test_eval_context_and_trainer_are_wired():
    src = (REPO / "lingbotvla/utils/open_loop_validation.py").read_text(encoding="utf-8")
    assert "eval_compile_disabled" in src
    assert "_compile_ctx.__enter__()" in src and "_compile_ctx.__exit__(None, None, None)" in src
    # 退出必须发生在 finally 里（异常路径也要关掉编译区）
    idx_finally = src.index("_compile_ctx.__exit__(None, None, None)")
    assert src.rindex("finally:", 0, idx_finally) > src.index("_compile_ctx.__enter__()")

    tr = (REPO / "tasks/vla/train_lingbotvla.py").read_text(encoding="utf-8")
    assert "apply_dynamo_tuning" in tr
    assert tr.index("apply_dynamo_tuning") < tr.index("model = torch.compile(model)")

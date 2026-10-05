#!/usr/bin/env python3
"""resume 守卫回归测试 —— **纯 CPU、秒级、无卡也能跑**。

背景（2026-10-05 实测踩坑）
--------------------------
`single_task_train.sh` 是「**先启动剪枝看门狗、再跑训练**」。看门狗的设计前提是
「不续训、只要评测」（见 `prune_dcp.py` docstring），所以默认 `PRUNE_KEEP=0` ——
把每份存档的 DCP 全删。但 `RESUME=1` 恰恰要读**最新那份**的 DCP ⇒ 冲突：
训练去 load 时 `model/.metadata` 已被删 ⇒ `FileNotFoundError`。

本测试验三件事
--------------
T1 脚本守卫：`RESUME=1 PRUNE=1 PRUNE_KEEP=0` ⇒ 脚本强制 `keep-last=1` 并告警
T2a 看门狗行为：`--keep-last 1` ⇒ **最新那份的 DCP 保留**
T2b 看门狗行为：`--keep-last 0` ⇒ **最新那份的 DCP 被删**（证明 T2a 不是空转）

为什么不用真 ckpt / 不需要 GPU
-----------------------------
* 被测的是**文件生命周期**（谁被删、谁被留），不是模型逻辑 ⇒ 造假目录即可
* `prune_dcp.py` 顶层**不 import torch**；慢的只有 `main()` 里的
  `_load_check_hf_ckpt()`（才去 import 校验模块，无卡时会挂住）
  ⇒ 这里**绕开 main()**，直接 import 模块调 `sweep()`，并把 `check_hf_ckpt`
  换成「永远判定完整」的桩函数 ⇒ 秒级、无卡可跑

用法
----
    /data/miniconda3/envs/lingbotvla/bin/python -u tools/resume_guard_test.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRUNE = REPO / "tools" / "prune_dcp.py"
TRAIN_SH = REPO / "experiment" / "robotwin" / "single_task_train.sh"

results: list[tuple[str, bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)


def load_prune_module():
    """只 import 模块本身（顶层无重依赖），不跑 main()。"""
    spec = importlib.util.spec_from_file_location("_prune_dcp_under_test", PRUNE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def always_complete(hf, min_age_seconds=None, now=None, **kw):
    """桩：跳过「完整性判据」（那需要重依赖），只测剪/不剪的生命周期。"""
    return True, [], [], 0, 0


def mk_tree(root: Path) -> None:
    """造两档假存档：global_step_100（旧）、global_step_200（新）。"""
    for s in (100, 200):
        d = root / "checkpoints" / f"global_step_{s}"
        for sub in ("hf_ckpt", "model", "optimizer", "extra_state"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        (d / "hf_ckpt" / "model.safetensors.index.json").write_text("{}")
        (d / "model" / ".metadata").write_text("m")
        (d / "optimizer" / ".metadata").write_text("o")
        (d / "extra_state" / "extra_state_0.pt").write_text("e")


def main() -> int:
    print(f"[test] prune_dcp = {PRUNE}")
    if not PRUNE.is_file():
        print(f"❌ 找不到 {PRUNE}", file=sys.stderr)
        return 2

    # ---------------- T1 脚本守卫 ----------------
    print("\n== T1 脚本守卫（dry-run，不训练、不碰 GPU）==")
    env_cmd = (
        f"cd {REPO} && DRY_RUN=1 RESUME=1 PRUNE=1 PRUNE_KEEP=0 TASK=click_bell "
        f"TRAIN_OUT=/data/tmp/_unused bash {TRAIN_SH} 2>&1"
    )
    out = subprocess.run(["bash", "-lc", env_cmd], capture_output=True, text=True).stdout
    chk("打印了「强制 PRUNE_KEEP=1」告警", "强制 PRUNE_KEEP=1" in out)
    chk("计划里 keep-last=1", "keep-last=1" in out)

    # ---------------- T2 看门狗行为 ----------------
    mod = load_prune_module()
    print("\n== T2a 看门狗 keep_last=1：最新那份 DCP 必须保留 ==")
    with tempfile.TemporaryDirectory(prefix="resume_guard_") as t:
        root = Path(t) / "a"
        mk_tree(root)
        mod.sweep([root], keep_last=1, min_age_seconds=0, dry_run=False,
                  check_hf_ckpt=always_complete)
        newest = root / "checkpoints" / "global_step_200" / "model" / ".metadata"
        oldest = root / "checkpoints" / "global_step_100" / "model" / ".metadata"
        chk("最新(200) 的 DCP 保留 ⇒ resume 不会挂", newest.is_file(),
            "" if newest.is_file() else "← 被剪了，resume 会 FileNotFoundError")
        chk("旧的(100) 的 DCP 被剪 ⇒ 看门狗正常干活", not oldest.is_file(),
            "" if not oldest.is_file() else "← 没剪，测试可能无效")

    print("\n== T2b 看门狗 keep_last=0：最新那份会被剪（证明 T2a 不是空转）==")
    with tempfile.TemporaryDirectory(prefix="resume_guard_") as t:
        root = Path(t) / "b"
        mk_tree(root)
        mod.sweep([root], keep_last=0, min_age_seconds=0, dry_run=False,
                  check_hf_ckpt=always_complete)
        newest = root / "checkpoints" / "global_step_200" / "model" / ".metadata"
        chk("keep_last=0 时最新那份被剪 ⇒ 守卫确有必要", not newest.is_file(),
            "" if not newest.is_file() else "← 竟没被剪，说明守卫是多余的？")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 72)
    print(f"  {len(results) - n_fail} 通过 / {n_fail} 失败")
    if n_fail == 0:
        print("  ✅ resume 守卫有效：RESUME=1 时看门狗不会剪掉要恢复的那份 ckpt")
    else:
        print("  ❌ 守卫失效 —— 不要用 RESUME=1 跑正式训练")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

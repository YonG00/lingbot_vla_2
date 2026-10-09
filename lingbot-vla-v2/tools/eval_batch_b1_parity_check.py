#!/usr/bin/env python3
"""旧串行 `_infer_one` vs 重构后 B1（共享 `_infer_core`）的**逐位对拍**（纯 CPU）。

背景：`_infer_one` / `_infer_batch` 被重构成共用 `_infer_core`。重构的硬约束是
**默认 serial 语义逐位不变** —— 本脚本直接按 AST 编译「旧 ref 的 `_infer_one`」
与「工作区当前文件的 `_infer_one`」，喂同一份输入，逐位比较：

* 模型**实际入参**（images/img_masks/lang_tokens/lang_masks/state/noise/grid）
* 返回的**动作输出**（以及 `ft.unapply` 后的其它键）

覆盖 4 组：dtype ∈ {float32, bfloat16} × `img_masks` 秩 ∈ {1, 2}。

用法::

    cd <repo>/lingbot-vla-v2
    python tools/eval_batch_b1_parity_check.py --old-ref aec8472

退出码：0 = 4/4 逐位一致；1 = 有不一致（必须停止后续 GPU 验收）。
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SOURCE_REL = "lingbotvla/utils/open_loop_validation.py"
if str(REPO) not in sys.path:      # `_infer_one` 内部会 import lingbotvla.utils.eval_precision
    sys.path.insert(0, str(REPO))


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _compile_validator(code: str, label: str):
    """按 AST 只取推理相关方法编译（不拉起 torchdata / 数据集依赖）。"""
    tree = ast.parse(code)
    cls = next(x for x in tree.body
               if isinstance(x, ast.ClassDef) and x.name == "OpenLoopValidator")
    cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in
                ("_infer_one", "_infer_batch", "_infer_core", "_noise_generator")]
    ast.fix_missing_locations(cls)
    namespace = {
        "torch": torch, "np": np, "os": os, "Dict": dict, "Any": object,
        "Sequence": list, "List": list, "EVAL_SEED": 1234,
        "_visual_grid_cache_clear": lambda model: None,
        "_visual_grid_cache_restore": lambda model, saved: None,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), f"<{label}>", "exec"), namespace)
    return namespace["OpenLoopValidator"]


class Policy(torch.nn.Module):
    """逐样本确定性策略：记录**实际入参**，输出 = noise + state[:, :1]。"""

    def __init__(self, dtype):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.calls = []

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, *,
                       noise, image_grid_thw=None):
        self.calls.append(tuple(
            None if v is None else v.detach().cpu().clone()
            for v in (images, img_masks, lang_tokens, lang_masks, state, noise, image_grid_thw)))
        return noise + state[:, :1].reshape(-1, 1, 1)


class Transform:
    def unapply(self, item):
        return {"actions": item["actions"], "state": item["state"]}


def _instance(cls, dtype):
    v = object.__new__(cls)
    v.model = Policy(dtype)
    v.device = "cpu"
    v._model_config = SimpleNamespace(n_action_steps=2, max_action_dim=3, action_fp32=False)
    v.args = SimpleNamespace(train=SimpleNamespace(eval_inference_dtype="auto", use_bf16=False))
    v._noise_gen = None
    v._precision_logged = True
    v.dump_dir = None
    v._dump_prefix = None
    return v


def _item(state_value, mask_rank, dtype):
    return {
        "images": torch.randn(2, 3, 2, 2).to(dtype),
        "img_masks": torch.ones((2,) if mask_rank == 1 else (1, 2)),
        "lang_tokens": torch.tensor([1, 2]),
        "lang_masks": torch.ones(2),
        "state": torch.tensor([state_value], dtype=dtype),
        "actions": torch.zeros(2, 3),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-ref", default="HEAD",
                        help="旧实现的 git ref（默认 HEAD；重构提交后请显式指定其父提交）")
    parser.add_argument("--repo", default=str(REPO))
    args = parser.parse_args(argv)
    repo = Path(args.repo)
    old_code = subprocess.check_output(
        ["git", "show", f"{args.old_ref}:./{SOURCE_REL}"], cwd=str(repo), text=True)
    new_code = _load(repo / SOURCE_REL)
    print(f"旧实现 = {args.old_ref}:{SOURCE_REL}  ({len(old_code)} chars)")
    print(f"新实现 = 工作区 {SOURCE_REL}  ({len(new_code)} chars)")
    old_cls, new_cls = _compile_validator(old_code, "old"), _compile_validator(new_code, "new")

    failures = []
    for dtype in (torch.float32, torch.bfloat16):
        for mask_rank in (1, 2):
            torch.manual_seed(31415)
            item = _item(4, mask_rank, dtype)
            a, b = _instance(old_cls, dtype), _instance(new_cls, dtype)
            out_a = a._infer_one(item, Transform())
            out_b = b._infer_one(item, Transform())
            checks = {
                "keys": set(out_a) == set(out_b),
                "outputs": all(torch.equal(out_a[k], out_b[k]) for k in out_a),
                "one_call_each": len(a.model.calls) == len(b.model.calls) == 1,
                "model_inputs": all((x is None and y is None) or torch.equal(x, y)
                                    for x, y in zip(a.model.calls[0], b.model.calls[0])),
            }
            ok = all(checks.values())
            if not ok:
                failures.append((str(dtype), mask_rank, checks))
            print(f"B1 baseline exact parity: dtype={str(dtype):<16} maskdim={mask_rank} "
                  f"=> {'PASS' if ok else 'FAIL ' + str(checks)}")
    if failures:
        print(f"\n❌ {len(failures)}/4 组不一致 ⇒ 重构改变了默认 serial 语义，禁止继续 GPU 验收")
        return 1
    print("\n✅ 4/4 组逐位一致（入参 + 输出）：默认 serial 语义未被重构改变")
    return 0


if __name__ == "__main__":
    sys.exit(main())

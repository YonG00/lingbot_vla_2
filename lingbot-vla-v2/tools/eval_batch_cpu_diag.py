#!/usr/bin/env python3
"""CPU 替身诊断入口：用假模型/假数据集**真实执行** `_infer_one` / `_infer_batch`。

为什么需要它（2026-10-09 教训）
--------------------------------
Batch2 与 serial 的动作误差**弥散**（14/14 维、多时间步，max ≈ 0.011–0.058），
但三问全无答案：噪声是否一致 / serial 自身是否稳定 / 首个差异在入参还是
`sample_actions` 内部。原因是**上一轮的三份诊断钩子本身不可判**：

* 文件名带毫秒时间戳、内容里没有样本身份 ⇒ 只能做"时间配对"；
* `AL_EVAL_BATCH_PROBE_REPEAT_SERIAL` 钩子 `UnboundLocalError` 被 `except Exception`
  吞掉 ⇒ **文件从未产出，却零失败信号**；
* 没有任何机制检查"批处理真正喂给模型的入参"。

本工具在 CPU 上把整条诊断链路跑通（**不需要 GPU**），并且：

1. 假模型是**逐样本**的（`images[i]`/`state[i]` 的均值只影响第 i 条输出）——
   所以"入参同、噪声同 ⇒ 输出逐位同"是**可判的**不变式，正常组差值必须恒等于 0；
2. 三类证据（`noise` / `serial_repeat` / `batch_actions`）**每条都带完整样本身份**：
   task / episode_id / chunk_start / dataset_index / inference_path / batch_position
   / repeat_index —— 不靠时间戳推断；
3. Serial Repeat = 同进程、同模型状态、**显式同一份 noise**（generator state 复位），
   并断言两次喂进去的噪声**逐位相同**（断言的是录制到的实际噪声，不是"复位了就算"）；
4. Batch1/2 对拍比对**实际入参**：images / img_masks / lang_tokens / lang_masks /
   state / image_grid_thw / noise，逐个字段记录逐位一致性结论；
5. **自动验收（fail-closed）**：文件缺失 / 钩子未执行 / 样本 ID 不匹配 / 非有限值 /
   应逐位一致的项不一致 ⇒ `BLOCKED` + 非零退出，**没有静默 catch**；
6. **扰动反例**证明检测器真的能识别：错误输入 / 错误噪声 / 错误样本对应 /
   仅输出偏差 / 非有限值 / 身份重复 / 钩子未执行 / 证据文件缺失。

退出码
------
* ``0``  —— 正常组 `PASS` 且**所有**扰动反例都被检出（`DETECTED`）；
* ``2``  —— `BLOCKED`（正常组有 problem）或 `MISSED`（有反例没被检出）或环境不可用。

用法
----
    cd /data/code/lingbot-vla-v2
    /data/miniconda3/envs/lingbotvla/bin/python tools/eval_batch_cpu_diag.py \\
        --out-dir /data/tmp/eval_batch_cpu_diag

    # 只跑正常组（不跑扰动反例）：
    ... --no-selftest
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lingbotvla.utils import eval_batch_probe as ebp  # noqa: E402


# 与真实数据契约一致（见 `_infer_batch` 的守卫）：
#   lang_tokens / lang_masks / state **逐条 1-D**；images 逐条 3-D；image_grid_thw 逐条 (1,3)
N_ACTION_STEPS = 50
MAX_ACTION_DIM = 14          # 真实动作是 14 维（2026-10-09 实测：14/14 维都有差异）
IMG_SHAPE = (3, 8, 8)
LANG_LEN = 5
STATE_DIM = 7
EPISODE_ID = 51
CHUNK_OFFSETS = (0, 50)      # 两条 chunk 相差 50 帧（= 官方 chunk_size 步进）


# ---------------------------------------------------------------------------
# 假模型 / 假数据
# ---------------------------------------------------------------------------
class FakeModel(torch.nn.Module):
    """**逐样本**确定性假模型：``out[i] = noise[i] + 0.01*(mean(images[i]) + ...)``。

    ⚠️ 关键点：**不能**用 `images.float().mean()`（跨样本求均值）。那样 batch 与
    serial 的输入统计量天生不同 ⇒ 正常组也会出现 0.003 的"差异"，把不变式搞成
    "差不多"而不是"逐位相等"，检测器就再也分不清"真 bug"和"假模型自己的锅"。
    逐样本归约在 batch/serial 下逐位一致（torch 按行归约顺序相同，已在 CPU 上验过）。
    """

    def __init__(self, *, batch_bias: float = 0.0, nan_in_batch: bool = False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))   # 让 `_infer_one` 能取到权重 dtype
        self.batch_bias = float(batch_bias)
        self.nan_in_batch = bool(nan_in_batch)
        self.calls: List[Dict[str, Any]] = []

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       noise=None, image_grid_thw=None):
        batch = int(images.shape[0])
        per_sample = (images.float().reshape(batch, -1).mean(dim=1)
                      + lang_tokens.float().reshape(batch, -1).mean(dim=1)
                      + state.float().reshape(batch, -1).mean(dim=1)) * 0.01
        noisy = noise.clone() if isinstance(noise, torch.Tensor) else torch.zeros(
            batch, N_ACTION_STEPS, MAX_ACTION_DIM)
        out = noisy + per_sample.to(noisy.dtype).view(batch, 1, 1)
        if self.batch_bias and batch > 1:
            out = out + self.batch_bias * torch.arange(1, batch + 1,
                                                       dtype=out.dtype).view(batch, 1, 1)
        if self.nan_in_batch and batch > 1:
            out = out.clone()
            out[0, 0, 0] = float("nan")
        self.calls.append({"batch": batch, "noise_shape": tuple(noisy.shape),
                           "images_shape": tuple(images.shape)})
        return out


class FakeFeatureTransform:
    """`ft.unapply` 在诊断里保持恒等（我们比的是**模型原始输出**，不是物理量）。"""

    def unapply(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return dict(item)


def make_item(dataset_index: int, value: float) -> Dict[str, Any]:
    """构造一条**已 transform** 的样本（形状/维度严格照真实契约）。"""
    return {
        "images": torch.full(IMG_SHAPE, float(value), dtype=torch.float32),
        "img_masks": torch.ones(1, dtype=torch.bool),
        "lang_tokens": torch.full((LANG_LEN,), float(value), dtype=torch.long),
        "lang_masks": torch.ones(LANG_LEN, dtype=torch.bool),
        "state": torch.full((STATE_DIM,), float(value), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, IMG_SHAPE[1], IMG_SHAPE[2]]], dtype=torch.long),
        "actions": torch.zeros(N_ACTION_STEPS, MAX_ACTION_DIM, dtype=torch.float32),
        "_meta": {"task": "click_bell", "episode_id": EPISODE_ID,
                  "dataset_index": int(dataset_index)},
    }


class _Args:
    class _Train:
        eval_inference_dtype = "auto"
        use_bf16 = False
        seed = 1234
        global_rank = 0
    train = _Train()


class _Logger:
    """把真实推理路径里的诊断日志打出来（`[diag] probe captured …` 就是"钩子已执行"证据）。"""

    def __init__(self, sink: Optional[List[str]] = None, verbose: bool = True):
        self.lines: List[str] = sink if sink is not None else []
        self.verbose = verbose

    def info_rank0(self, msg: str) -> None:
        self.lines.append(str(msg))
        if self.verbose:
            print(f"    | {msg}")

    def warning(self, msg: str) -> None:
        self.lines.append(f"WARNING {msg}")
        if self.verbose:
            print(f"    ! {msg}")

    warning_rank0 = warning


def build_validator(model: torch.nn.Module, logger: _Logger):
    """构造一个**跳过重依赖 `__init__`、但用真实方法**的 validator。"""
    from lingbotvla.utils.open_loop_validation import OpenLoopValidator

    class _Shim(OpenLoopValidator):
        pass

    validator = object.__new__(_Shim)
    validator._noise_gen = None
    validator.model = model
    validator.device = torch.device("cpu")
    validator.args = _Args()
    validator.logger = logger
    validator.dump_dir = None            # `_infer_one` 会读它（真实属性，必须存在）
    validator._dump_prefix = None
    validator._precision_logged = True
    validator._model_config = type("C", (), {
        "chunk_size": N_ACTION_STEPS, "n_action_steps": N_ACTION_STEPS,
        "max_action_dim": MAX_ACTION_DIM, "action_fp32": False})()
    # 诊断录制相关（真实生产属性）
    validator._probe_recorder = None
    validator._probe_group_ids = None
    validator._probe_pass = "serial"
    validator._probe_position = 0
    validator._probe_ep_map = None
    validator._probe_task = None
    validator._probe_tag = None
    return validator


# ---------------------------------------------------------------------------
# 一次 probe 组：真实 `_infer_one` / `_infer_batch`
# ---------------------------------------------------------------------------
def _fake_ep_map() -> np.ndarray:
    """所有 chunk 都属于同一个 episode ⇒ chunk_start = dataset_index - 回合首行。"""
    return np.full(int(CHUNK_OFFSETS[-1]) + 1, EPISODE_ID, dtype=np.int64)


def run_probe(*, out_dir: str, fault: Optional[str] = None, repeat: bool = True,
              logger: Optional[_Logger] = None) -> Dict[str, Any]:
    """跑一次诊断组并落盘三类证据 + 判定。``fault`` 注入**上游**故障（反例用）。

    故障全部注入在**真实方法的上游**（模型 / 入参 / generator / 身份声明），
    所以走的仍然是生产代码路径，而不是"手改证据文件"。
    """
    log = logger or _Logger(verbose=False)
    model = FakeModel(batch_bias=(1e-3 if fault == "batch_bug" else 0.0),
                      nan_in_batch=(fault == "nonfinite"))
    validator = build_validator(model, log)
    ft = FakeFeatureTransform()

    items = [make_item(0, 0.5), make_item(1, 0.7)]
    starts = list(CHUNK_OFFSETS)
    validator._probe_ep_map = _fake_ep_map()
    validator._probe_task = "click_bell"
    validator._probe_tag = "cpu_diag"
    validator._probe_group_ids = [validator._probe_base_identity(i) for i in starts]
    if fault == "identity":
        # 反例：两条位置声明成**同一个样本**（身份重复 ⇒ 必须被检出）
        validator._probe_group_ids[1] = dict(validator._probe_group_ids[0])
    validator._probe_recorder = ebp.ProbeRecorder()
    if fault == "hook_missing":
        # 反例：模拟"串行侧的钩子根本没执行"（身份没准备好 ⇒ 录制为空）
        validator._probe_group_ids = None

    gen = validator._noise_generator(validator.device)
    generator_start = gen.get_state()
    serial = validator._infer_serial_group(items, ft, path="serial")
    serial_repeat: List[Dict[str, Any]] = []
    if repeat:
        gen.set_state(generator_start)               # 显式同一份 noise
        serial_repeat = validator._infer_serial_group(items, ft, path="serial_repeat")
    generator_end = gen.get_state()

    if fault == "hook_missing":
        validator._probe_group_ids = [validator._probe_base_identity(i) for i in starts]
    batch_items = items
    if fault == "input":
        batch_items = [items[0], make_item(1, 0.7)]
        batch_items[1]["state"] = batch_items[1]["state"] + 1.0     # 批处理拿到不同入参
    if fault == "swap":
        batch_items = [items[1], items[0]]                          # 批内顺序被交换
    gen.set_state(generator_start)
    if fault == "noise":
        gen.manual_seed(20261009)                                   # 批处理拿到不同噪声
    batched = validator._infer_batch(batch_items, ft)
    gen.set_state(generator_end)

    parity = _production_parity(serial, batched)
    verdict = validator._write_probe_evidence(
        out_dir, require_repeat=repeat, group_index=0,
        extra={"fault": fault, "entry": "cpu_diag", "device": "cpu",
               "production_parity": bool(parity),
               "production_parity_atol": 1e-5, "production_parity_rtol": 1e-3,
               "model_calls": list(model.calls)})
    verdict["production_parity"] = bool(parity)
    # 反例 8：证据文件缺失必须被检出（在**已落盘**的目录上按判定器口径复核）
    if fault == "dropfile":
        os.remove(os.path.join(out_dir, "noise.npz"))
        verdict = ebp.detect_problems(validator._probe_recorder,
                                      [dict(b) for b in validator._probe_group_ids],
                                      out_dir, require_repeat=repeat)
        ebp.write_verdict(out_dir, verdict)
    return verdict


def _production_parity(serial: Sequence[Dict[str, Any]],
                       batched: Sequence[Dict[str, Any]]) -> bool:
    """用**生产同款**谓词（`outputs_close`, atol=1e-5, rtol=1e-3）比一次。"""
    from lingbotvla.auto_learning.eval_batch_policy import outputs_close
    from lingbotvla.auto_learning.scan_accel import normalized_action_predictions
    keys = ("actions",)
    try:
        return bool(outputs_close(normalized_action_predictions(serial, keys),
                                  normalized_action_predictions(batched, keys),
                                  atol=1e-5, rtol=1e-3))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------------
#: 反例 → 必须出现的 problem code 前缀（检测器灵敏度证明）
NEGATIVE_CONTROLS: Tuple[Tuple[str, str], ...] = (
    ("input", "batch_input_not_bitwise:state"),
    ("noise", "batch_noise_not_bitwise"),
    ("swap", "sample_correspondence_mismatch"),
    ("batch_bug", "batch_output_not_bitwise"),
    ("nonfinite", "nonfinite:"),
    ("identity", "duplicate_"),
    ("hook_missing", "hooks_not_executed"),
    ("dropfile", "missing_or_empty:noise.npz"),
)


def _verdict_path(out_dir: str) -> str:
    return os.path.join(out_dir, "verdict.json")


def run_all(out_dir: str, *, selftest: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, Any] = {"out_dir": out_dir, "groups": {}, "problems": []}

    print("=" * 78)
    print("① 正常组（同入参 / 同噪声 / 逐样本假模型）⇒ 必须 PASS 且所有逐位检查 True")
    print("=" * 78)
    ok_dir = os.path.join(out_dir, "ok")
    ok_verdict = run_probe(out_dir=ok_dir, fault=None, repeat=True, logger=_Logger())
    ok_verdict["status_expected"] = "PASS"
    ok_verdict["state"] = "PASS" if ok_verdict["status"] == "PASS" else "BLOCKED"
    ebp.write_verdict(ok_dir, ok_verdict)
    summary["groups"]["ok"] = _digest(ok_verdict)
    print(f"    ⇒ status={ok_verdict['status']} problems={ok_verdict['problems']} "
          f"production_parity={ok_verdict.get('production_parity')}")
    if ok_verdict["status"] != "PASS":
        summary["problems"].append(f"normal_group_{ok_verdict['status']}")

    if selftest:
        print("=" * 78)
        print("② 扰动反例（上游注入）⇒ 检测器必须报 BLOCKED 且命中指定 problem code")
        print("=" * 78)
        for fault, expected_code in NEGATIVE_CONTROLS:
            fault_dir = os.path.join(out_dir, f"neg_{fault}")
            verdict = run_probe(out_dir=fault_dir, fault=fault, repeat=True, logger=_Logger())
            hit = any(p.startswith(expected_code) for p in verdict["problems"])
            state = "DETECTED" if hit else "MISSED"
            verdict["status_expected"] = "BLOCKED"
            verdict["expected_problem_prefix"] = expected_code
            verdict["state"] = state
            verdict["detected"] = bool(hit)
            ebp.write_verdict(fault_dir, verdict)
            summary["groups"][f"neg_{fault}"] = _digest(verdict)
            print(f"    [{state}] fault={fault:<12} expected≈{expected_code:<34} "
                  f"problems={verdict['problems'][:4]}")
            if not hit:
                summary["problems"].append(f"detector_missed:{fault}")

    summary["status"] = "PASS" if not summary["problems"] else "BLOCKED"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print("=" * 78)
    print(f"总结论 = {summary['status']}  problems={summary['problems']}")
    print("=" * 78)
    return summary


def _digest(verdict: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": verdict["status"], "state": verdict.get("state"),
            "problems": list(verdict["problems"]),
            "n_records": verdict.get("n_records"),
            "files": {k: {kk: vv for kk, vv in (v or {}).items() if kk.startswith("sha256")}
                      for k, v in (verdict.get("files") or {}).items()},
            "production_parity": verdict.get("production_parity"),
            "per_sample": verdict.get("per_sample")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU 替身 Eval Batch 诊断（真实 _infer_one/_infer_batch + 身份化证据 + 扰动反例）")
    parser.add_argument("--out-dir", required=True, help="证据输出目录（每组一个子目录）")
    parser.add_argument("--no-selftest", action="store_true",
                        help="只跑正常组，不跑扰动反例")
    parser.add_argument("--clean", action="store_true", help="先清空 --out-dir（默认拒绝覆盖非空目录）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.clean:
        print(f"[BLOCKED] 输出目录非空，拒绝覆盖: {out_dir}（要覆盖请加 --clean）")
        return 2
    if args.clean and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    try:
        import lingbotvla.utils.open_loop_validation  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[BLOCKED] 无法导入真实推理路径（缺依赖？）: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 2
    try:
        summary = run_all(out_dir, selftest=not args.no_selftest)
    except Exception as exc:  # noqa: BLE001
        print(f"[BLOCKED] 诊断执行失败: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 2
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())

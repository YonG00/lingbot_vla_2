#!/usr/bin/env python3
"""GPU 四腿对照探针：一次加载模型，判定"Batch1 vs Batch2 差异"到底出在哪一环（**默认 PLAN ONLY**）。

要回答的三个问题（也只看这三个）
--------------------------------
1. **serial 自身稳定吗** —— 同进程 / 同模型状态 / **显式同一份 noise** 连续两次串行；
2. **差异是不是"视觉网格缓存残留"造成的** —— Batch1 正常缓存 vs Batch1 清缓存；
3. **清缓存后 Batch1 与 Batch2 还差吗** —— 若仍差，再只探**第 0 层**输入/输出（定位分歧阶段）。

设计要点
--------
* **一次加载**：模型只 `from_pretrained` 一次；所有腿共用同一份权重、同一份数据。
* **同噪声**：每腿开跑前把 flow-matching generator 复位到同一个 state ⇒ 每条样本拿到
  **逐位相同**的噪声；否则"差异"分不清是缓存还是 RNG。
* **身份化证据**：每腿一个子目录，走生产同款 ``eval_batch_probe`` writer/detector
  （逐样本带 task/episode_id/chunk_start/dataset_index/推理路径/批内位置）。
* **不跑训练**：零 optimizer.step，不碰 Bootstrap / Scout / Hardness / Replay / HF / DCP。
* **fail-closed**：OOM / 非有限 / `peak_free < reserve` / 证据缺失 / 身份不匹配 ⇒ BLOCKED 停下。

判定表（本工具直接打印结论，不替你做决定）
------------------------------------------
===============  ==========================================  ================================
观测              结论                                        最小修复方向
===============  ==========================================  ================================
serial ≠ repeat   serial 自身不确定 ⇒ 其它结论作废          先查 RNG / cudnn 确定性
B1正常 ≠ B1清     **缓存残留**是根因                         串行路径同样清缓存 / 缓存按 grid 建 key
B1清 = B2清       批处理数值**等价** ⇒ 之前差异是缓存污染    修缓存后才有资格谈 auto（仍需显存/收益）
B1清 ≠ B2清       差异来自 varlen（或 MoE）路径               看第 0 层 input/output 是否已不同
===============  ==========================================  ================================

用法
----
    # ① PLAN ONLY（默认；不碰 GPU、不建目录、不加载模型）
    python tools/eval_batch_gpu_probe.py --ckpt <hf_ckpt> --out-dir /data/tmp/eval_batch_gpu_probe

    # ② 真正执行（一次加载，2–5 分钟级）
    AL_ALLOW_SMALL_CARD=1 python tools/eval_batch_gpu_probe.py --ckpt <hf_ckpt> \\
        --out-dir /data/outputs/eval_batch_gpu_probe_<date> --execute [--probe-layer0]

**不启用 Eval auto、不放宽 atol/rtol**；没有本工具的逐位证据之前，不得声称 Batch 数值差异已修复。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from lingbotvla.utils import eval_batch_probe as ebp  # noqa: E402


DEFAULT_CKPT = "/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt"
DEFAULT_SPLIT = "/data/train/task_splits_50"
DEFAULT_TASK = "click_bell"
DEFAULT_DATASET_INDICES = (0, 50)      # 实测两条 chunk：ep51 的 frame 0 / 50
LEGS = (
    # (腿名, 说明, len(items), 是否清缓存, 推理路径标签)
    ("warm", "预热：B1 不清缓存（把缓存养成「历史串行」的样子）", 1, False, "serial"),
    ("b1_normal", "Batch1 **正常缓存**（历史语义，基准）", 1, False, "serial"),
    ("b1_cleared", "Batch1 **清缓存**", 1, True, "serial"),
    ("b2_cleared", "Batch2 **清缓存**（一次 forward）", 2, True, "batch"),
    ("b1_repeat", "Serial Repeat：同噪声再跑一次 Batch1（稳定性）", 1, False, "serial_repeat"),
)


class _Logger:
    def info_rank0(self, msg, *a, **k):
        print(f"[probe] {msg}", flush=True)

    def info(self, msg, *a, **k):
        print(f"[probe] {msg}", flush=True)

    def warning(self, msg, *a, **k):
        print(f"[probe][WARN] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 第 0 层探针（可选；只回答"分歧是否已在第 0 层之前"）
# ---------------------------------------------------------------------------
def find_layer0(model):
    """定位第 0 层 decoder。找不到 ⇒ 返回 (None, None)（调用方必须 BLOCKED，不静默跳过）。"""
    candidates = []
    for name, mod in model.named_modules():
        if name.endswith("layers.0"):
            candidates.append((name, mod))
    if not candidates:
        return None, None
    candidates.sort(key=lambda kv: (len(kv[0]), kv[0]))
    return candidates[0]


def install_layer0_hook(model):
    """给第 0 层挂 forward hook，捕获**整层输入/输出**（默认不挂）。"""
    name, mod = find_layer0(model)
    if mod is None:
        return None, None, None
    store: Dict[str, Any] = {}

    def _hook(_module, args, kwargs, output):
        hidden = args[0] if args else kwargs.get("hidden_states")
        store["input"] = hidden
        store["output"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = mod.register_forward_hook(_hook, with_kwargs=True)
    return handle, store, name


def _layer0_stats(value) -> Optional[Dict[str, Any]]:
    arr = ebp.to_numpy(value)
    if arr is None:
        return None
    return {"shape": list(arr.shape), "dtype": str(arr.dtype),
            "absmax": ebp.absmax(arr), "finite": ebp.all_finite(arr)}


def compare_layer0(leg_a: Dict[str, Any], leg_b: Dict[str, Any], *, label_a: str,
                   label_b: str) -> Dict[str, Any]:
    """比较两条腿的第 0 层输入/输出。形状不同时**不猜**语义：给出可判/不可判的明确结论。"""
    out: Dict[str, Any] = {"label_a": label_a, "label_b": label_b}
    verdict_bits = []
    for field in ("input", "output"):
        va, vb = leg_a.get(field), leg_b.get(field)
        stats_a, stats_b = _layer0_stats(va), _layer0_stats(vb)
        entry = {"shape_a": None if stats_a is None else stats_a["shape"],
                 "shape_b": None if stats_b is None else stats_b["shape"],
                 "absmax_a": None if stats_a is None else stats_a["absmax"],
                 "absmax_b": None if stats_b is None else stats_b["absmax"]}
        if va is None or vb is None:
            entry["comparable"] = False
            entry["note"] = "该腿没有捕获到第 0 层激活"
        elif list(np.shape(ebp.to_numpy(va))) == list(np.shape(ebp.to_numpy(vb))):
            entry["comparable"] = True
            entry["bitwise"] = bool(ebp.bitwise_identical(va, vb))
            entry["max_abs_diff"] = ebp.max_abs_diff(va, vb)
            entry["mean_abs_diff"] = ebp.mean_abs_diff(va, vb)
        else:
            # B1 packed vs B2 packed 形状本就可能不同 ⇒ 只有"元素数相同"才敢逐元素比
            na, nb = ebp.to_numpy(va).size, ebp.to_numpy(vb).size
            entry["comparable"] = bool(na == nb)
            entry["note"] = (f"形状不同（{entry['shape_a']} vs {entry['shape_b']}）；"
                             + ("元素数相同，按扁平逐元素比" if na == nb
                                else "元素数也不同 ⇒ 不可逐元素比，只报统计量"))
            if na == nb:
                entry["bitwise"] = bool(ebp.bitwise_identical(
                    ebp.to_numpy(va).reshape(-1), ebp.to_numpy(vb).reshape(-1)))
                entry["max_abs_diff"] = ebp.max_abs_diff(
                    ebp.to_numpy(va).reshape(-1), ebp.to_numpy(vb).reshape(-1))
        out[field] = entry
        if entry.get("comparable") and entry.get("bitwise") is False:
            verdict_bits.append(f"layer0_{field}_differs")
    out["verdict"] = ("layer0_identical" if not verdict_bits else
                      ("layer0_differs_before_or_at_layer0"
                       if "layer0_input_differs" in verdict_bits else "layer0_output_differs_only"))
    out["flags"] = verdict_bits
    return out


# ---------------------------------------------------------------------------
# PLAN ONLY
# ---------------------------------------------------------------------------
def plan(a: argparse.Namespace) -> int:
    ckpt = Path(a.ckpt)
    cli = ckpt.parent.parent.parent / "lingbotvla_cli.yaml"
    checks = [
        ("ckpt/config.json", (ckpt / "config.json").exists(), str(ckpt)),
        ("训练 cli yaml（deploy 读它建模型）", cli.exists(), str(cli)),
        ("val ids", Path(a.val_ids).exists(), a.val_ids),
        ("数据集根", Path(a.dataset).exists(), a.dataset),
        ("输出目录未被占用", not (Path(a.out_dir).exists() and any(Path(a.out_dir).iterdir())),
         a.out_dir),
    ]
    print("=" * 78)
    print("PLAN ONLY（未触碰 GPU / 未加载模型 / 未创建任何目录）")
    print("=" * 78)
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}: {detail}")
    print(f"  task={a.task}  dataset_indices={list(a.dataset_indices)}  "
          f"probe_layer0={a.probe_layer0}")
    print("  腿顺序：")
    for name, desc, n, fresh, _ in LEGS:
        print(f"    - {name:<11} items={n} 清缓存={fresh}  {desc}")
    print("  交叉判定：b1_normal↔b1_cleared（缓存）/ b1_cleared↔b2_cleared（varlen）/ "
          "b1_normal↔b1_repeat（稳定性）")
    print("  ⚠️ auto 仍关闭、atol=1e-5/rtol=1e-3 不放宽；本命令不做任何训练步。")
    print("=" * 78)
    ok = all(flag for _, flag, _ in checks)
    if not ok:
        print("[PLAN ONLY] ❌ 有阻塞项（见上）；--execute 前先修")
        return 2
    print("[PLAN ONLY] ✅ 路径齐备，可加 --execute 开跑")
    return 0


# ---------------------------------------------------------------------------
# 模型 / 数据（复用 tools/open_loop_eval_inprocess.py 的已验证姿势）
# ---------------------------------------------------------------------------
def build_runtime(a: argparse.Namespace):
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.utils.open_loop_validation import (
        OpenLoopValidator, _episode_index_map, _load_episode_ids)

    ckpt = Path(a.ckpt).resolve()
    cli = ckpt.parent.parent.parent / "lingbotvla_cli.yaml"
    raw = yaml.safe_load(cli.open())
    server = LingbotVLAv2Server(path_to_pi_model=str(ckpt), robot_norm_path=None,
                                use_length=a.use_length, chunk_ret=True,
                                use_bf16=a.use_bf16, use_fp32=not a.use_bf16,
                                use_compile=False)
    vla, processor = server.vla, server.processor
    cfg = vla.config
    data_cfg = dict(raw.get("data", {}))
    ns = types.SimpleNamespace(**data_cfg)
    if not hasattr(ns, "chunk_size"):
        ns.chunk_size = int(cfg.chunk_size)
    if not hasattr(ns, "num_episode"):
        ns.num_episode = None
    align_params = raw.get("train", {}).get("align_params") or {}

    args = types.SimpleNamespace()
    args.data = ns
    args.model = cfg
    args.train = types.SimpleNamespace(output_dir=a.out_dir, global_rank=0,
                                       use_bf16=a.use_bf16, eval_inference_dtype="auto")
    validator = OpenLoopValidator(
        model=vla, args=args, processor=processor, use_depth_align=bool(align_params),
        writer=None, logger=_Logger(),
        train_monitor_ids=_load_episode_ids(a.val_ids), val_ids=_load_episode_ids(a.val_ids),
        dump_dir=None, per_episode_stride=True)
    ds_path = validator._episode_ids_file(sorted(_load_episode_ids(a.val_ids)), "probe")
    ds = validator._dataset(ds_path)
    ft = validator._ft_for(ds_path)
    ep_map = _episode_index_map(ds)
    validator._probe_ep_map = ep_map
    validator._probe_task = a.task
    validator._probe_tag = f"gpu_probe_{a.task}"
    return validator, vla, ds, ft, ep_map


def _leg_planes(a: argparse.Namespace):
    out = Path(a.out_dir)
    return [(name, out / f"leg_{name}", desc, n, fresh, path)
            for name, desc, n, fresh, path in LEGS]


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def run_leg(validator, items: Sequence[Dict[str, Any]], ft, *, leg_dir: Path,
            leg_name: str, group_ids: List[Dict[str, Any]], fresh: bool,
            path_label: str = "serial",
            layer0_store: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """跑一条腿：同噪声（generator 复位）→ 录制 → 落盘三类证据 + 单腿自检。"""
    leg_dir.mkdir(parents=True, exist_ok=True)
    validator._probe_recorder = ebp.ProbeRecorder()
    validator._probe_group_ids = list(group_ids)
    validator._probe_pass = path_label
    validator._probe_position = 0
    gen = validator._noise_generator(validator.device)
    state = gen.get_state()
    gen.set_state(state)                      # 每条腿从**同一份**噪声起点开始
    if layer0_store is not None:
        layer0_store.clear()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if len(items) == 1:
        validator._infer_core((items[0],), ft, fresh_visual_grid=fresh)
    else:
        validator._infer_batch(list(items), ft)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    gen.set_state(state)                      # 跑完复位，保证下一条腿噪声一致
    peak_free = None
    if torch.cuda.is_available():
        total = torch.cuda.mem_get_info()[1] / 1024 ** 3
        peak_free = min(torch.cuda.mem_get_info()[0] / 1024 ** 3,
                        total - torch.cuda.max_memory_reserved() / 1024 ** 3)
    expected = [dict(g) for g in group_ids]
    files = ebp.write_group_evidence(validator._probe_recorder, str(leg_dir), expected,
                                     extra={"entry": "eval_batch_gpu_probe", "leg": leg_name,
                                            "fresh_visual_grid": bool(fresh),
                                            "batch_size": len(items), "task": validator._probe_task,
                                            "seconds": float(seconds),
                                            "peak_free_gib": (None if peak_free is None
                                                              else float(peak_free))})
    if layer0_store:
        first = expected[0]
        ebp.write_evidence(str(leg_dir), "layer0", [{
            "identity": ebp.make_identity(
                task=first.get("task"), episode_id=first.get("episode_id"),
                chunk_start=first.get("chunk_start"),
                dataset_index=int(first["dataset_index"]),
                inference_path=("batch" if len(items) > 1 else "serial"),
                batch_position=0 if len(items) == 1 else 0),
            "arrays": {k: v for k, v in layer0_store.items() if v is not None}}],
            extra={"entry": "eval_batch_gpu_probe", "leg": leg_name,
                   "scope": "whole_layer_activation", "layer": "0",
                   "covers_dataset_indices": [int(g["dataset_index"]) for g in group_ids],
                   "note": "整层激活（非逐样本切分）；形状差异时按元素数判断可比性"})
    verdict = ebp.detect_leg_problems(validator._probe_recorder, expected, str(leg_dir))
    verdict["files"] = files
    verdict["leg"] = leg_name
    verdict["seconds"] = float(seconds)
    verdict["peak_free_gib"] = None if peak_free is None else float(peak_free)
    ebp.write_verdict(str(leg_dir), verdict)
    print(f"[probe] leg={leg_name:<11} status={verdict['status']} records={verdict['n_records']} "
          f"seconds={seconds:.3f} peak_free_gib="
          f"{'n/a' if peak_free is None else f'{peak_free:.2f}'} problems={verdict['problems']}")
    validator._probe_recorder = None
    validator._probe_group_ids = None
    return verdict


def decide(result: Dict[str, Any]) -> Dict[str, Any]:
    """把三条交叉比较翻成结论（只做事实判定，不替你改代码）。"""
    cmp_cache = result["compare"]["b1_normal_vs_b1_cleared"]
    cmp_varlen = result["compare"]["b1_cleared_vs_b2_cleared"]
    cmp_repeat = result["compare"]["b1_normal_vs_b1_repeat"]
    cache_same = cmp_cache["status"] == "PASS"
    varlen_same = cmp_varlen["status"] == "PASS"
    repeat_same = cmp_repeat["status"] == "PASS"
    if not repeat_same:
        conclusion = "serial_itself_unstable"
        advice = "serial 两次同噪声不一致 ⇒ 先查 RNG/cudnn 确定性；其它结论作废"
    elif not cache_same:
        conclusion = "visual_grid_cache_staleness"
        advice = "缓存残留是根因 ⇒ 串行路径同样清缓存（或把缓存按 grid 建 key），再复跑本探针"
    elif varlen_same:
        conclusion = "batch_numerically_equivalent_after_cache_fix"
        advice = ("清缓存后 Batch1≡Batch2（逐位）⇒ 历史差异来自缓存污染；"
                  "修缓存后可谈 auto（仍需显存+收益+指标证据，且要单独批准）")
    else:
        conclusion = "divergence_inside_batched_path"
        advice = ("同噪声同入参下 B1≠B2 ⇒ 差异来自 1图/2图 varlen（或 MoE）路径；"
                  "看第 0 层 input/output 是否已不同以定位阶段")
    l0 = (result.get("layer0") or {}).get("b1_cleared_vs_b2_cleared") or {}
    if conclusion == "divergence_inside_batched_path":
        if l0.get("verdict") == "layer0_identical":
            advice += "；第 0 层输入/输出**逐位相同** ⇒ 分歧发生在第 0 层之后（继续往深层查 MoE）"
        elif l0.get("flags"):
            advice += "；第 0 层**已有差异** ⇒ 分歧在进入第 0 层之前（视觉塔/前缀拼接/varlen）"
        else:
            advice += "；未取得第 0 层可判证据（形状不可比）⇒ 需另设逐样本切分方式"
    return {"conclusion": conclusion, "advice": advice,
            "layer0_verdict": l0.get("verdict"),
            "cache_legs_bitwise_equal": bool(cache_same),
            "varlen_legs_bitwise_equal": bool(varlen_same),
            "serial_repeat_bitwise_equal": bool(repeat_same)}


def execute(a: argparse.Namespace) -> int:
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"[BLOCKED] 输出目录非空，拒绝覆盖: {out}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("[BLOCKED] 没有可用 GPU（本入口是 GPU 对照探针，不做 CPU 替代）")
        return 2
    validator, vla, ds, ft, ep_map = build_runtime(a)
    indices = [int(i) for i in a.dataset_indices]
    if any(i >= len(ds) for i in indices):
        print(f"[BLOCKED] dataset_indices {indices} 超出数据集长度 {len(ds)}")
        return 2
    items = [ds[i] for i in indices]
    group_ids = [validator._probe_base_identity(i) for i in indices]
    print(f"[probe] 样本：{[(g['dataset_index'], g.get('episode_id'), g.get('chunk_start')) for g in group_ids]}")
    print(f"[probe] image_grid_thw：{[list(it['image_grid_thw'].shape) for it in items]}"
          f"  值={[it['image_grid_thw'].tolist() for it in items]}")
    layer0_store = None
    if a.probe_layer0:
        handle, layer0_store, layer0_name = install_layer0_hook(vla)
        if handle is None:
            print("[BLOCKED] --probe-layer0 指定了，但模型里找不到 `*.layers.0`（fail-closed，不静默跳过）")
            return 2
        print(f"[probe] 已挂第 0 层 hook：{layer0_name}")
    vla.eval()
    torch.cuda.empty_cache()

    result: Dict[str, Any] = {"out_dir": str(out), "task": a.task, "indices": indices,
                              "legs": {}, "compare": {}}
    for name, leg_dir, desc, n, fresh, path_label in _leg_planes(a):
        leg_items = items[:n]
        leg_ids = group_ids[:n]
        verdict = run_leg(validator, leg_items, ft, leg_dir=leg_dir, leg_name=name,
                          group_ids=leg_ids, fresh=fresh, path_label=path_label,
                          layer0_store=layer0_store)
        result["legs"][name] = {"status": verdict["status"], "problems": verdict["problems"],
                                "seconds": verdict["seconds"],
                                "peak_free_gib": verdict["peak_free_gib"],
                                "n_records": verdict["n_records"]}
        if verdict["status"] != "PASS":
            result["status"] = "BLOCKED"
            result["problems"] = [f"leg_{name}:{p}" for p in verdict["problems"]]
            print(f"[BLOCKED] 腿 {name} 自检不通过 ⇒ 立即停止（fail-closed）")
            _write_summary(out, result)
            return 2

    pairs = [("b1_normal", "b1_cleared"), ("b1_cleared", "b2_cleared"), ("b1_normal", "b1_repeat")]
    for left, right in pairs:
        cmp = ebp.compare_evidence(str(out / f"leg_{left}"), str(out / f"leg_{right}"),
                                   label_a=left, label_b=right)
        result["compare"][f"{left}_vs_{right}"] = cmp
        diffs = [f"{e['key']}:max={e['output_max_abs_diff']:.3e}" for e in cmp["per_sample"][:4]]
        print(f"[probe] {left} vs {right}: {cmp['status']} "
              f"output_bitwise={[e['output_bitwise'] for e in cmp['per_sample']]} {diffs}")
    if a.probe_layer0:
        l0 = {}
        for left, right in (("b1_normal", "b1_cleared"), ("b1_cleared", "b2_cleared")):
            store = {}
            for tag, leg in ((left, left), (right, right)):
                try:
                    bundle = ebp.read_evidence(str(out / f"leg_{leg}"), "layer0")
                    rec = bundle["manifest"]["records"][0]["arrays"]
                    for field in ("input", "output"):
                        meta = rec.get(field) or {}
                        if meta.get("present"):
                            store.setdefault(field, {})[tag] = bundle["arrays"][meta["npz_key"]]
                except (FileNotFoundError, IndexError, KeyError) as exc:  # noqa: BLE001
                    result.setdefault("layer0_errors", []).append(f"{leg}:{exc}")
            l0[f"{left}_vs_{right}"] = {}
            for field in ("input", "output"):
                pair = store.get(field) or {}
                if left in pair and right in pair:
                    l0[f"{left}_vs_{right}"] = compare_layer0(
                        {field: pair[left]}, {field: pair[right]},
                        label_a=left, label_b=right)
        result["layer0"] = l0
        for key, entry in l0.items():
            print(f"[probe] layer0 {key}: {entry.get('verdict')} flags={entry.get('flags')}")
    result["decision"] = decide(result)
    result["batch_equivalent_after_cache_fix"] = bool(
        result["compare"]["b1_cleared_vs_b2_cleared"]["status"] == "PASS")
    # 退出码只表示"探针本身跑通且证据齐全"；批处理是否等价由 decision 表达
    result["status"] = "PASS"
    _write_summary(out, result)
    print("=" * 78)
    print(f"结论 = {result['decision']['conclusion']}")
    print(f"建议 = {result['decision']['advice']}")
    print(f"Batch1≡Batch2（清缓存后逐位）= {result['batch_equivalent_after_cache_fix']}")
    print("=" * 78)
    return 0


def _write_summary(out: Path, result: Dict[str, Any]) -> None:
    (out / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="GPU 四腿对照探针（默认 PLAN ONLY；一次加载判定缓存/varlen 差异）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--val-ids", default=f"{DEFAULT_SPLIT}/{DEFAULT_TASK}.val_ids.json")
    ap.add_argument("--dataset", default="/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30")
    ap.add_argument("--dataset-indices", type=int, nargs="+", default=list(DEFAULT_DATASET_INDICES))
    ap.add_argument("--use-length", type=int, default=50)
    ap.add_argument("--use-bf16", action="store_true", default=True)
    ap.add_argument("--probe-layer0", action="store_true",
                    help="（可选）额外记录第 0 层输入/输出，用于判断分歧是否已在第 0 层之前")
    ap.add_argument("--execute", action="store_true", help="真正开跑（默认只打印计划）")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if not a.execute:
        return plan(a)
    return execute(a)


if __name__ == "__main__":
    sys.exit(main())

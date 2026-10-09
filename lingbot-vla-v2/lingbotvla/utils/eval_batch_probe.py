"""Eval Batch 诊断证据的共享实现（**纯 numpy/json**，可离线单测，不依赖 torch）。

背景（2026-10-09）：Batch2 与 serial 的动作误差**弥散**（14/14 维、多时间步，
max ≈ 0.011–0.058），而"噪声是否一致 / serial 自身是否稳定 / 首个差异在入参还是
`sample_actions` 内部"三问**都没有答案**。原因不是模型，而是**诊断证据本身不可判**：

1. 证据文件名带**毫秒时间戳**、内容里**没有样本身份** ⇒ 只能做"时间配对"，
   无法证明"这两条记录说的是同一个样本"；
2. Serial Repeat 钩子**静默失效**（`keys` 未定义 ⇒ `UnboundLocalError` 被
   `except Exception` 吞掉）⇒ 文件从未产出，却没有任何失败信号；
3. 没有任何机制检查"批处理真正喂给模型的入参"是否与串行逐位一致。

本模块把这三件事固化成**可复用、可判**的机制：

* :func:`make_identity` / :func:`identity_key` —— 每条证据都带**完整样本身份**
  （task / episode_id / chunk_start / dataset_index / inference_path /
  batch_position / repeat_index），**不用时间戳推断**；
* :class:`ProbeRecorder` —— 记录 `sample_actions` 的**实际入参**（逐样本规范化）
  与输出；
* :func:`write_evidence` / :func:`read_evidence` —— 三类证据（noise /
  serial_repeat / batch_actions）落盘 = `.npz`（数值）+ `.json`（清单）；
* :func:`detect_problems` —— **fail-closed 判定器**：文件缺失 / 钩子未执行 /
  样本 ID 不匹配 / 非有限值 / 应逐位一致的项不一致 ⇒ 明确的 problem code。

判定器不"猜"：它只比对**应当逐位一致**的量，并把每次比较的结论写进 checks。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1

#: 样本身份字段（顺序即落盘清单里的字段顺序）。**不允许缺项**，缺项写 `None` 也要显式出现。
IDENTITY_FIELDS: Tuple[str, ...] = (
    "task",
    "episode_id",
    "chunk_start",
    "dataset_index",
    "inference_path",
    "batch_position",
    "repeat_index",
)

#: 推理路径：serial=第一次串行；serial_repeat=同噪声的第二次串行；batch=批量。
INFERENCE_PATHS: Tuple[str, ...] = ("serial", "serial_repeat", "batch")

#: 应当逐位一致的模型入参字段（`image_grid_thw` 允许为 None）。
MODEL_INPUT_FIELDS: Tuple[str, ...] = (
    "images",
    "img_masks",
    "lang_tokens",
    "lang_masks",
    "state",
    "image_grid_thw",
)

#: 三类证据的固定文件名主干。
EVIDENCE_STEMS: Tuple[str, ...] = ("noise", "serial_repeat", "batch_actions")

#: 附加证据（不参与"三类必生成"的判定）：`layer0` = 第 0 层整层激活（作用域见 extra）。
EXTRA_STEMS: Tuple[str, ...] = ("layer0",)

_MANIFEST_KEY = "__manifest__"


# ---------------------------------------------------------------------------
# 样本身份
# ---------------------------------------------------------------------------
def make_identity(
    *,
    task: Optional[str],
    episode_id: Optional[int],
    chunk_start: Optional[int],
    dataset_index: int,
    inference_path: str,
    batch_position: int = 0,
    repeat_index: int = 0,
) -> Dict[str, Any]:
    """构造一条样本身份记录（全部字段显式，缺失用 ``None``，**不臆造**）。"""
    if inference_path not in INFERENCE_PATHS:
        raise ValueError(f"inference_path 必须是 {INFERENCE_PATHS}，收到 {inference_path!r}")
    for name, value in (("dataset_index", dataset_index),
                        ("batch_position", batch_position),
                        ("repeat_index", repeat_index)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} 必须是非负整数，收到 {value!r}")
    for name, value in (("episode_id", episode_id), ("chunk_start", chunk_start)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{name} 必须是整数或 None，收到 {value!r}")
    if task is not None and not isinstance(task, str):
        raise ValueError(f"task 必须是 str 或 None，收到 {task!r}")
    return {
        "task": task,
        "episode_id": episode_id,
        "chunk_start": chunk_start,
        "dataset_index": dataset_index,
        "inference_path": inference_path,
        "batch_position": batch_position,
        "repeat_index": repeat_index,
    }


def identity_key(identity: Mapping[str, Any]) -> str:
    """样本身份的稳定键（可读、可排序、不含时间戳）。

    ``ds12:click_bell:ep51:c0:batch:b1:r0``
    """
    missing = [f for f in IDENTITY_FIELDS if f not in identity]
    if missing:
        raise ValueError(f"样本身份缺少字段 {missing}: {identity!r}")
    task = identity["task"] if identity["task"] is not None else "task?"
    ep = identity["episode_id"] if identity["episode_id"] is not None else "ep?"
    chunk = identity["chunk_start"] if identity["chunk_start"] is not None else "c?"
    return (f"ds{int(identity['dataset_index'])}:{task}:ep{ep}:c{chunk}:"
            f"{identity['inference_path']}:b{int(identity['batch_position'])}:"
            f"r{int(identity['repeat_index'])}")


def base_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """去掉推理路径维度，只留"哪个样本"（用于跨路径配对）。

    同时接受**完整身份**与**基础身份**（只有 dataset_index/episode_id/chunk_start），
    后者用于 driver 声明的"本组样本顺序"。
    """
    return {
        "task": identity.get("task"),
        "episode_id": identity.get("episode_id"),
        "chunk_start": identity.get("chunk_start"),
        "dataset_index": int(identity["dataset_index"]),
    }


def base_key(identity: Mapping[str, Any]) -> str:
    """样本键（不含推理路径）：``ds12:click_bell:ep51:c0``。"""
    return identity_key({**dict(identity), "inference_path": "serial",
                         "batch_position": 0, "repeat_index": 0}).rsplit(":serial:", 1)[0]


# ---------------------------------------------------------------------------
# 数值工具（duck typing：torch.Tensor / numpy / list 都吃）
# ---------------------------------------------------------------------------
def to_numpy(value: Any):
    """把任意数组样对象转成 numpy（torch.Tensor 走 detach().cpu().numpy()）。

    ⚠️ **bfloat16 必须上转 float32**：numpy 没有 bf16 dtype，直接 `.numpy()` 会抛
    ``TypeError: Got unsupported ScalarType BFloat16``（2026-10-09 GPU 首跑实测踩到）。
    bf16 → fp32 是**无损**的，所以逐位比较语义不变；原始 dtype 另行记录在证据清单里。
    """
    import numpy as np

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        try:
            value = value.numpy()
        except TypeError:
            import torch as _torch

            value = value.to(_torch.float32).numpy()
    # 复制一份：CPU 张量的 `.numpy()` 是**共享内存的视图**，证据不能被后续复用改写
    return np.array(value, copy=True)


def source_dtype(value: Any) -> str:
    """记录**原始** dtype（bf16 会在 to_numpy 里被上转，别把上转后的 dtype 当原始值）。"""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return type(value).__name__
    return str(dtype)


def bitwise_identical(a: Any, b: Any) -> bool:
    """**逐位**相等：shape / dtype / 原始字节全部一致（NaN 也按字节比）。"""
    import numpy as np

    x, y = to_numpy(a), to_numpy(b)
    if x is None or y is None:
        return x is None and y is None
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    return x.tobytes() == y.tobytes()


def all_finite(value: Any) -> bool:
    """非数值 dtype 视为有限（掩码/索引不需要有限性）。"""
    import numpy as np

    arr = to_numpy(value)
    if arr is None:
        return True
    if arr.dtype.kind not in "fc":
        return True
    return bool(np.isfinite(arr).all())


def max_abs_diff(a: Any, b: Any) -> Optional[float]:
    """逐元素最大绝对差；不可比（shape 不同 / 非数值）⇒ ``None``。"""
    import numpy as np

    x, y = to_numpy(a), to_numpy(b)
    if x is None or y is None or x.shape != y.shape or x.dtype.kind not in "fc":
        return None
    return float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())


def mean_abs_diff(a: Any, b: Any) -> Optional[float]:
    """逐元素平均绝对差（**真实误差**，报告用；判 parity 用生产容差，不在这里放宽）。"""
    import numpy as np

    x, y = to_numpy(a), to_numpy(b)
    if x is None or y is None or x.shape != y.shape or x.dtype.kind not in "fc":
        return None
    return float(np.abs(x.astype(np.float64) - y.astype(np.float64)).mean())


def absmax(value: Any) -> Optional[float]:
    """参考量级（|action| 的绝对值最大），用来判断误差是否"量级显著"。"""
    import numpy as np

    arr = to_numpy(value)
    if arr is None or arr.dtype.kind not in "fc" or not arr.size:
        return None
    return float(np.max(np.abs(arr)))


def array_stats(value: Any) -> Dict[str, Any]:
    """落盘清单里记录的数组摘要（形状/dtype/有限性/极值）。"""
    import numpy as np

    arr = to_numpy(value)
    if arr is None:
        return {"shape": None, "dtype": None, "finite": True}
    out: Dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                           "finite": all_finite(arr)}
    if arr.dtype.kind in "fc" and arr.size:
        out["min"] = float(np.min(arr))
        out["max"] = float(np.max(arr))
        out["absmax"] = float(np.max(np.abs(arr)))
    return out


def _sanitize(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in text)


# ---------------------------------------------------------------------------
# 记录器
# ---------------------------------------------------------------------------
class ProbeRecorder:
    """收集 `sample_actions` 的**实际入参**与输出，按样本身份索引。

    用法（生产侧在 `_infer_one` / `_infer_batch` 的调用点）：

        rec.add(identity=..., inputs={'images': images[0], ...}, noise=noise[0],
                output=actions[0], raw_shapes={'images': tuple(images.shape), ...})

    传入的 ``inputs`` / ``noise`` / ``output`` 必须是**该样本自己的**数组
    （批处理路径要按 batch 内位置切出来），这样跨路径才能逐位对拍。
    """

    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self.errors: List[str] = []

    def add(self, *, identity: Mapping[str, Any], inputs: Mapping[str, Any],
            noise: Any, output: Any,
            raw_shapes: Optional[Mapping[str, Any]] = None) -> str:
        key = identity_key(identity)
        if key in self.records:
            self.errors.append(f"duplicate_sample_record:{key}")
            return key
        unknown = [k for k in inputs if k not in MODEL_INPUT_FIELDS]
        if unknown:
            self.errors.append(f"unknown_input_field:{','.join(sorted(unknown))}")
        self.records[key] = {
            "identity": {f: identity[f] for f in IDENTITY_FIELDS},
            "key": key,
            "inputs": {k: to_numpy(v) for k, v in inputs.items()},
            "noise": to_numpy(noise),
            "output": to_numpy(output),
            "raw_shapes": {k: list(v) for k, v in (raw_shapes or {}).items()},
            # 原始 dtype（bf16 在 to_numpy 里会被无损上转成 fp32，别丢了这条信息）
            "source_dtypes": {**{f"input.{k}": source_dtype(v) for k, v in inputs.items()},
                              "noise": source_dtype(noise), "output": source_dtype(output)},
        }
        return key

    def __len__(self) -> int:
        return len(self.records)

    def get(self, identity: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        return self.records.get(identity_key(identity))

    def by_path(self, path: str) -> List[Dict[str, Any]]:
        out = [r for r in self.records.values() if r["identity"]["inference_path"] == path]
        return sorted(out, key=lambda r: int(r["identity"]["batch_position"]))

    def lookup(self, *, task, episode_id, chunk_start, dataset_index,
               inference_path, batch_position=0, repeat_index=0) -> Optional[Dict[str, Any]]:
        return self.records.get(identity_key(make_identity(
            task=task, episode_id=episode_id, chunk_start=chunk_start,
            dataset_index=dataset_index, inference_path=inference_path,
            batch_position=batch_position, repeat_index=repeat_index)))


# ---------------------------------------------------------------------------
# 落盘 / 读回
# ---------------------------------------------------------------------------
def write_evidence(out_dir: str, stem: str, records: Sequence[Mapping[str, Any]],
                   *, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """写一份证据：``<stem>.npz``（数值，键带样本身份）+ ``<stem>.json``（清单）。

    ``records`` 的每一项形如 ``{"identity": {...}, "arrays": {name: array}}``；
    也接受 :class:`ProbeRecorder` 的原生记录（``inputs``/``noise``/``output``）。
    """
    import numpy as np

    if stem not in EVIDENCE_STEMS + EXTRA_STEMS:
        raise ValueError(f"未知证据类型 {stem!r}，允许 {EVIDENCE_STEMS + EXTRA_STEMS}")
    os.makedirs(out_dir, exist_ok=True)
    payload: Dict[str, Any] = {}
    manifest: List[Dict[str, Any]] = []
    for record in records:
        identity = dict(record["identity"])
        key = identity_key(identity)
        arrays = record.get("arrays")
        if arrays is None:  # ProbeRecorder 原生记录
            arrays = {**record.get("inputs", {}), "noise": record.get("noise"),
                      "output": record.get("output")}
        entry = {"identity": {f: identity[f] for f in IDENTITY_FIELDS}, "key": key,
                 "arrays": {}, "raw_shapes": dict(record.get("raw_shapes", {})),
                 "source_dtypes": dict(record.get("source_dtypes", {}))}
        for name, value in arrays.items():
            arr = to_numpy(value)
            if arr is None:
                entry["arrays"][name] = {"present": False}
                continue
            npz_key = f"{_sanitize(key)}__{_sanitize(name)}"
            payload[npz_key] = arr
            entry["arrays"][name] = {"present": True, "npz_key": npz_key,
                                     **array_stats(arr)}
        manifest.append(entry)
    manifest_doc = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"eval_batch_probe_{stem}",
        "stem": stem,
        "count": len(manifest),
        "identity_fields": list(IDENTITY_FIELDS),
        "records": manifest,
    }
    if extra:
        manifest_doc.update(dict(extra))
    # 清单也内嵌进 npz（0-d 字符串数组）⇒ 单个文件即可自证身份，不靠外部约定。
    payload[_MANIFEST_KEY] = np.asarray(json.dumps(manifest_doc, ensure_ascii=False, sort_keys=True))
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    json_path = os.path.join(out_dir, f"{stem}.json")
    _atomic_savez(npz_path, payload)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_doc, fh, ensure_ascii=False, sort_keys=True, indent=1)
    return {"npz": npz_path, "json": json_path,
            "sha256_npz": _sha256(npz_path), "sha256_json": _sha256(json_path)}


def _atomic_savez(path: str, payload: Mapping[str, Any]) -> None:
    import numpy as np

    tmp = f"{path}.tmp.npz"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, path)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_evidence(out_dir: str, stem: str) -> Dict[str, Any]:
    """读回一份证据，返回 ``{manifest, arrays, json, npz}``；缺失 ⇒ 抛错（不静默）。"""
    import numpy as np

    json_path = os.path.join(out_dir, f"{stem}.json")
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    for path in (json_path, npz_path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f"证据文件缺失或为空: {path}")
    with open(json_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with np.load(npz_path, allow_pickle=False) as bundle:
        arrays = {k: bundle[k] for k in bundle.files}
    return {"manifest": manifest, "arrays": arrays, "json": json_path, "npz": npz_path}


def evidence_files(out_dir: str) -> List[str]:
    names: List[str] = []
    for stem in EVIDENCE_STEMS:
        names.extend([f"{stem}.npz", f"{stem}.json"])
    return [os.path.join(out_dir, n) for n in names]


# ---------------------------------------------------------------------------
# 判定器（fail-closed）
# ---------------------------------------------------------------------------
def _problem_file_checks(out_dir: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    problems: List[str] = []
    checks: List[Dict[str, Any]] = []
    for stem in EVIDENCE_STEMS:
        for ext in ("npz", "json"):
            path = os.path.join(out_dir, f"{stem}.{ext}")
            ok = os.path.exists(path) and os.path.getsize(path) > 0
            checks.append({"check": f"evidence_file:{stem}.{ext}", "ok": bool(ok)})
            if not ok:
                problems.append(f"missing_or_empty:{stem}.{ext}")
    return problems, checks


def _matches_other_position(expected_list, position, lookup, field, value):
    """批内第 ``position`` 条的 ``field`` 是否**其实是别的样本**的（顺序/对应错位）。

    返回匹配到的位置下标；没有任何其它位置逐位相同 ⇒ ``None``（那就是普通的不一致）。
    """
    if value is None:
        return None
    for other_pos in range(len(expected_list)):
        if other_pos == position:
            continue
        record = lookup(expected_list[other_pos])
        if record is None:
            continue
        if field == "noise":
            other_value = record.get("noise")
        else:
            other_value = record.get("inputs", {}).get(field)
        if other_value is not None and bitwise_identical(other_value, value):
            return other_pos
    return None


def detect_problems(recorder: ProbeRecorder, expected: Sequence[Mapping[str, Any]],
                    out_dir: Optional[str] = None, *,
                    require_repeat: bool = True,
                    require_evidence_files: bool = True) -> Dict[str, Any]:
    """对一次 probe 的录制结果做 fail-closed 判定。

    ``expected`` = 该组样本的**批内顺序**身份列表（含 task/episode_id/chunk_start/
    dataset_index；推理路径维度由本函数补全）。

    返回 ``{'status', 'problems', 'checks', 'per_sample'}``：

    * ``PASS``      —— 无 problem（所有应当逐位一致项都逐位一致）；
    * ``BLOCKED``   —— 有 problem（文件缺失 / 钩子未执行 / 样本 ID 不匹配 /
      非有限值 / 数值不一致）。
    """
    problems: List[str] = list(recorder.errors)
    checks: List[Dict[str, Any]] = []
    per_sample: List[Dict[str, Any]] = []

    if require_evidence_files and out_dir is not None:
        file_problems, file_checks = _problem_file_checks(out_dir)
        problems.extend(file_problems)
        checks.extend(file_checks)

    expected_list = [base_identity(e) for e in expected]
    expected_keys = [base_key(e) for e in expected_list]
    if len(set(expected_keys)) != len(expected_keys):
        problems.append(f"duplicate_expected_sample:{expected_keys}")

    total_expected = len(expected_list) * (3 if require_repeat else 2)
    if len(recorder) < total_expected:
        problems.append(f"hooks_not_executed:records={len(recorder)}<expected={total_expected}")
    checks.append({"check": "hooks_executed", "ok": bool(len(recorder) >= total_expected),
                   "records": len(recorder), "expected": total_expected})

    def _rec(base, path, batch_position=0, repeat_index=0):
        return recorder.lookup(task=base["task"], episode_id=base["episode_id"],
                               chunk_start=base["chunk_start"],
                               dataset_index=base["dataset_index"],
                               inference_path=path, batch_position=batch_position,
                               repeat_index=repeat_index)

    # 批内位置索引：**按位置**取记录（而不是按身份），这样"位置声明的身份与期望不符"
    # 才可能被发现（按身份查表的话，键就是声明本身 ⇒ 自证循环）。
    batch_by_position = {int(r["identity"]["batch_position"]): r
                         for r in recorder.by_path("batch")}

    for position, base in enumerate(expected_list):
        item: Dict[str, Any] = {"dataset_index": base["dataset_index"], "position": position,
                                "task": base["task"], "episode_id": base["episode_id"],
                                "chunk_start": base["chunk_start"]}
        serial = _rec(base, "serial")
        batched = batch_by_position.get(position)
        repeat = _rec(base, "serial_repeat", repeat_index=1) if require_repeat else None

        if serial is None:
            problems.append(f"missing_record:serial:{base_key(base)}")
        if batched is None:
            problems.append(f"missing_record:batch:pos{position}:{base_key(base)}")
        if require_repeat and repeat is None:
            problems.append(f"missing_record:serial_repeat:{base_key(base)}")
        # 批内位置声明的身份必须就是期望的身份（防"位置串了但内容看着一样"）。
        if batched is not None:
            declared = base_key(batched["identity"])
            ok = declared == base_key(base)
            checks.append({"check": f"sample_id_at_position:{position}", "ok": bool(ok),
                           "declared": declared, "expected": base_key(base)})
            if not ok:
                problems.append(f"sample_id_mismatch:pos{position}:{declared}!={base_key(base)}")
        if serial is None or batched is None:
            item["comparable"] = False
            per_sample.append(item)
            continue
        item["comparable"] = True

        # ① 噪声：批内第 i 条必须与串行第 i 条逐位一致；不一致时进一步判断是否"对到了别的样本"
        noise_ok = bitwise_identical(serial["noise"], batched["noise"])
        checks.append({"check": f"noise_bitwise:pos{position}", "ok": bool(noise_ok),
                       "max_abs_diff": max_abs_diff(serial["noise"], batched["noise"])})
        item["noise_bitwise"] = bool(noise_ok)
        if not noise_ok:
            swapped_to = _matches_other_position(
                expected_list, position, lambda other: _rec(other, "serial"), "noise",
                batched["noise"])
            if swapped_to is not None:
                problems.append(f"sample_correspondence_mismatch:noise:pos{position}->pos{swapped_to}")
                item["noise_matches_position"] = swapped_to
            else:
                problems.append(f"batch_noise_not_bitwise:pos{position}:{base_key(base)}")

        # ② 实际入参：应当逐位一致的字段逐个比；不一致时先判断"是不是对到了别的样本"
        for field in MODEL_INPUT_FIELDS:
            if field not in serial["inputs"] and field not in batched["inputs"]:
                continue
            a = serial["inputs"].get(field)
            b = batched["inputs"].get(field)
            same = bitwise_identical(a, b)
            checks.append({"check": f"input_bitwise:{field}:pos{position}", "ok": bool(same),
                           "serial_shape": None if a is None else list(to_numpy(a).shape),
                           "batch_shape": None if b is None else list(to_numpy(b).shape),
                           "max_abs_diff": None if same else max_abs_diff(a, b)})
            if not same:
                swapped_to = _matches_other_position(
                    expected_list, position, lambda other: _rec(other, "serial"), field, b)
                if swapped_to is not None:
                    problems.append(
                        f"sample_correspondence_mismatch:{field}:pos{position}->pos{swapped_to}")
                    item.setdefault("input_matches_position", {})[field] = swapped_to
                else:
                    problems.append(f"batch_input_not_bitwise:{field}:pos{position}")
            item.setdefault("inputs_bitwise", {})[field] = bool(same)

        # ③ 输出：同入参同噪声 ⇒ 必须逐位一致
        out_ok = bitwise_identical(serial["output"], batched["output"])
        checks.append({"check": f"output_bitwise:pos{position}", "ok": bool(out_ok),
                       "max_abs_diff": max_abs_diff(serial["output"], batched["output"])})
        item["output_bitwise"] = bool(out_ok)
        if not out_ok:
            problems.append(f"batch_output_not_bitwise:pos{position}:"
                            f"max_abs_diff={max_abs_diff(serial['output'], batched['output'])}")

        # ④ Serial Repeat（同进程 / 同模型状态 / 显式同噪声）
        if repeat is not None:
            rep_noise = bitwise_identical(serial["noise"], repeat["noise"])
            if not rep_noise:
                problems.append(f"repeat_noise_not_bitwise:pos{position}")
            rep_out = bitwise_identical(serial["output"], repeat["output"])
            checks.append({"check": f"repeat_output_bitwise:pos{position}", "ok": bool(rep_out),
                           "max_abs_diff": max_abs_diff(serial["output"], repeat["output"])})
            item["repeat_noise_bitwise"] = bool(rep_noise)
            item["repeat_output_bitwise"] = bool(rep_out)
            if not rep_out:
                problems.append(f"serial_repeat_not_bitwise:pos{position}:"
                                f"max_abs_diff={max_abs_diff(serial['output'], repeat['output'])}")

        # ⑤ 有限性
        for label, value in (("noise", serial["noise"]), ("output", serial["output"]),
                             ("batch_noise", batched["noise"]), ("batch_output", batched["output"])):
            if not all_finite(value):
                problems.append(f"nonfinite:{label}:pos{position}")
        for field, value in serial["inputs"].items():
            if not all_finite(value):
                problems.append(f"nonfinite:input:{field}:pos{position}")
        per_sample.append(item)

    # 录制里出现"期望之外"的样本 ⇒ 也要报（防张冠李戴）
    expected_record_keys = set()
    for position, base in enumerate(expected_list):
        expected_record_keys.add(identity_key({**base, "inference_path": "serial",
                                               "batch_position": 0, "repeat_index": 0}))
        expected_record_keys.add(identity_key({**base, "inference_path": "batch",
                                               "batch_position": position, "repeat_index": 0}))
        if require_repeat:
            expected_record_keys.add(identity_key({**base, "inference_path": "serial_repeat",
                                                   "batch_position": 0, "repeat_index": 1}))
    unexpected = sorted(set(recorder.records) - expected_record_keys)
    if unexpected:
        problems.append(f"unexpected_records:{unexpected}")
    checks.append({"check": "no_unexpected_records", "ok": bool(not unexpected),
                   "unexpected": unexpected})

    problems = sorted(set(problems))
    return {"status": "BLOCKED" if problems else "PASS",
            "problems": problems, "checks": checks, "per_sample": per_sample,
            "n_records": len(recorder),
            "expected_keys": expected_keys}


def write_group_evidence(recorder: ProbeRecorder, out_dir: str,
                         expected: Sequence[Mapping[str, Any]],
                         *, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """把一次 probe 组的录制结果写成三类证据文件，返回文件名/sha256 清单。"""
    grouped = recorder_to_evidence(recorder)
    files: Dict[str, Any] = {}
    for stem, records in grouped.items():
        files[stem] = write_evidence(out_dir, stem, records, extra=extra)
    return files


# ---------------------------------------------------------------------------
# 单腿自检 / 跨腿对拍（GPU 四腿对照用；纯函数，可离线单测）
# ---------------------------------------------------------------------------
def detect_leg_problems(recorder: ProbeRecorder, expected: Sequence[Mapping[str, Any]],
                        out_dir: Optional[str] = None, *,
                        require_evidence_files: bool = True) -> Dict[str, Any]:
    """**单腿**自检：文件齐 / 钩子执行 / 身份对得上 / 数值有限。

    与 :func:`detect_problems` 的区别：本函数**不做**"串行 vs 批量"的组内比较
    （一条腿可能只有串行或只有批量记录），跨腿比较交给 :func:`compare_evidence`。
    """
    problems: List[str] = list(recorder.errors)
    checks: List[Dict[str, Any]] = []
    if require_evidence_files and out_dir is not None:
        file_problems, file_checks = _problem_file_checks(out_dir)
        problems.extend(file_problems)
        checks.extend(file_checks)

    expected_list = [base_identity(e) for e in expected]
    expected_keys = [base_key(e) for e in expected_list]
    if len(set(expected_keys)) != len(expected_keys):
        problems.append(f"duplicate_expected_sample:{expected_keys}")

    batch_by_position = {int(r["identity"]["batch_position"]): r
                         for r in recorder.by_path("batch")}
    # ⚠️ 逐样本路径**不止** "serial" —— `serial_repeat` 也是逐样本前向。
    #    2026-10-09 GPU 实测：只认 "serial" ⇒ repeat 腿被误判 missing_record + BLOCKED。
    serial_by_key = {base_key(r["identity"]): r for r in recorder.records.values()
                     if r["identity"]["inference_path"] != "batch"}
    for position, base in enumerate(expected_list):
        key = base_key(base)
        record = batch_by_position.get(position) if recorder.by_path("batch") else serial_by_key.get(key)
        if record is None:
            problems.append(f"missing_record:pos{position}:{key}")
            continue
        declared = base_key(record["identity"])
        if declared != key:
            problems.append(f"sample_id_mismatch:pos{position}:{declared}!={key}")
        for label, value in (("noise", record["noise"]), ("output", record["output"])):
            if not all_finite(value):
                problems.append(f"nonfinite:{label}:pos{position}")
        for field, value in record["inputs"].items():
            if not all_finite(value):
                problems.append(f"nonfinite:input:{field}:pos{position}")
    n_expected = len(expected_list)
    if len(recorder) < n_expected:
        problems.append(f"hooks_not_executed:records={len(recorder)}<expected={n_expected}")
    checks.append({"check": "hooks_executed", "ok": bool(len(recorder) >= n_expected),
                   "records": len(recorder), "expected": n_expected})
    problems = sorted(set(problems))
    return {"status": "BLOCKED" if problems else "PASS", "problems": problems,
            "checks": checks, "n_records": len(recorder), "expected_keys": expected_keys}


def load_leg(dir_path: str) -> Dict[str, Dict[str, Any]]:
    """读回一条腿的三类证据，按**样本键**（不含推理路径）归并。

    返回 ``{base_key: {'identity', 'inputs', 'noise', 'output', 'raw_shapes'}}``。
    同一腿里同一 key 的多条记录（串行/批量）会合并（后到的不覆盖已有值）。
    """
    leg: Dict[str, Dict[str, Any]] = {}
    for stem in EVIDENCE_STEMS:
        try:
            bundle = read_evidence(dir_path, stem)
        except FileNotFoundError:
            continue
        for record in bundle["manifest"]["records"]:
            key = base_key(record["identity"])
            entry = leg.setdefault(key, {"identity": dict(record["identity"]),
                                         "inputs": {}, "noise": None, "output": None,
                                         "raw_shapes": dict(record.get("raw_shapes") or {})})
            for name, meta in record["arrays"].items():
                if not meta.get("present"):
                    continue
                value = bundle["arrays"][meta["npz_key"]]
                if name == "noise" and entry["noise"] is None:
                    entry["noise"] = value
                elif name == "output" and entry["output"] is None:
                    entry["output"] = value
                elif name not in ("noise", "output"):
                    entry["inputs"].setdefault(name, value)
    return leg


def compare_evidence(dir_a: str, dir_b: str, *,
                     label_a: str = "A", label_b: str = "B") -> Dict[str, Any]:
    """**跨腿**逐位对拍：噪声 / 实际入参（逐字段）/ 输出，并保留**真实误差**。

    比对按样本身份配对（不是按顺序、不是按时间戳）。任何身份集合不一致 ⇒ BLOCKED。
    """
    leg_a, leg_b = load_leg(dir_a), load_leg(dir_b)
    problems: List[str] = []
    if not leg_a or not leg_b:
        problems.append(f"empty_leg:{label_a}={len(leg_a)}:{label_b}={len(leg_b)}")
    if set(leg_a) != set(leg_b):
        problems.append(f"sample_set_mismatch:{sorted(set(leg_a) ^ set(leg_b))}")
    per_sample: List[Dict[str, Any]] = []
    for key in sorted(set(leg_a) & set(leg_b)):
        ra, rb = leg_a[key], leg_b[key]
        entry: Dict[str, Any] = {
            "key": key, "dataset_index": ra["identity"]["dataset_index"],
            "task": ra["identity"].get("task"), "chunk_start": ra["identity"].get("chunk_start"),
            "episode_id": ra["identity"].get("episode_id"),
        }
        pairs = [("noise", ra["noise"], rb["noise"])]
        for field in MODEL_INPUT_FIELDS:
            pairs.append((field, ra["inputs"].get(field), rb["inputs"].get(field)))
        for name, va, vb in pairs:
            if va is None and vb is None:
                continue
            same = bitwise_identical(va, vb)
            entry[f"{name}_bitwise"] = bool(same)
            if not same:
                entry[f"{name}_max_abs_diff"] = max_abs_diff(va, vb)
                if name != "noise":
                    problems.append(f"{name}_not_bitwise:{key}")
                else:
                    problems.append(f"noise_not_bitwise:{key}")
        entry["output_bitwise"] = bool(bitwise_identical(ra["output"], rb["output"]))
        entry["output_max_abs_diff"] = max_abs_diff(ra["output"], rb["output"])
        entry["output_mean_abs_diff"] = mean_abs_diff(ra["output"], rb["output"])
        entry["output_ref_absmax"] = absmax(ra["output"])
        if ra["output"] is None or rb["output"] is None:
            problems.append(f"missing_output:{key}")
        elif not entry["output_bitwise"]:
            # 🔴 跨腿判定必须把**输出**差异算进去（"Batch1 与 Batch2 还差吗"就靠这一条）
            problems.append(f"output_not_bitwise:{key}:max={entry['output_max_abs_diff']}")
        per_sample.append(entry)
    problems = sorted(set(problems))
    return {"status": "BLOCKED" if problems else "PASS", "label_a": label_a, "label_b": label_b,
            "problems": problems, "per_sample": per_sample,
            "n_a": len(leg_a), "n_b": len(leg_b)}


def write_verdict(out_dir: str, verdict: Mapping[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "verdict.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(verdict), fh, ensure_ascii=False, sort_keys=True, indent=1)
    return path


def recorder_to_evidence(recorder: ProbeRecorder) -> Dict[str, List[Dict[str, Any]]]:
    """按三类证据拆出可落盘的记录列表（每类职责单一）。

    * ``noise``          —— 每条 (样本 × 推理路径) 的**实际噪声**；
    * ``batch_actions``  —— 每条 (样本 × 推理路径) 的**实际入参 + 输出**；
    * ``serial_repeat``  —— 两次串行（同进程/同模型状态/显式同噪声）的噪声 + 输出。
    """
    out: Dict[str, List[Dict[str, Any]]] = {stem: [] for stem in EVIDENCE_STEMS}
    for record in recorder.records.values():
        path = record["identity"]["inference_path"]
        if path == "serial_repeat":
            out["serial_repeat"].append({
                "identity": record["identity"],
                "arrays": {"noise": record["noise"], "output": record["output"]},
                "raw_shapes": record.get("raw_shapes", {}),
                "source_dtypes": record.get("source_dtypes", {}),
            })
            # 实际入参也一并落盘：跨腿对拍要比 images/state 等字段
            arrays_r = dict(record["inputs"])
            arrays_r["output"] = record["output"]
            out["batch_actions"].append({
                "identity": record["identity"], "arrays": arrays_r,
                "raw_shapes": record.get("raw_shapes", {}),
                "source_dtypes": record.get("source_dtypes", {}),
            })
            continue
        out["noise"].append({
            "identity": record["identity"],
            "arrays": {"noise": record["noise"]},
            "raw_shapes": record.get("raw_shapes", {}),
            "source_dtypes": record.get("source_dtypes", {}),
        })
        arrays = dict(record["inputs"])
        arrays["output"] = record["output"]
        out["batch_actions"].append({
            "identity": record["identity"], "arrays": arrays,
            "raw_shapes": record.get("raw_shapes", {}),
            "source_dtypes": record.get("source_dtypes", {}),
        })
    for items in out.values():
        items.sort(key=lambda r: (r["identity"]["dataset_index"],
                                  r["identity"]["batch_position"],
                                  r["identity"]["repeat_index"]))
    return out


__all__ = [
    "IDENTITY_FIELDS", "INFERENCE_PATHS", "MODEL_INPUT_FIELDS", "EVIDENCE_STEMS",
    "EXTRA_STEMS",
    "SCHEMA_VERSION", "ProbeRecorder", "all_finite", "array_stats", "base_identity",
    "base_key", "bitwise_identical", "detect_problems", "evidence_files",
    "identity_key", "make_identity", "max_abs_diff", "read_evidence",
    "recorder_to_evidence", "to_numpy", "write_evidence", "write_group_evidence",
    "absmax", "mean_abs_diff", "detect_leg_problems", "load_leg", "compare_evidence",
    "source_dtype",
    "write_verdict",
]

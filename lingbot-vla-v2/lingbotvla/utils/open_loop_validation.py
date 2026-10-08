"""训练中原地 open-loop validation（in-process：不落盘、不复制模型）。

设计约束
--------
* **复用当前训练中的模型权重**：不重新加载 ckpt、不额外实例化 6B inference policy
* 走 ``model.sample_actions`` **推理路径**，**不走训练 ``.forward()`` loss 路径**
* ``model.eval()`` + ``torch.inference_mode()``；结束后**逐模块恢复** training 标志
* **整次 eval 固定 seed**，且**前后快照/恢复 RNG（含 CUDA）**
  ⇒ 既保证不同 step 的评测可比，也**不改变训练后续步的随机流**
* 数据侧复用 ``build_vla_dataset``（与训练同一份 ``dataset_config``，只换 ``episode_ids_file``）
  ⇒ 预处理必然与训练一致
* **不触发 checkpoint 保存**（正式存档仍由 ``save_steps`` 单独控制）
* 第一版：稳定简单 —— 无异步 / 无多进程 / 无额外模型实例

口径一致性（重要）
------------------
``FeatureTransform.apply(item, policy_eval)`` 的差异只有三处：

1. ``action_is_pad``：训练用真实 pad mask，eval 用全 0 —— 与指标无关
2. **图像增强**：``train = image_augment and not policy_eval``
   ⇒ 当 ``image_augment=False`` 时（本项目配置），训练 apply 与 eval apply 的**观测完全等价**
3. ``policy_eval=True`` 会**跳过 action 相关处理** ⇒ **GT 动作必须走 ``policy_eval=False``**

所以直接用 dataset item（训练 apply）即可同时拿到「与官方等价的观测」和「可用的 GT」，
**前提是 ``image_augment=False``**；本模块在 ``image_augment=True`` 时打警告。

已知限制
--------
* 🔴 **只支持单卡**：``global_rank == 0`` 跑 eval，其余 rank 直接进入下一步 ⇒ 多卡下
  FSDP2 的 all-gather 会**死锁**，且 rank0 上参数是分片的、结果无效。
  ``world_size > 1`` 时**构造即 fail-fast**（见 ``_world_size``）。上多卡前必须改成
  「rank0 跑 + 其他 rank barrier + 参数 all-gather」。
* ``sample_actions`` 的调用 glue 是**照抄** ``deploy/lingbot_vla_v2_policy.py`` 的
  ``PolicyPreprocessMixin.sample_actions_batch``（无法真正复用：那个 mixin 依赖一个
  inference policy 实例，构造它就要分配 25.5G，违背"不复制模型"）。
  **改动时务必与 deploy 逐行对齐**，靠"与官方 open_loop_eval.py 的一致性校验"兜底。

评测期间会**临时改**、并在 ``finally`` 里恢复的东西（一个都不能漏）
------------------------------------------------------------------
1. ``model.config.use_cache`` → ``True``
   🔴 训练配置里 ``use_cache=False``（``configuration_lingbot_vla.py:104``，V2 不覆盖），
   而 ``sample_actions`` 传 ``fill_kv_cache=True, use_cache=self.config.use_cache``
   ⇒ ``handle_kv_cache`` 直接跳过 ⇒ ``past_key_values`` 恒为 ``None`` ⇒ **去噪阶段完全
   丢掉 VLM 前缀条件**（不报错、结果错）。2026-10-04 CPU 实测确认。
   deploy 路径 ``lingbot_vla_v2_policy.py:310`` 显式 ``config.use_cache = True``，我们补上。
2. Qwen3-VL 视觉塔的 5 个预计算网格缓存（``precompute_grid_thw``）
3. 每个 dataset 自己的 ``feature_transform.image_augment``
4. 逐模块 ``training`` 标志 / ``_use_compile_predict_velocity``
5. torch（CPU+CUDA）+ Python ``random`` + NumPy 的 RNG
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import random
import time
import traceback
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from lingbotvla.utils.eval_precision import METRIC_DTYPE_LABEL, metric_arrays_for_aggregation
import torch

from lingbotvla.data.dataset import build_vla_dataset

EVAL_SEED = 1234
# 与 tools/task_split.py 的 quantile 划分一致（click_bell）：5 条 train-monitor + 10 条 val
DEFAULT_TRAIN_MONITOR_IDS: List[int] = [50, 63, 76, 87, 99]
DEFAULT_VAL_IDS: List[int] = [51, 52, 56, 66, 73, 75, 78, 84, 94, 97]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _rng_snapshot() -> Dict[str, Any]:
    """快照全部随机源。

    ⚠️ 只存 torch 是不够的（2026-10-04 审查 H2）：``MultiVLADataset.__getitem__`` 的
    重试路径用 ``np.random.randint``，图像增强参数采样可能用 ``random``/numpy
    ⇒ eval 会推进这两条流、破坏「不同 step 之间可比」。
    """
    snap = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state_all()
    return snap


def _rng_restore(snap: Dict[str, Any]) -> None:
    torch.set_rng_state(snap["torch_cpu"])
    if "torch_cuda" in snap:
        torch.cuda.set_rng_state_all(snap["torch_cuda"])
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])


def _world_size() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:  # noqa: BLE001
        pass
    return 1


def _module_training_flags(model: torch.nn.Module) -> List[tuple]:
    """快照每个子模块的 training 标志。

    ⚠️ 不能事后一律 ``model.train()`` —— 那会把**有意冻结**的子模块（例如
    ``freeze_vision_encoder=true`` 时的 ``qwenvl.visual``）也放回训练模式，
    破坏冻结语义。必须逐一恢复。
    """
    return [(m, m.training) for m in model.modules()]


def _module_training_restore(flags: List[tuple]) -> None:
    for module, was_training in flags:
        module.training = was_training


def _load_episode_ids(path: str) -> List[int]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是裸列表（episode_ids 白名单格式）")
    return [int(x) for x in data]


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _replace_attr(obj, **kwargs):
    """`dataclasses.replace` 的非 dataclass 版本（浅拷贝 + setattr，不改原对象）。"""
    import copy as _copy

    new = _copy.copy(obj)
    for k, v in kwargs.items():
        setattr(new, k, v)
    return new


def _find_feature_transform(ds) -> Any:
    """从 ``build_vla_dataset`` 的返回值里取出 ``feature_transform``。

    ``data_name == 'multi'`` 时返回的是 **``MultiVLADataset``** —— 它自己**没有**
    ``feature_transform``，而是持有 ``feature_transforms: {data_name: ft}`` 与
    ``_datasets: [VLADataset]``；单数据集时返回 ``VLADataset``（直接有
    ``feature_transform``）。2026-10-04 smoke 实测踩到（报
    ``RuntimeError: 数据集上没有 feature_transform``）。
    """
    ft = getattr(ds, "feature_transform", None)
    if ft is not None:
        return ft
    fts = getattr(ds, "feature_transforms", None)
    if isinstance(fts, dict) and fts:
        return next(iter(fts.values()))
    inner = getattr(ds, "_datasets", None)
    if isinstance(inner, (list, tuple)) and inner:
        return getattr(inner[0], "feature_transform", None)
    return None


def _as_frames(arr: np.ndarray, key: str) -> np.ndarray:
    """把动作数组规范成 ``(帧, 维)``。

    ⚠️ 一维数组的含义是「**1 帧 × D 维**」而不是「N 帧 × 1 维」：旧实现用
    ``reshape(-1, 1)`` 会把**时间轴当成维度轴**（2026-10-04 审查 B3）。
    """
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


def _episode_index_map(ds) -> Optional[np.ndarray]:
    """返回「过滤后 local_idx → episode_index」映射；取不到返回 ``None``。

    指标必须**按完整 trajectory 聚合**（官方口径），所以需要知道每个 chunk 属于哪个回合。
    ``MultiVLADataset`` 单条目时下钻到 ``_datasets[0].dataset``（仓库的 LeRobotDataset
    子类），其 ``hf_dataset['episode_index']`` 就是过滤后逐行的回合号。
    """
    node = ds
    inner = getattr(ds, "_datasets", None)
    if isinstance(inner, (list, tuple)) and len(inner) == 1:
        node = inner[0]
    node = getattr(node, "dataset", node)
    hf = getattr(node, "hf_dataset", None)
    if hf is None:
        return None
    try:
        col = hf["episode_index"]
        arr = np.asarray([int(x) for x in col], dtype=np.int64)
    except Exception:  # noqa: BLE001
        return None
    if len(arr) != len(ds):
        return None
    return arr


def per_episode_starts(ep_map: np.ndarray, stride: int) -> List[int]:
    """按**回合各自从首帧**跳步，返回过滤后 local_idx 的起点列表。

    与官方 ``scripts/open_loop_eval.py`` 的
    ``for data_id in range(start_id, end_id, action_horizon)`` **等价**
    （``start_id/end_id`` 是该回合的 ``dataset_from_index/to_index``）。

    ⚠️ 不要改回「在整个（拼接后的）数据集上 ``range(0, len(ds), stride)``」：
    那会让 chunk 起点跨回合边界 —— 实测 click_bell val 上，**16 个起点里只有 1 个
    落在回合首帧**，且与官方只有 2 个起点重合（800 帧 vs 1000 帧）⇒ 根本不是同一个评测集。

    ``ep_map`` 必须是**按 local_idx 递增**的回合号数组（``_episode_index_map`` 的返回值）。
    独立成函数是为了让 ``tools/open_loop_parity_check.py`` 能直接单测它（无需模型）。
    """
    starts: List[int] = []
    i = 0
    n = int(len(ep_map))
    while i < n:
        ep = ep_map[i]
        j = i
        while j < n and ep_map[j] == ep:
            j += 1
        starts.extend(range(i, j, stride))
        i = j
    return starts


def pick_action_keys(ft, gt_phys: Dict[str, Any], pred: Dict[str, Any]) -> List[str]:
    """挑出「预测与 GT 都有」的动作键（**口径关键**，独立成函数供复用）。

    官方 ``scripts/open_loop_eval.py`` 读的是
    ``feature_transform.org_features['actions']``（原始 robot-config 键名），
    这里以它为准，再与实际 unapply 返回的键取交集兜底。

    ``Stage B0``：``lingbotvla/auto_learning`` 的 Fixed Baseline 也走这里 ——
    保证 baseline 与 evaluator 的 **action space 完全一致**。
    传 ``pred=gt_phys`` 即「只看 GT 有哪些键」。
    """
    both = {k for k in gt_phys if k in pred}
    org = list(getattr(ft, "org_features", {}).get("actions", []) or [])
    keys = [k for k in org if k in both]
    if not keys:
        keys = sorted(k for k in both if str(k).startswith("action."))
    if not keys:
        keys = sorted(both)
    return keys


def assemble_chunk(gt_phys: Dict[str, Any], pred_phys: Dict[str, Any],
                   action_keys: Sequence[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """按 ``action_keys`` 取帧、校验形状、拼成 ``(gt(N,D), pr(N,D))``；无有效键返回 ``None``。

    与 ``_evaluate_ids`` 共用 ⇒ baseline 与 evaluator 的
    **normalization / valid region / 拼接口径完全一致**。
    """
    g_parts, p_parts, n_frames = [], [], None
    for key in action_keys:
        g = _as_frames(_to_numpy(gt_phys[key]), key)
        p = _as_frames(_to_numpy(pred_phys[key]), key)
        # 🔴 2026-10-04 审查 B3：长度/维度不一致必须**显式报错**。
        #    旧实现用 `n = min(...)` + `[:n]` 静默截断 —— 一旦
        #    `n_action_steps != chunk_size`，会拿「预测前 16 帧」去比「GT 50 帧」，
        #    数字看着合理但完全是错的。
        if g.shape[0] != p.shape[0]:
            raise ValueError(
                f"[open_loop] 预测与 GT 的**帧数**不一致（{key}）："
                f"pred={p.shape[0]} vs gt={g.shape[0]}；"
                f"预测帧数应等于 model.config.n_action_steps")
        if g.shape[1] != p.shape[1]:
            raise ValueError(
                f"[open_loop] 预测与 GT 的**维度**不一致（{key}）："
                f"pred={p.shape[1]} vs gt={g.shape[1]}")
        g_parts.append(g)
        p_parts.append(p)
        n_frames = g.shape[0] if n_frames is None else min(n_frames, g.shape[0])
    if not g_parts or not n_frames:
        return None
    gt = np.concatenate([x[:n_frames] for x in g_parts], axis=1)     # (N_i, D)
    pr = np.concatenate([x[:n_frames] for x in p_parts], axis=1)
    return gt, pr


def aggregate_chunks(chunks: List[tuple]) -> Dict[str, Any]:
    """把 ``(episode_key, gt(N,D), pr(N,D))`` 列表按**完整 trajectory** 聚合。

    官方口径（``scripts/open_loop_eval.py``）：对每条轨迹，先把它的**所有 chunk
    沿时间轴拼成整条轨迹**算一个 MSE，再对轨迹**简单平均**。

    ⚠️ 不要退回「把每个 chunk 当一条 trajectory」。注意差异**不在帧数而在权重** ——
    评测里每个 chunk 的长度都等于 ``horizon``（不足处由 padding 补齐），所以
    `mean(chunk_mse)` 与 `mean(per-traj mse)` 的差别是：**拿到 2 个 chunk 的回合被计两次**。
    实测 click_bell val（10 回合，其中 4 个只拿到 1 个 chunk）：
    按轨迹 = 0.007510，按 chunk = 0.007738 ⇒ **差 +3.0%**（恰好那几个 1-chunk 回合 MSE 更低，
    被降权 ⇒ 旧口径偏**悲观**）。

    "拼接"在这里只有一种含义 —— **按回合把 chunk 拼回整条轨迹**，不是把不同回合拼在一起。

    独立成函数是为了让 ``tools/open_loop_parity_check.py`` 能直接单测它（无需模型）。
    """
    nan = float("nan")
    if not chunks:
        return {"n": 0, "n_chunks": 0, "frames": 0, "dims": 0,
                "mse": nan, "mae": nan, "r2": nan,
                "mse_pooled": nan, "mean_baseline_mse": nan,
                "mean_baseline_mse_per_traj": nan,
                "per_traj_mse": [], "per_traj_mae": [],
                "per_traj_ids": [], "per_traj_frames": [],
                "metric_dtype": METRIC_DTYPE_LABEL}

    groups: Dict[Any, List[tuple]] = {}
    order: List[Any] = []
    for ep_key, gt, pr in chunks:
        if ep_key not in groups:
            groups[ep_key] = []
            order.append(ep_key)
        groups[ep_key].append((gt, pr))

    per_traj: List[Dict[str, Any]] = []
    gt_chunks, pred_chunks = [], []
    for ep_key in order:
        gts = np.concatenate([g for g, _ in groups[ep_key]], axis=0)
        prs = np.concatenate([p for _, p in groups[ep_key]], axis=0)
        err = prs - gts                                 # 误差按输入精度（fp32）计算
        err64, gt64 = metric_arrays_for_aggregation(err, gts)   # 聚合提升 fp64
        per_traj.append({
            "episode": ep_key,
            "mse": float(np.mean(err64 ** 2)),
            "mae": float(np.mean(np.abs(err64))),
            "baseline": float(np.var(gt64, axis=0).mean()),
            "frames": int(gts.shape[0]),
        })
        gt_chunks.append(gts)
        pred_chunks.append(prs)

    # A) 官方口径：先把整条轨迹的帧拼起来算一个 MSE，再对 trajectory 简单平均
    # 跨轨迹平均同样 fp64 累加（fp32 在轨迹多/尺度差异大时累积舍入）
    mse = float(np.mean(np.asarray([t["mse"] for t in per_traj], dtype=np.float64)))
    mae = float(np.mean(np.asarray([t["mae"] for t in per_traj], dtype=np.float64)))
    mean_baseline_mse_per_traj = float(
        np.mean(np.asarray([t["baseline"] for t in per_traj], dtype=np.float64)))

    # B) 离线口径：所有帧 pool 在一起，逐维方差 → 对各维取均值
    gt_all = np.concatenate(gt_chunks, axis=0)       # (N, D)
    pr_all = np.concatenate(pred_chunks, axis=0)
    err_all = pr_all - gt_all
    err_all64, gt_all64 = metric_arrays_for_aggregation(err_all, gt_all)
    mse_pooled = float(np.mean(err_all64 ** 2))
    mean_baseline_mse = float(np.var(gt_all64, axis=0).mean())

    r2 = (float(1.0 - mse_pooled / mean_baseline_mse)
          if mean_baseline_mse > 0 else nan)
    return {
        "n": len(per_traj),                               # 轨迹数（官方口径的分母）
        "n_chunks": len(chunks),                          # 推理次数
        "frames": int(gt_all.shape[0]),
        "dims": int(gt_all.shape[1]),
        "mse": mse,                                       # 官方口径
        "mae": mae,                                       # 官方口径
        "mse_pooled": mse_pooled,                         # 离线口径
        "mean_baseline_mse": mean_baseline_mse,           # global constant baseline（离线口径）
        "mean_baseline_mse_per_traj": mean_baseline_mse_per_traj,
        "r2": r2,                                         # = 1 - mse_pooled / mean_baseline_mse
        "per_traj_mse": [t["mse"] for t in per_traj],     # 供与官方逐条对齐
        "per_traj_mae": [t["mae"] for t in per_traj],
        "metric_dtype": METRIC_DTYPE_LABEL,   # 误差 fp32 / 聚合 fp64（可核查）
        "per_traj_ids": [t["episode"] for t in per_traj],
        "per_traj_frames": [t["frames"] for t in per_traj],
    }


# ---------------------------------------------------------------------------
# `config.use_cache`（🔴 2026-10-04 审查 B1）
# ---------------------------------------------------------------------------
def _use_cache_owners(model: torch.nn.Module) -> List[Any]:
    """找出所有「``config`` 上带 ``use_cache``」的配置对象（按 id 去重）。

    传进来的是训练顶层包装，真正被 `sample_actions` / `predict_velocity` 读的
    ``self.config`` 可能在更内层 ⇒ 全部覆盖。
    """
    owners, seen = [], set()

    def _take(cfg):
        if cfg is not None and hasattr(cfg, "use_cache") and id(cfg) not in seen:
            seen.add(id(cfg))
            owners.append(cfg)

    _take(getattr(model, "config", None))
    try:
        for m in model.modules():
            _take(getattr(m, "config", None))
    except Exception:  # noqa: BLE001
        pass
    return owners


def _force_use_cache(model: torch.nn.Module, value: bool):
    owners = _use_cache_owners(model)
    saved = [(cfg, cfg.use_cache) for cfg in owners]
    for cfg in owners:
        cfg.use_cache = value
    return saved


def _restore_use_cache(saved) -> None:
    for cfg, old in saved or []:
        cfg.use_cache = old


# ---------------------------------------------------------------------------
# attention_implementation（🔴 2026-10-04 实测 H1）
# ---------------------------------------------------------------------------
# `modeling_lingbot_vla_v2.py::forward` 在 `attention_implementation == "flex_cached"` 时：
#     _full_len = query_states.shape[1]        # ← 只取 query 长度
#     build_block_mask(attention_mask, heads, _full_len, _full_len)
# 而 `predict_velocity` 里 query=suffix、key=prefix+suffix（用了 KV cache）
# ⇒ 两者不等：
#   * `use_cache=False`：key 也只有 suffix ⇒ 长度**恰好相等** ⇒ **不报错，但前缀被静默丢掉**
#   * `use_cache=True` ：key = prefix+suffix ⇒ flex_attention 直接抛
#       ValueError: block_mask was created for block_mask.shape=(1,32,128,128)
#                   but got q_len=128 and kv_len=337
#
# 官方 deploy 早就绕开了：`deploy/lingbot_vla_v2_policy.py:290`
#     config.attention_implementation = 'eager'   # 建模**之前**就改掉
# ⇒ 本模块在评测期间做同样的事（改 config + 换 attention_interface），评测后还原。
def _attention_owners(model: torch.nn.Module):
    out = []
    for m in model.modules():
        if hasattr(m, "attention_interface") and hasattr(m, "get_attention_interface"):
            out.append(m)
    return out


def _force_eager_attention(model: torch.nn.Module):
    """把 attention 换成 eager（等价于 deploy 的 `attention_implementation='eager'`）。"""
    saved = []
    for m in _attention_owners(model):
        cfg = getattr(m, "config", None)
        old_impl = getattr(cfg, "attention_implementation", None)
        saved.append((m, cfg, old_impl, m.attention_interface))
        if cfg is not None and old_impl is not None:
            cfg.attention_implementation = "eager"
        m.attention_interface = m.get_attention_interface()
    return saved


def _restore_eager_attention(saved) -> None:
    for m, cfg, old_impl, old_iface in saved or []:
        if cfg is not None and old_impl is not None:
            cfg.attention_implementation = old_impl
        m.attention_interface = old_iface


# ---------------------------------------------------------------------------
# 恢复审计
# ---------------------------------------------------------------------------
# 2026-10-04：回答「eval 结束后，训练状态是否真的回到原样」。
# 每次评测的 `finally` 里逐项**断言**临时改过的东西都还原了，不一致就报错。
# 只靠「我写了 try/finally」不算证据 —— 这是唯一能证明恢复成功的东西。
AUDIT_STRICT_ENV = "OPEN_LOOP_AUDIT_STRICT"   # 默认 "1"；置 "0" 降级为仅告警


def _audit_strict() -> bool:
    return os.environ.get(AUDIT_STRICT_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _audit_restore(*, rng_snap, train_flags, compile_flag, use_cache_saved,
                   ft_aug_orig, model, attn_saved=None) -> List[str]:
    """逐项核对「临时状态是否 100% 还原」；返回不一致的描述列表（空 = 全过）。"""
    problems: List[str] = []

    # 1) 逐模块 training 标志
    now = [(m, m.training) for m in model.modules()]
    if len(now) != len(train_flags):
        problems.append(f"模块数变了: {len(train_flags)} → {len(now)}")
    else:
        bad = [(i, t, c) for i, ((_, t), (_, c)) in enumerate(zip(train_flags, now)) if t != c]
        if bad:
            problems.append(
                f"{len(bad)} 个子模块 training 标志没还原，前 3 个 (idx, 期望, 实际)={bad[:3]}")

    # 2) 每个 config 的 use_cache
    for cfg, old in use_cache_saved or []:
        if getattr(cfg, "use_cache", None) != old:
            problems.append(f"config.use_cache 没还原: 期望 {old}，实际 {cfg.use_cache}")

    # 3) 每个 feature_transform 的 image_augment
    for _ft, orig in (ft_aug_orig or {}).values():
        if hasattr(_ft, "image_augment") and _ft.image_augment != orig:
            problems.append(
                f"feature_transform.image_augment 没还原: 期望 {orig}，实际 {_ft.image_augment}")

    # 4) compile 开关（`_compiled_predict_velocity` 被置 None 是**有意**的，不查）
    if compile_flag is not None and \
            getattr(model, "_use_compile_predict_velocity", None) != compile_flag:
        problems.append("_use_compile_predict_velocity 没还原")

    # 4b) attention_implementation + attention_interface
    for m, cfg, old_impl, old_iface in attn_saved or []:
        if cfg is not None and old_impl is not None and \
                getattr(cfg, "attention_implementation", None) != old_impl:
            problems.append(f"attention_implementation 没还原: 期望 {old_impl}，"
                            f"实际 {cfg.attention_implementation}")
        if m.attention_interface is not old_iface:
            problems.append(f"{type(m).__name__}.attention_interface 没还原（对象身份不一致）")

    # 5) RNG：torch CPU / CUDA、numpy、python
    if not torch.equal(torch.get_rng_state(), rng_snap["torch_cpu"]):
        problems.append("torch CPU RNG 没还原")
    if "torch_cuda" in rng_snap and torch.cuda.is_available():
        cur = torch.cuda.get_rng_state_all()
        if len(cur) != len(rng_snap["torch_cuda"]) or not all(
                torch.equal(a, b) for a, b in zip(cur, rng_snap["torch_cuda"])):
            problems.append("torch CUDA RNG 没还原")
    s0 = rng_snap["numpy"]
    st = np.random.get_state()
    if st[0] != s0[0] or st[2] != s0[2] or st[3] != s0[3] or not np.array_equal(st[1], s0[1]):
        problems.append("numpy RNG 没还原")
    if random.getstate() != rng_snap["python"]:
        problems.append("python random 没还原")

    return problems


# ---------------------------------------------------------------------------
# Qwen3-VL 视觉塔「预计算网格缓存」
# ---------------------------------------------------------------------------
# `modeling_lingbot_vla_v2.py::QwenvlWithExpertV2Model.get_image_features` 在
# `config.precompute_grid_thw=True` 时，把这 5 个属性**按首次调用时的 grid_thw 缓存**，
# 之后只在 `position_embeddings is None` 时重算：
#     pos_embeds / position_embeddings / cu_seqlens / visual_split_sizes / visual_max_seqlen
# ⇒ 训练 batch 与评测单样本的网格不同时，`visual_split_sizes` 会张冠李戴。
_VISUAL_GRID_CACHE_KEYS = (
    "pos_embeds",
    "position_embeddings",
    "cu_seqlens",
    "visual_split_sizes",
    "visual_max_seqlen",
)


def _visual_grid_cache_owner(model: torch.nn.Module):
    """返回持有那 5 个缓存属性的子模块。

    ⚠️ 不能只 `getattr(model, "qwenvl_with_expert")` —— 传进来的是**训练用的顶层包装**
    （policy / FSDP2 包装），`qwenvl_with_expert` 在更里面一层。2026-10-04 第一版就是这么
    写空的：`saved` 直接是 None、静默没清缓存，评测照样炸。所以这里**遍历 `modules()`**。
    """
    # 先看顶层自己（有些版本直接挂顶层）
    if any(hasattr(model, k) for k in _VISUAL_GRID_CACHE_KEYS):
        return model
    for m in model.modules():
        if m is model:
            continue
        if any(hasattr(m, k) for k in _VISUAL_GRID_CACHE_KEYS):
            return m
    return None


def _visual_grid_cache_clear(model: torch.nn.Module):
    """把缓存全部置 None（下次 `get_image_features` 会按新网格重算）；返回原值供恢复。"""
    owner = _visual_grid_cache_owner(model)
    if owner is None:
        return None
    saved = {k: getattr(owner, k, None) for k in _VISUAL_GRID_CACHE_KEYS if hasattr(owner, k)}
    for k in saved:
        setattr(owner, k, None)
    return {"owner": owner, "values": saved}


def _visual_grid_cache_restore(model: torch.nn.Module, saved) -> None:
    if not saved:
        return
    for k, v in saved["values"].items():
        setattr(saved["owner"], k, v)


# ---------------------------------------------------------------------------
# 共用的「安全评测上下文」（Stage B0 抽出）
# ---------------------------------------------------------------------------
# 原来这套 snapshot → 强制 → finally 恢复 → 审计 是写在 `validate()` 方法体里的。
# `lingbotvla/auto_learning` 的 `evaluate_task()` 需要**同一套**保护（否则
# use_cache / attention / 视觉网格缓存 / RNG 都会残留、污染训练）。
# ⇒ 抽成共用上下文，`validate()` 与新接口都走它。**语义逐条保持不变**。
@dataclasses.dataclass
class EvalSetupReport:
    """评测开始前「被临时改过什么」的清单（供调用方打日志 / 断言）。"""

    rng_snap: Dict[str, Any]
    train_flags: List[tuple]
    compile_flag: Any
    use_cache_saved: Optional[List[tuple]] = None
    attn_saved: Optional[List[tuple]] = None
    visual_cache_saved: Optional[Dict[str, Any]] = None


@contextlib.contextmanager
def safe_eval_context(
    *,
    model: torch.nn.Module,
    logger,
    ft_aug_registry: Optional[Dict[int, tuple]] = None,
    on_setup: Optional[Callable[[EvalSetupReport], None]] = None,
    on_audit_ok: Optional[Callable[[], None]] = None,
    seed: int = EVAL_SEED,
) -> Iterator[EvalSetupReport]:
    """评测期间临时改状态的 **snapshot / 强制 / finally 恢复 / 审计** 四件套。

    进入时：强制 eager（避开 compile 产物）→ 固定 seed → ``model.eval()`` →
    清显存碎片 → ``use_cache=True`` → eager attention → 清视觉网格缓存。
    退出时（**任何路径**）：按相反顺序全部还原，并跑一次**恢复审计**。

    ``ft_aug_registry`` 是「每个评测集自己的 feature_transform」登记表
    （``OpenLoopValidator._ft_aug_orig``）；它在 **with 体内**被填充，
    所以必须在退出时读取**当时的**内容 —— 因此传引用而不是传值。

    审计不通过时：默认 ``raise``（``OPEN_LOOP_AUDIT_STRICT=0`` 可降级为仅告警）。
    """
    rng_snap = _rng_snapshot()
    train_flags = _module_training_flags(model)
    compile_flag = getattr(model, "_use_compile_predict_velocity", None)
    use_cache_saved = None
    attn_saved = None
    visual_cache_saved = None
    report = EvalSetupReport(rng_snap=rng_snap, train_flags=train_flags,
                             compile_flag=compile_flag)
    try:
        # ① 强制 eager：避免拿到训练用的编译产物；也避免 inference tensor 逃逸进 compile cache
        if compile_flag is not None:
            model._use_compile_predict_velocity = False
            model._compiled_predict_velocity = None
        # ② 固定 seed（整次 eval 一致 ⇒ 不同 step 可比）
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # ③ 逐模块 eval（不是 model.eval() 一刀切，但那也可以：下面是等价写法）
        model.eval()
        # ③b 释放训练侧遗留的碎片/缓存 —— 单卡余量本来就紧，不清的话
        #     sample_actions 自己的激活 + KV cache 可能装不下
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ③c 🔴 临时打开 use_cache（2026-10-04 审查 B1 + CPU 实测确认）
        #   训练配置 `use_cache=False`（`configuration_lingbot_vla.py:104`，V2 不覆盖），
        #   而 `sample_actions` 传的是 `fill_kv_cache=True, use_cache=self.config.use_cache`
        #   ⇒ `handle_kv_cache` 直接跳过 ⇒ `past_key_values` 恒为 None
        #   ⇒ **去噪阶段完全没有 VLM 前缀条件**（不报错、结果错）。
        #   ⇒ 只在 eval 期间开，`finally` 恢复；**不动训练全局配置**。
        use_cache_saved = _force_use_cache(model, True)
        # ③c-2 🔴 临时换 eager attention（等价 deploy 的 `attention_implementation='eager'`）
        #   训练配置是 `flex_cached`，其 block_mask 按 **query 长度**建
        #   （`_full_len = query_states.shape[1]`）；`predict_velocity` 用了 KV cache 时
        #   key = prefix+suffix ≠ query ⇒ 长度不匹配。
        attn_saved = _force_eager_attention(model)
        # ③d 🔴 清掉 Qwen3-VL 视觉塔的「预计算网格缓存」（2026-10-04 micro=10 实测）
        #   `precompute_grid_thw: true` 时这 5 个属性按**首次调用时的 grid_thw** 缓存
        #   ⇒ 训练 batch（micro × 相机数）与评测单样本（1 × 相机数）不同时会张冠李戴，
        #      `torch.split` 直接 RuntimeError。
        visual_cache_saved = _visual_grid_cache_clear(model)

        report.use_cache_saved = use_cache_saved
        report.attn_saved = attn_saved
        report.visual_cache_saved = visual_cache_saved
        if on_setup is not None:
            on_setup(report)
        yield report
    finally:
        # 全部临时状态恢复（顺序与设置相反）
        _restore_use_cache(use_cache_saved)
        _restore_eager_attention(attn_saved)
        _visual_grid_cache_restore(model, visual_cache_saved)
        for _ft, _orig_aug in (ft_aug_registry or {}).values():
            if hasattr(_ft, "image_augment"):
                _ft.image_augment = _orig_aug
        _module_training_restore(train_flags)
        if compile_flag is not None:
            model._use_compile_predict_velocity = compile_flag
            model._compiled_predict_velocity = None
        _rng_restore(rng_snap)

        # ---- 恢复审计：逐项断言临时状态已 100% 还原 ----
        # 这是回答「eval 完还能不能接着训」的唯一证据（默认不一致就抛错）。
        try:
            _problems = _audit_restore(
                rng_snap=rng_snap,
                train_flags=train_flags,
                compile_flag=compile_flag,
                use_cache_saved=use_cache_saved,
                ft_aug_orig=ft_aug_registry,
                model=model,
                attn_saved=attn_saved,
            )
        except Exception as _audit_exc:  # noqa: BLE001
            _problems = [f"审计自身异常: {type(_audit_exc).__name__}: {_audit_exc}"]
        if _problems:
            _msg = ("[open_loop] ❌ 恢复审计未通过（eval 可能污染了训练状态）:\n  - "
                    + "\n  - ".join(_problems))
            logger.warning(_msg)
            if _audit_strict():
                raise RuntimeError(
                    _msg + f"\n  （确认无碍可设 {AUDIT_STRICT_ENV}=0 降级为仅告警）")
        elif on_audit_ok is not None:
            on_audit_ok()


# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------
class OpenLoopValidator:
    """在训练进程内、用当前权重跑 open-loop 验证，结果写 TB。"""

    def __init__(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        model_config=None,
        args,
        processor,
        use_depth_align: bool,
        writer=None,
        logger,
        train_monitor_ids: Optional[Sequence[int]] = None,
        val_ids: Optional[Sequence[int]] = None,
        device: str = "cuda",
        dump_dir: Optional[str] = None,
        per_episode_stride: bool = True,
    ):
        # `model=None` + `model_config=<HF config>` ⇒ **只收集 GT / 算 baseline** 的离线模式：
        # 不需要权重（Stage B0 的 Fixed Baseline 预计算）。推理路径仍需传 model。
        if model is None and model_config is None:
            raise ValueError("OpenLoopValidator 需要 model 或 model_config 至少给一个")
        self.model = model
        self._model_config = model_config if model_config is not None else model.config
        self.args = args
        self.processor = processor
        self.use_depth_align = use_depth_align
        self.writer = writer
        self.logger = logger
        self.device = device
        # 可选：把每个 chunk 的 GT/pred 存成 .npy（供与官方脚本逐值对拍）
        self.dump_dir = dump_dir
        self._dump_prefix = None
        # True（默认）= 每个回合各自从首帧跳步（对齐官方 open_loop_eval.py）；
        # False = 在拼接序列上跳（旧行为，仅供 A/B 量化用）
        self.per_episode_stride = per_episode_stride
        self.train_monitor_ids = list(train_monitor_ids or DEFAULT_TRAIN_MONITOR_IDS)
        self.val_ids = list(val_ids or DEFAULT_VAL_IDS)
        self._ds_cache: Dict[str, Any] = {}
        self._ft_by_path: Dict[str, Any] = {}      # 每个评测集**自己的** feature_transform
        self._ft_aug_orig: Dict[int, tuple] = {}   # id(ft) -> (ft, 原始 image_augment)
        self._ft = None          # 兼容用：第一个 dataset 的 feature_transform
        self._warned_augment = False
        self._shape_dumped = False   # 只在第一次 eval 时打印 item 的键名/形状
        self._chunk_dumped = False   # 只在第一次 eval 时打印 chunk 内动作是否随时间变化
        self._grid_cache_logged = False   # 只在第一次 eval 时打印被清掉的视觉网格缓存
        self._use_cache_logged = False    # 只在第一次 eval 时打印 use_cache 临时开关
        self._attn_logged = False         # 只在第一次 eval 时打印 attention 临时开关
        self._audit_logged = False        # 只在第一次 eval 时打印「恢复审计通过」
        self._normstats_logged = False    # 只在第一次 eval 时打印归一化统计指纹
        self._strict_logged = False
        self._noise_gen = None            # flow-matching noise 的专用 generator（每次 eval 重置）

        # 🔴 只支持单卡：多卡下 rank0 跑 eval 时其他 rank 会直接进入下一步 ⇒ FSDP2
        #    all-gather 死锁；且 rank0 上参数是分片的、评测结果无效。
        #    2026-10-04 审查 B6：这里直接 fail-fast，不实现多卡 eval。
        #    （离线「只收 GT」模式 model=None，不走 FSDP/推理，不受此限制。）
        if model is not None:
            ws = _world_size()
            if ws > 1:
                raise RuntimeError(
                    f"[open_loop] 训练中原地 open-loop validation 目前**只支持单卡**，"
                    f"当前 world_size={ws}。\n"
                    f"  rank0 跑 eval 时其他 rank 会立刻进入下一步 ⇒ FSDP2 的 all-gather 会死锁；"
                    f"且 rank0 上的参数是分片的，评测结果无效。\n"
                    f"  请二选一：① 用单卡训练；② 把 --train.open_loop_eval_steps 设为 0 关闭本功能。")

        if getattr(args.data, "image_augment", False):
            logger.warning(
                "[open_loop] ⚠️ image_augment=True ⇒ 本模块的观测会带增强，"
                "与官方 scripts/open_loop_eval.py 口径不一致，指标不可直接对比。"
                "建议 eval 期间用 image_augment=False 的配置。")

    # -- 数据 -----------------------------------------------------------------
    def _dataset(self, episode_ids_file: str):
        """按训练同一份 dataset_config 建一个只含指定回合的子集数据集（缓存复用）。"""
        if episode_ids_file in self._ds_cache:
            return self._ds_cache[episode_ids_file]
        cfg = (dataclasses.replace(self.args.data, episode_ids_file=episode_ids_file)
               if dataclasses.is_dataclass(self.args.data)
               # 离线解耦评测（tools/open_loop_eval_inprocess.py）传进来的可能是
               # SimpleNamespace，不是 dataclass ⇒ 浅拷贝 + setattr
               else _replace_attr(self.args.data, episode_ids_file=episode_ids_file))
        # ⚠️ 官方 `scripts/open_loop_eval.py` 里这两个字段是**调用方**补上的
        #    （`policy.data_config.chunk_size = policy.config.chunk_size` /
        #      `policy.data_config.num_episode = None`）；`MyDataArguments` 本身不声明它们。
        #    漏了就会 AttributeError: 'MyDataArguments' object has no attribute 'chunk_size'
        #    （2026-10-04 smoke 实测）
        if not hasattr(cfg, "chunk_size"):
            setattr(cfg, "chunk_size", int(getattr(self._model_config, "chunk_size", 50)))
        if not hasattr(cfg, "num_episode"):
            setattr(cfg, "num_episode", None)
        ds = build_vla_dataset(
            dataset_config=cfg,
            model_config=self.args.model,
            config=self._model_config,
            processor=self.processor,
            use_depth_align=self.use_depth_align,
        )
        # ⚠️ 每个评测集**各自**有一份 FeatureTransform（`MultiVLADataset` 每次构造都会新建
        #    自己的 `feature_transforms`，见 multi_vla_dataset.py:86-111）
        #    ⇒ 不能共用一份做 unapply（2026-10-04 审查 B5）。
        ft = _find_feature_transform(ds)
        if ft is None:
            raise RuntimeError(
                f"从 {type(ds).__name__} 上找不到 feature_transform，无法反归一化"
                f"（可用属性: {[a for a in dir(ds) if 'feature' in a or 'dataset' in a]}）")
        self._ft_by_path[episode_ids_file] = ft
        if self._ft is None:
            self._ft = ft

        # 把「归一化统计到底来自哪个文件」打出来 —— 2026-10-04 那个「训练/评测两套 stats」
        # 的坑就是因为它完全不可见。这里给一个可肉眼核对的指纹。
        if not self._normstats_logged:
            self._normstats_logged = True
            _ns = getattr(getattr(ft, "normalizer", None), "norm_stats", None) or {}
            _k = "observation.state.arm.position"
            _m = _ns.get(_k, {}).get("mean")
            _fp = (np.round(np.asarray(_m).reshape(-1)[:3], 5).tolist()
                   if _m is not None else None)
            self.logger.info_rank0(
                f"[open_loop] 归一化统计指纹 {_k}.mean[:3] = {_fp}"
                f"（本数据集那套 = [-0.21669, 1.08914, 0.79395]；"
                f"robotwin.json 那套 = [-0.23846, 1.13016, 0.80707]）")

        # 🔴 eval 专用 strict 模式（2026-10-04 审查 B4）：`MultiVLADataset.__getitem__`
        #    在异常时会 `np.random.randint` **换一帧随机重试**（multi_vla_dataset.py:165-179）
        #    ⇒ 解码偶发失败会静默给出 (图像, 动作) 不匹配的样本。eval 下改为直接抛错。
        #    该开关默认 False，**不影响正式训练**。
        if hasattr(ds, "strict_getitem"):
            ds.strict_getitem = True
            if not self._strict_logged:
                self._strict_logged = True
                self.logger.info_rank0(
                    "[open_loop] 评测数据集已开启 strict_getitem：读帧失败直接报错，不随机重试")
        elif not self._strict_logged:
            self._strict_logged = True
            self.logger.warning(
                "[open_loop] ⚠️ 数据集没有 strict_getitem 开关 ⇒ 读帧失败时仍会随机重试换帧")

        self._ds_cache[episode_ids_file] = ds
        return ds

    def _ft_for(self, episode_ids_file: str):
        ft = self._ft_by_path.get(episode_ids_file)
        if ft is None:
            raise RuntimeError(f"{episode_ids_file} 还没有建过数据集，取不到 feature_transform")
        # 每次评测都重新关掉增强（因为 validate 的 finally 会还原）
        if hasattr(ft, "image_augment"):
            self._ft_aug_orig.setdefault(id(ft), (ft, ft.image_augment))
            ft.image_augment = False
        return ft

    # -- 内存 -----------------------------------------------------------------
    def clear_dataset_cache(self) -> int:
        """释放按 ``episode_ids_file`` 缓存的子集数据集；返回释放的条目数。

        🔴 **长跑时必须定期调用**（Stage B0 新增）。
        `_ds_cache` / `_ft_by_path` 只增不减 —— 每换一组 episode ids 就多留一份数据集
        （含 LeRobot 元数据与 HF 表）。Auto Learning 会按
        ``(task, split, ids 指纹)`` **反复评测** ⇒ 不清理必然把容器内存吃满。

        ⚠️ **只能在评测之外调用**（`safe_eval_context` 的 `finally` 会读
        `_ft_aug_orig` 做 image_augment 还原；评测中途清空会让还原失效）。
        """
        n = len(self._ds_cache)
        self._ds_cache.clear()
        self._ft_by_path.clear()
        self._ft_aug_orig.clear()
        self._ft = None
        return n

    def _episode_ids_file(self, ids: Sequence[int], tag: str) -> str:
        """把一组回合号落成临时白名单文件（build_vla_dataset 只认文件）。"""
        out_dir = os.path.join(self.args.train.output_dir, "_open_loop_ids")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{tag}.json")
        with open(path, "w") as f:
            json.dump(sorted(int(i) for i in ids), f)
        return path

    # -- 噪声（显式喂，保证可复现 + 可与官方对拍）--------------------------------
    def _noise_generator(self, device):
        """专用 generator：**只**给 flow-matching 的 noise 用。

        ⚠️ 不能靠 `torch.manual_seed` + `noise=None`：
        ``torch.randn`` 取的是**随机流上第几次**的值，模型加载本身会消耗随机数
        （实测有 4 次 `randn(256,2560)`）⇒「同一个 seed」并不等于「同一份噪声」。
        用独立 generator 就能与官方评测**喂完全相同的噪声**做对拍。
        """
        if self._noise_gen is None:
            dev = "cuda" if str(device).startswith("cuda") else "cpu"
            g = torch.Generator(device=dev)
            g.manual_seed(EVAL_SEED)
            self._noise_gen = g
        return self._noise_gen

    # -- 推理（glue 照抄 deploy.PolicyPreprocessMixin.sample_actions_batch）----
    def _infer_one(self, item: Dict[str, Any], ft) -> Dict[str, np.ndarray]:
        """对一条已 transform 的样本跑一次 ``model.sample_actions``，返回**物理量**动作块。

        ``ft`` 必须是**这条 item 所属数据集自己的** ``feature_transform``（审查 B5）。
        """
        # 🔴 推理 dtype 必须跟**模型权重的实际 dtype** 对齐。
        #    旧写法 `getattr(self.args.train, "use_bf16", False)` 在本仓**恒为 False**
        #    —— 精度开关是 `enable_mixed_precision`（`arguments.py` 里没有 `use_bf16`），
        #    于是 bf16 训练出来的权重 + 评测按 float32 喂输入 ⇒ `embed_suffix` 的
        #    `state_proj(state)` 直接报
        #      RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
        #    （2026-10-07 A–G 回归首次跑到 bf16 组合时实测）。以前全是 F32 所以没暴露。
        #    优先级：`config.action_fp32`（模型自己会 `_fp32_linear` 上转权重）> 模型权重 dtype。
        # 精度三分离（用户 2026-10-08）：权重 dtype / 推理 dtype / 指标 dtype。
        # 决议抽到 eval_precision.resolve_inference_dtype（纯函数、可单测）；
        # 显式请求与权重不一致时 fail-fast（不允许按 fp32 推理再 cast 结果）。
        from lingbotvla.utils.eval_precision import describe_precision, resolve_inference_dtype
        _cfg0 = getattr(self, "_model_config", None)
        _action_fp32 = bool(getattr(_cfg0, "action_fp32", False))
        _p0 = next(self.model.parameters(), None)
        _wdt = _p0.dtype if (_p0 is not None and _p0.dtype.is_floating_point) else None
        _requested = str(getattr(self.args.train, "eval_inference_dtype", "auto") or "auto")
        dtype, _prec = resolve_inference_dtype(
            requested=_requested, action_fp32=_action_fp32, weight_dtype=_wdt,
            fallback_bf16=bool(getattr(self.args.train, "use_bf16", False)))
        use_bf16 = (dtype == torch.bfloat16)
        if not getattr(self, "_precision_logged", False):
            self._precision_logged = True
            self.logger.info_rank0(f"[open_loop] 精度口径：{describe_precision(_prec)}")

        images = item["images"]
        img_masks = item["img_masks"]
        lang_tokens = item["lang_tokens"]
        lang_masks = item["lang_masks"]
        state = item["state"]
        image_grid_thw = item.get("image_grid_thw", None)

        if img_masks.ndim < 2:
            images = images.unsqueeze(0)
            img_masks = img_masks.unsqueeze(0)
        if lang_tokens.ndim == 1:
            lang_tokens = lang_tokens.unsqueeze(0)
            lang_masks = lang_masks.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        grid = image_grid_thw
        if isinstance(grid, torch.Tensor):
            grid = grid.to(device=self.device, dtype=torch.long)

        # 可选：dump **模型输入**（供与官方逐值对拍）
        if self.dump_dir and self._dump_prefix:
            os.makedirs(self.dump_dir, exist_ok=True)
            _p = os.path.join(self.dump_dir, f"{self._dump_prefix}_in")
            for _k, _v in (("images", images), ("img_masks", img_masks),
                           ("lang_tokens", lang_tokens), ("lang_masks", lang_masks),
                           ("state", state), ("grid", grid)):
                if _v is not None:
                    np.save(f"{_p}_{_k}.npy",
                            _v.detach().float().cpu().numpy())

        # 显式喂噪声（形状/dtype/device 与 `sample_actions` 内部默认值一致）
        _cfg = self._model_config
        _shape = (1,
                  int(getattr(_cfg, "n_action_steps", 50)),
                  int(getattr(_cfg, "max_action_dim", 55)))
        noise = torch.randn(
            _shape,
            generator=self._noise_generator(self.device),
            device=self.device,
            dtype=dtype,
        )

        actions = self.model.sample_actions(
            images.to(dtype=dtype, device=self.device),
            img_masks.to(device=self.device),
            lang_tokens.to(device=self.device),
            lang_masks.to(device=self.device),
            state.to(dtype=dtype, device=self.device),
            noise=noise,
            image_grid_thw=grid,
        )
        # 反归一化回物理量（与 deploy.LingbotVLAv2InferencePolicy.select_action 同路径）
        # ⚠️ 必须 squeeze 掉 batch 维：``FeatureTransform.reverse_pad_and_concat`` 里是
        #    ``item['actions'][:, action_joint_mask]``，要求 actions 是 **(T, D) 二维**；
        #    传 (1, T, D) 会报
        #      IndexError: The shape of the mask [55] at index 0 does not match the
        #                  shape of the indexed tensor [1, 50, 55] at index 1
        #    （2026-10-04 smoke 实测）。deploy 的单样本路径正是 ``.squeeze(0)``。
        single = dict(item)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions.squeeze(0)
        single["actions"] = actions.to(dtype=torch.float32, device="cpu")
        if use_bf16 and "state" in single:
            single["state"] = single["state"].to(dtype=torch.float32)
        return ft.unapply(single)

    # -- 动作键 ---------------------------------------------------------------
    def _pick_action_keys(self, ft, gt_phys: Dict[str, Any],
                          pred: Dict[str, Any]) -> List[str]:
        """挑出「预测与 GT 都有」的动作键（薄封装，实现在模块级 ``pick_action_keys``）。"""
        return pick_action_keys(ft, gt_phys, pred)

    # -- 指标 -----------------------------------------------------------------
    def _evaluate_ids(self, ids: Sequence[int], tag: str) -> Dict[str, Any]:
        """在给定回合集合上算指标。**两套口径各自对齐自己的参照物**。

        A) **官方 `scripts/open_loop_eval.py` 口径** —— 先逐条 trajectory 算、再对
           trajectory **简单平均**（官方就是 ``np.mean(all_mse)``）：
               ``mse`` / ``mae``
        B) **离线 `tools/collect_open_loop.py` 口径** —— 把所有帧 pool 在一起，
           **逐维算方差再对各维取均值**：
               ``mse_pooled`` / ``mean_baseline_mse``
           这是「**global constant-action baseline**」：每个维度都输出自己的**全局**常数。
               ``r2 = 1 - mse_pooled / mean_baseline_mse``   ← 标准 R²，分子分母同口径

        ⚠️ **`mean_baseline_mse` 与 per-trajectory 版本不是同一个东西**（见下方
        ``mean_baseline_mse_per_traj``）。由全方差公式：
            ``Var_pooled = E_i[Var_i] + Var_i[E_i]``
        即 ``Var_pooled ≥ E_i[Var_i]``，**差的是「各 trajectory 均值之间的差异」**。
          * ``mean_baseline_mse``（global，本函数返回、写 TB）= ``Var_pooled`` 的逐维均值
            = 「每维输出一个全局常数」的 MSE  ← 与离线脚本一致
          * ``mean_baseline_mse_per_traj`` = ``E_i[Var_i]``
            = 「每条 trajectory 各自输出自己的常数」的 MSE  ← 更小
        demo_clean 下初始条件固定 ⇒ 两者应很接近，但**不能假设相等**。

        train-monitor 与 val **各自独立计算**（评测集不同 ⇒ baseline 不同）。
        """
        ds_path = self._episode_ids_file(ids, tag)
        ds = self._dataset(ds_path)
        ft = self._ft_for(ds_path)
        ep_map = _episode_index_map(ds)
        if ep_map is None:
            self.logger.warning(
                f"[open_loop] ⚠️ [{tag}] 拿不到 local_idx→episode_index 映射，"
                "退化为「每个 chunk 当作一条 trajectory」，聚合口径会与官方有偏差")

        chunks: List[tuple] = []        # (episode_key, gt(N,D), pr(N,D))
        action_keys: List[str] = []
        # 与官方 ``scripts/open_loop_eval.py`` 对齐：``chunk_ret=True`` 时按
        # ``action_horizon = chunk_size`` 步进（官方 ``range(start_id, end_id, action_horizon)``），
        # 每次推理用**整段** chunk。
        # ⚠️ 逐帧（stride=1）会把同一批帧反复预测 ~chunk_size 次：一条 77 帧的轨迹要
        #    77 次推理（2026-10-04 smoke 实测单次推理数秒级）⇒ 完全做不到"高频"，
        #    而且**口径与官方不一致**（官方一条轨迹只推 2 次）。
        stride = max(1, int(getattr(self._model_config, "chunk_size", 50) or 50))
        # 🔴 2026-10-05：跳步必须**按回合各自从首帧开始**（见 `per_episode_starts`），
        #   与官方 `range(start_id, end_id, action_horizon)` 等价。
        #   ⚠️ padding 占比会变（实测我们 34.5% vs 官方 23.0%），但**偏差方向未实测**：
        #      「padding 是送分题所以 MSE 偏低」只是**推测** —— 恒定 GT 利于"预测均值"，
        #      可训练好的模型会预测**运动延续**，在这些帧上反而可能更差。
        #      要量化请用 `tools/open_loop_eval_inprocess.py --flat-stride` 在同一 ckpt 上跑两遍。
        if self.per_episode_stride and ep_map is not None:
            starts = per_episode_starts(ep_map, stride)
        else:
            starts = list(range(0, len(ds), stride))
        for local_idx in starts:
            item = ds[local_idx]
            if not self._shape_dumped:
                self._shape_dumped = True
                keys = sorted(item.keys())
                self.logger.info_rank0(
                    f"[open_loop][debug] dataset item 共 {len(keys)} 个键: {keys}")
                for _k in ("images", "img_masks", "lang_tokens", "lang_masks",
                           "state", "image_grid_thw", "actions"):
                    if _k in item:
                        _v = item[_k]
                        _sh = tuple(_v.shape) if hasattr(_v, "shape") else (type(_v).__name__,)
                        self.logger.info_rank0(
                            f"[open_loop][debug]   {_k}: {type(_v).__name__} {_sh}")
            # GT 必须与预测走**同一条**反归一化路径：dataset item 里的 ``actions`` 是
            # **归一化后**的 (chunk, max_action_dim)，原始键（``action.*``）已被 apply 吃掉。
            # 官方 open_loop_eval.py 也是 apply→unapply 才拿到物理量 GT。
            gt_phys = ft.unapply(dict(item))
            self._dump_prefix = f"{tag}_ep{local_idx}"
            # 可选：dump **归一化前的原始 state**（用于定位「模型输入 state 不同」的来源）
            if self.dump_dir:
                try:
                    _sub = (ds._datasets[0].dataset
                            if getattr(ds, "_datasets", None) else getattr(ds, "dataset", None))
                    if _sub is not None:
                        _raw = _sub[local_idx]
                        np.save(os.path.join(self.dump_dir,
                                             f"{tag}_ep{local_idx}_rawstate.npy"),
                                np.asarray(_raw["observation.state"], dtype=np.float32))
                        np.save(os.path.join(self.dump_dir,
                                             f"{tag}_ep{local_idx}_rawidx.npy"),
                                np.asarray([int(_raw["index"])], dtype=np.int64))
                except Exception as _e:  # noqa: BLE001
                    self.logger.warning(f"[open_loop] raw state dump 失败: {_e}")
            pred = self._infer_one(item, ft)
            self._dump_prefix = None
            if not action_keys:
                action_keys = self._pick_action_keys(ft, gt_phys, pred)
                self.logger.info_rank0(
                    f"[open_loop][debug] action keys = {action_keys}；"
                    f"unapply 后的键 = {sorted(gt_phys)}")
                for _k in action_keys:
                    self.logger.info_rank0(
                        f"[open_loop][debug]   GT {_k}: {tuple(_to_numpy(gt_phys[_k]).shape)}"
                        f" | pred: {tuple(_to_numpy(pred[_k]).shape)}")
            if not self._chunk_dumped:
                self._chunk_dumped = True
                _g = _to_numpy(gt_phys[action_keys[0]])
                _a = _to_numpy(item["actions"])
                _pad = _to_numpy(item["action_is_pad"]) if "action_is_pad" in item else None
                self.logger.info_rank0(
                    f"[open_loop][debug] chunk: local_idx={local_idx} len(ds)={len(ds)} "
                    f"stride={stride} gt.shape={_g.shape}")
                self.logger.info_rank0(
                    f"[open_loop][debug]   GT物理 action[0,:4]={_g[0, :4]} "
                    f"[25,:4]={_g[25, :4]} [49,:4]={_g[49, :4]}")
                self.logger.info_rank0(
                    f"[open_loop][debug]   GT归一化 actions[0,:4]={_a[0, :4]} "
                    f"[25,:4]={_a[25, :4]} [49,:4]={_a[49, :4]}")
                self.logger.info_rank0(
                    f"[open_loop][debug]   GT逐维方差={np.var(_g, axis=0)}")
                if _pad is not None:
                    self.logger.info_rank0(
                        f"[open_loop][debug]   action_is_pad 前12={_pad[:12].astype(int)} "
                        f"后12={_pad[-12:].astype(int)}")
            # 取帧 + 形状校验 + 拼接：抽成模块级 `assemble_chunk`，与 Stage B0 的
            # Fixed Baseline 共用同一条口径（normalization / valid region / 拼接口径）。
            _assembled = assemble_chunk(gt_phys, pred, action_keys)
            if _assembled is None:
                continue
            gt, pr = _assembled
            ep_key = (int(ep_map[local_idx])
                      if ep_map is not None and local_idx < len(ep_map)
                      else f"chunk@{local_idx}")
            if self.dump_dir:
                os.makedirs(self.dump_dir, exist_ok=True)
                _p = os.path.join(self.dump_dir, f"{tag}_ep{local_idx}")
                np.save(_p + "_gt.npy", gt)
                np.save(_p + "_pred.npy", pr)
            chunks.append((ep_key, gt, pr))

        # ---- 按**完整 trajectory** 聚合（🔴 2026-10-04 审查 B2）----
        # 抽成纯函数 `aggregate_chunks` 是为了让 tools/open_loop_parity_check.py 能单测它。
        out = aggregate_chunks(chunks)
        # review v0.1 #9：把实际用到的 action keys 一并返回（供审计 action space）；
        # 之前只有 `collect_gt_chunks` 返回它，evaluator 侧拿不到。
        out["action_keys"] = list(action_keys)
        return out

    # -- TB --------------------------------------------------------------------
    def _write_tb(self, global_step: int, tr: Dict[str, float], va: Dict[str, float],
                  elapsed: float) -> None:
        """写 TB。

        ⚠️ **必须在 ``torch.inference_mode()`` 之外调用** —— inference_mode 里创建的
        tensor 会带 "inference tensor" 标记，一旦被 `AsyncTBWriter` 的后台线程拿去
        写盘/转 numpy 就会炸（"Inference tensors cannot be saved for backward" 一类）。
        所以这里只传 **Python float**，且调用点在 inference_mode 之外。
        """
        if self.args.train.global_rank != 0 or self.writer is None:
            return
        w = self.writer
        w.add_scalar("open_loop/train_mse", float(tr["mse"]), global_step)
        w.add_scalar("open_loop/train_mae", float(tr["mae"]), global_step)
        w.add_scalar("open_loop/train_r2", float(tr["r2"]), global_step)
        w.add_scalar("open_loop/val_mse", float(va["mse"]), global_step)
        w.add_scalar("open_loop/val_mae", float(va["mae"]), global_step)
        w.add_scalar("open_loop/val_r2", float(va["r2"]), global_step)
        # mean_baseline_mse = 该评测集上「每个维度都输出自己的常数均值」的 MSE
        # ⇒ 画成参考线：MSE 低于它才算学到。train / val 各自一份（评测集不同）
        w.add_scalar("open_loop/train_mean_baseline_mse",
                     float(tr.get("mean_baseline_mse", float("nan"))), global_step)
        w.add_scalar("open_loop/val_mean_baseline_mse",
                     float(va.get("mean_baseline_mse", float("nan"))), global_step)
        # 轨迹数 —— 官方口径的分母（`mse` 是对**轨迹**取平均，不是对 chunk）
        w.add_scalar("open_loop/train_n_traj", float(tr.get("n", 0)), global_step)
        w.add_scalar("open_loop/val_n_traj", float(va.get("n", 0)), global_step)
        w.add_scalar("open_loop/eval_seconds", float(elapsed), global_step)
        # 训练循环只在存档前 flush；eval 在存档之后 ⇒ 不主动 flush 的话曲线最多要等 750 步才可见
        try:
            w.flush()
        except Exception:  # noqa: BLE001
            pass

    # -- 对外入口 -------------------------------------------------------------
    def _run(self, global_step: int) -> Dict[str, Dict[str, float]]:
        """跑一次 train-monitor + val 评测并写 TB（不含任何状态切换，由 validate 负责）。"""
        t0 = time.time()
        # 每次评测都从同一个噪声序列起点开始 ⇒ 不同 step 之间的曲线可比（噪声不是变量）
        self._noise_gen = None
        with torch.inference_mode():
            tr = self._evaluate_ids(self.train_monitor_ids, "train_monitor")
            va = self._evaluate_ids(self.val_ids, "val")
        elapsed = time.time() - t0
        self.logger.info_rank0(
            f"[open_loop] step {global_step}: "
            f"train mse={tr['mse']:.4f} mae={tr['mae']:.4f} | "
            f"val mse={va['mse']:.4f} mae={va['mae']:.4f} | "
            f"baseline(global) train={tr['mean_baseline_mse']:.4f} "
            f"val={va['mean_baseline_mse']:.4f} | "
            f"r2(val)={va['r2']:+.3f} | "
            f"{tr['n']}轨迹/{tr['n_chunks']}chunk + {va['n']}轨迹/{va['n_chunks']}chunk，"
            f"{tr['frames']}+{va['frames']} 帧 × {va['dims']} 维 | {elapsed:.1f}s")
        # 逐条打印，供与官方 open_loop_eval.py 的 "MSE for trajectory <id>" 逐条对齐
        for _tag, _res in (("train", tr), ("val  ", va)):
            if not _res["per_traj_mse"]:
                continue
            self.logger.info_rank0(
                f"[open_loop] per-traj MSE {_tag}: "
                + " ".join(
                    f"ep{i}:{m:.4f}({f}帧)"
                    for i, m, f in zip(_res["per_traj_ids"],
                                       _res["per_traj_mse"],
                                       _res["per_traj_frames"])))
        self._write_tb(global_step, tr, va, elapsed)   # ← inference_mode 之外
        return {"train": tr, "val": va}

    # -- 安全评测上下文（Stage B0 抽出：validate / evaluate_ids 共用）-------------
    def _log_setup(self, rep: "EvalSetupReport") -> None:
        """把「本次临时改了什么」按需打印（每项只打一次）。"""
        if rep.use_cache_saved is not None and not self._use_cache_logged:
            self._use_cache_logged = True
            self.logger.info_rank0(
                f"[open_loop] eval 期间临时 use_cache: "
                f"{[old for _, old in rep.use_cache_saved]} → True"
                f"（共 {len(rep.use_cache_saved)} 个 config）")
        if rep.attn_saved is not None and not self._attn_logged:
            self._attn_logged = True
            self.logger.info_rank0(
                f"[open_loop] eval 期间临时 attention_implementation: "
                f"{[o for _, _, o, _ in rep.attn_saved]} → eager（{len(rep.attn_saved)} 个模块）；"
                f"对齐 deploy/lingbot_vla_v2_policy.py:290")
        if rep.visual_cache_saved is None:
            self.logger.warning(
                "[open_loop] ⚠️ 没找到视觉塔预计算网格缓存（config.precompute_grid_thw）；"
                "训练 micro>1 时评测可能因 visual_split_sizes 陈旧而报 split_with_sizes")
        elif not self._grid_cache_logged:
            self._grid_cache_logged = True
            _old = rep.visual_cache_saved["values"].get("visual_split_sizes")
            self.logger.info_rank0(
                f"[open_loop] 已清空视觉网格缓存"
                f"（{type(rep.visual_cache_saved['owner']).__name__}）："
                f"旧 visual_split_sizes={_old}")

    def _log_audit_ok(self) -> None:
        if not self._audit_logged:
            self._audit_logged = True
            self.logger.info_rank0(
                "[open_loop] ✅ 恢复审计通过：逐模块 training 标志 / config.use_cache / "
                "image_augment / compile 开关 / torch+numpy+python RNG 全部还原")

    def _eval_context(self) -> "EvalSetupReport":
        """共用的安全评测上下文（snapshot → 强制 → finally 恢复 → 审计）。"""
        return safe_eval_context(
            model=self.model,
            logger=self.logger,
            ft_aug_registry=self._ft_aug_orig,
            on_setup=self._log_setup,
            on_audit_ok=self._log_audit_ok,
        )

    def validate(self, global_step: int) -> Optional[Dict[str, Dict[str, float]]]:
        """在训练循环里调用：固定 seed、跑 eval、**任何情况下**恢复全部临时状态。

        Stage B0 起，临时的 5 类状态（use_cache / attention / 视觉网格缓存 /
        模块 training 标志 / 三套 RNG）由 `safe_eval_context` 统一处理 ——
        与 `evaluate_ids()` 走**同一条** snapshot-restore-审计路径。
        """
        if self.args.train.global_rank != 0:
            # 第一版只在 rank0 做；上多卡前这里要改成 "rank0 跑 + 其他 rank barrier"
            return None

        result = None
        with self._eval_context():
            try:
                result = self._run(global_step)
            except Exception as exc:  # noqa: BLE001
                # ⚠️ 只打一行 type+msg 会让 smoke 阶段无法定位（2026-10-04 实测：
                #    split_with_sizes 形状错只报一行，栈全丢）。这里必须打完整栈。
                self.logger.warning(f"[open_loop] ⚠️ step {global_step} 评测失败: "
                                    f"{type(exc).__name__}: {exc}")
                self.logger.warning("[open_loop] ---- traceback ----\n" + traceback.format_exc())
        return result

    # ------------------------------------------------------------------ #
    # Stage B0：给 Auto Learning 用的两个公开入口
    # ------------------------------------------------------------------ #
    def collect_gt_chunks(self, ids: Sequence[int], tag: str, *,
                          strict: bool = False
                          ) -> Tuple[List[Tuple[Any, np.ndarray]], List[str]]:
        """只收集 GT（**不推理**），供 Fixed Task Baseline 预计算使用。

        与 `_evaluate_ids` **共用同一条口径路径**：同样的回合白名单 →
        同样的 `per_episode_starts` 跳步 → 同样的 `ft.unapply` 反归一化 →
        同样的 `pick_action_keys` / `assemble_chunk`。
        ⇒ baseline 与 evaluator 的 action space / normalization / valid region
        **完全一致**（这是 Stage B0 的硬要求）。

        ``strict=True``（review v0.1 #6）：拿不到 ``local_idx→episode_index`` 映射时
        **直接抛错**。默认 ``False`` 保持与 `_evaluate_ids` 一致的「退化 + 警告」行为；
        但**算固定分母时必须 strict** —— 退化会让分母被永久污染。

        返回 ``(chunks, action_keys)``，其中 ``chunks = [(episode_key, gt(N,D)), ...]``。
        """
        ds_path = self._episode_ids_file(ids, tag)
        ds = self._dataset(ds_path)
        ft = self._ft_for(ds_path)
        ep_map = _episode_index_map(ds)
        if ep_map is None:
            _msg = (f"[auto_learning] ⚠️ [{tag}] 拿不到 local_idx→episode_index 映射，"
                    "会退化为「每个 chunk 当作一条 trajectory」")
            if strict:
                raise RuntimeError(
                    _msg + "\n  ⇒ strict=True 下拒绝继续：固定分母一旦退化就会被永久污染。")
            self.logger.warning(_msg)
        stride = max(1, int(getattr(self._model_config, "chunk_size", 50) or 50))
        if self.per_episode_stride and ep_map is not None:
            starts = per_episode_starts(ep_map, stride)
        else:
            starts = list(range(0, len(ds), stride))

        chunks: List[Tuple[Any, np.ndarray]] = []
        action_keys: List[str] = []
        for local_idx in starts:
            item = ds[local_idx]
            gt_phys = ft.unapply(dict(item))
            if not action_keys:
                # GT-only：pred 用 gt 自身占位 ⇒ 键集合 = org_features['actions'] ∩ GT 键
                action_keys = pick_action_keys(ft, gt_phys, gt_phys)
            _assembled = assemble_chunk(gt_phys, gt_phys, action_keys)
            if _assembled is None:
                continue
            gt, _ = _assembled
            ep_key = (int(ep_map[local_idx])
                      if ep_map is not None and local_idx < len(ep_map)
                      else f"chunk@{local_idx}")
            chunks.append((ep_key, gt))
        return chunks, action_keys

    def evaluate_ids(self, ids: Sequence[int], tag: str) -> Dict[str, Any]:
        """在**安全评测上下文**里对任意回合集合跑一次 open-loop 评测（**不写 TB**）。

        🔴 **不要**绕过本方法直接调 `_evaluate_ids()` —— 那会跳过 RNG /
        training flag / use_cache / attention / 视觉网格缓存的
        snapshot-restore-审计，污染训练状态。

        与 `validate()` 的区别只有三点：不写 TB、只跑给定的一个集合、失败**抛出**。
        """
        if self.model is None:
            raise RuntimeError(
                "evaluate_ids 需要真实模型（当前实例是离线 GT-only 模式，只能 collect_gt_chunks）")
        if getattr(self.args.train, "global_rank", 0) != 0:
            raise RuntimeError("open-loop 评测目前只在 rank0 上做（本模块只支持单卡）")

        with self._eval_context():
            try:
                with torch.inference_mode():
                    # 每次评测都从同一个噪声序列起点开始 ⇒ 不同 step 之间可比
                    self._noise_gen = None
                    return self._evaluate_ids(list(ids), tag)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"[open_loop] ⚠️ [{tag}] 评测失败: "
                                    f"{type(exc).__name__}: {exc}")
                self.logger.warning("[open_loop] ---- traceback ----\n" + traceback.format_exc())
                raise

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
* 多 rank：本模块只在 ``global_rank == 0`` 调用，其余 rank 需 barrier 等待。
  第一版单卡不做 barrier（见调用处注释），上多卡前必须补。
* ``sample_actions`` 的调用 glue 是**照抄** ``deploy/lingbot_vla_v2_policy.py`` 的
  ``PolicyPreprocessMixin.sample_actions_batch``（无法真正复用：那个 mixin 依赖一个
  inference policy 实例，构造它就要分配 25.5G，违背"不复制模型"）。
  **改动时务必与 deploy 逐行对齐**，靠"与官方 open_loop_eval.py 的一致性校验"兜底。
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
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
    snap = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def _rng_restore(snap: Dict[str, Any]) -> None:
    torch.set_rng_state(snap["cpu"])
    if "cuda" in snap:
        torch.cuda.set_rng_state_all(snap["cuda"])


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
# 主体
# ---------------------------------------------------------------------------
class OpenLoopValidator:
    """在训练进程内、用当前权重跑 open-loop 验证，结果写 TB。"""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        args,
        processor,
        use_depth_align: bool,
        writer,
        logger,
        train_monitor_ids: Optional[Sequence[int]] = None,
        val_ids: Optional[Sequence[int]] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.args = args
        self.processor = processor
        self.use_depth_align = use_depth_align
        self.writer = writer
        self.logger = logger
        self.device = device
        self.train_monitor_ids = list(train_monitor_ids or DEFAULT_TRAIN_MONITOR_IDS)
        self.val_ids = list(val_ids or DEFAULT_VAL_IDS)
        self._ds_cache: Dict[str, Any] = {}
        self._ft = None          # feature_transform（从 dataset 上取，与训练同一份）
        self._warned_augment = False
        self._shape_dumped = False   # 只在第一次 eval 时打印 item 的键名/形状
        self._chunk_dumped = False   # 只在第一次 eval 时打印 chunk 内动作是否随时间变化
        self._grid_cache_logged = False   # 只在第一次 eval 时打印被清掉的视觉网格缓存

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
        cfg = dataclasses.replace(self.args.data, episode_ids_file=episode_ids_file)
        # ⚠️ 官方 `scripts/open_loop_eval.py` 里这两个字段是**调用方**补上的
        #    （`policy.data_config.chunk_size = policy.config.chunk_size` /
        #      `policy.data_config.num_episode = None`）；`MyDataArguments` 本身不声明它们。
        #    漏了就会 AttributeError: 'MyDataArguments' object has no attribute 'chunk_size'
        #    （2026-10-04 smoke 实测）
        if not hasattr(cfg, "chunk_size"):
            setattr(cfg, "chunk_size", int(getattr(self.model.config, "chunk_size", 50)))
        if not hasattr(cfg, "num_episode"):
            setattr(cfg, "num_episode", None)
        ds = build_vla_dataset(
            dataset_config=cfg,
            model_config=self.args.model,
            config=self.model.config,
            processor=self.processor,
            use_depth_align=self.use_depth_align,
        )
        if self._ft is None:
            self._ft = _find_feature_transform(ds)
            if self._ft is None:
                raise RuntimeError(
                    f"从 {type(ds).__name__} 上找不到 feature_transform，无法反归一化"
                    f"（可用属性: {[a for a in dir(ds) if 'feature' in a or 'dataset' in a]}）")
        self._ds_cache[episode_ids_file] = ds
        return ds

    def _episode_ids_file(self, ids: Sequence[int], tag: str) -> str:
        """把一组回合号落成临时白名单文件（build_vla_dataset 只认文件）。"""
        out_dir = os.path.join(self.args.train.output_dir, "_open_loop_ids")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{tag}.json")
        with open(path, "w") as f:
            json.dump(sorted(int(i) for i in ids), f)
        return path

    # -- 推理（glue 照抄 deploy.PolicyPreprocessMixin.sample_actions_batch）----
    def _infer_one(self, item: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """对一条已 transform 的样本跑一次 ``model.sample_actions``，返回**物理量**动作块。"""
        use_bf16 = bool(getattr(self.args.train, "use_bf16", False))
        dtype = torch.bfloat16 if use_bf16 else torch.float32

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

        actions = self.model.sample_actions(
            images.to(dtype=dtype, device=self.device),
            img_masks.to(device=self.device),
            lang_tokens.to(device=self.device),
            lang_masks.to(device=self.device),
            state.to(dtype=dtype, device=self.device),
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
        return self._ft.unapply(single)

    # -- 动作键 ---------------------------------------------------------------
    def _pick_action_keys(self, gt_phys: Dict[str, Any], pred: Dict[str, Any]) -> List[str]:
        """挑出「预测与 GT 都有」的动作键。

        官方 ``scripts/open_loop_eval.py`` 读的是
        ``feature_transform.org_features['actions']``（原始 robot-config 键名），
        这里以它为准，再与实际 unapply 返回的键取交集兜底。
        """
        both = {k for k in gt_phys if k in pred}
        org = list(getattr(self._ft, "org_features", {}).get("actions", []) or [])
        keys = [k for k in org if k in both]
        if not keys:
            keys = sorted(k for k in both if str(k).startswith("action."))
        if not keys:
            keys = sorted(both)
        return keys

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
        ds = self._dataset(self._episode_ids_file(ids, tag))

        per_traj: List[tuple] = []      # (mse_i, mae_i, baseline_i, frames_i, dims_i)
        gt_chunks, pred_chunks = [], []
        action_keys: List[str] = []
        # 与官方 ``scripts/open_loop_eval.py`` 对齐：``chunk_ret=True`` 时按
        # ``action_horizon = chunk_size`` 步进（官方 ``range(start_id, end_id, action_horizon)``），
        # 每次推理用**整段** chunk。
        # ⚠️ 逐帧（stride=1）会把同一批帧反复预测 ~chunk_size 次：一条 77 帧的轨迹要
        #    77 次推理（2026-10-04 smoke 实测单次推理数秒级）⇒ 完全做不到"高频"，
        #    而且**口径与官方不一致**（官方一条轨迹只推 2 次）。
        stride = max(1, int(getattr(self.model.config, "chunk_size", 50) or 50))
        for local_idx in range(0, len(ds), stride):
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
            gt_phys = self._ft.unapply(dict(item))
            pred = self._infer_one(item)
            if not action_keys:
                action_keys = self._pick_action_keys(gt_phys, pred)
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
            g_parts, p_parts, n_frames = [], [], None
            for key in action_keys:
                g = _to_numpy(gt_phys[key])
                p = _to_numpy(pred[key])
                # 统一成 (帧, 维)；1 维动作视作 (帧, 1)
                g = g.reshape(g.shape[0], -1) if g.ndim > 1 else g.reshape(-1, 1)
                p = p.reshape(p.shape[0], -1) if p.ndim > 1 else p.reshape(-1, 1)
                n = min(g.shape[0], p.shape[0])      # 按**帧**对齐，不能按压平后的元素对齐
                g_parts.append(g[:n])
                p_parts.append(p[:n])
                n_frames = n if n_frames is None else min(n_frames, n)
            if not g_parts or not n_frames:
                continue
            gt = np.concatenate([x[:n_frames] for x in g_parts], axis=1)     # (N_i, D)
            pr = np.concatenate([x[:n_frames] for x in p_parts], axis=1)
            err = pr - gt
            per_traj.append((
                float(np.mean(err ** 2)),
                float(np.mean(np.abs(err))),
                float(np.var(gt, axis=0).mean()),
                int(gt.shape[0]),
                int(gt.shape[1]),
            ))
            gt_chunks.append(gt)
            pred_chunks.append(pr)

        nan = float("nan")
        if not per_traj:
            return {"n": 0, "frames": 0, "dims": 0,
                    "mse": nan, "mae": nan, "r2": nan,
                    "mse_pooled": nan, "mean_baseline_mse": nan,
                    "mean_baseline_mse_per_traj": nan,
                    "per_traj_mse": [], "per_traj_mae": []}

        # A) 官方口径：逐 trajectory 算，再对 trajectory 简单平均
        mse = float(np.mean([x[0] for x in per_traj]))
        mae = float(np.mean([x[1] for x in per_traj]))
        mean_baseline_mse_per_traj = float(np.mean([x[2] for x in per_traj]))

        # B) 离线口径：所有帧 pool 在一起，逐维方差 → 对各维取均值
        gt_all = np.concatenate(gt_chunks, axis=0)       # (N, D)
        pr_all = np.concatenate(pred_chunks, axis=0)
        err_all = pr_all - gt_all
        mse_pooled = float(np.mean(err_all ** 2))
        mean_baseline_mse = float(np.var(gt_all, axis=0).mean())

        r2 = (float(1.0 - mse_pooled / mean_baseline_mse)
              if mean_baseline_mse > 0 else nan)
        return {
            "n": len(per_traj),
            "frames": int(sum(x[3] for x in per_traj)),
            "dims": per_traj[0][4],
            "mse": mse,                                       # 官方口径
            "mae": mae,                                       # 官方口径
            "mse_pooled": mse_pooled,                         # 离线口径
            "mean_baseline_mse": mean_baseline_mse,           # global constant baseline（离线口径）
            "mean_baseline_mse_per_traj": mean_baseline_mse_per_traj,
            "r2": r2,                                         # = 1 - mse_pooled / mean_baseline_mse
            "per_traj_mse": [x[0] for x in per_traj],         # 供与官方逐条对齐
            "per_traj_mae": [x[1] for x in per_traj],
        }

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
            f"{tr['frames']}+{va['frames']} 帧 × {va['dims']} 维 | {elapsed:.1f}s")
        # 逐条打印，供与官方 open_loop_eval.py 的 "MSE for trajectory <id>" 逐条对齐
        self.logger.info_rank0(
            "[open_loop] per-traj MSE train: "
            + " ".join(f"{x:.4f}" for x in tr["per_traj_mse"]))
        self.logger.info_rank0(
            "[open_loop] per-traj MSE val  : "
            + " ".join(f"{x:.4f}" for x in va["per_traj_mse"]))
        self._write_tb(global_step, tr, va, elapsed)   # ← inference_mode 之外
        return {"train": tr, "val": va}

    def validate(self, global_step: int) -> Optional[Dict[str, Dict[str, float]]]:
        """在训练循环里调用：固定 seed、跑 eval、**任何情况下**恢复全部临时状态。"""
        if self.args.train.global_rank != 0:
            # 第一版只在 rank0 做；上多卡前这里要改成 "rank0 跑 + 其他 rank barrier"
            return None

        rng_snap = _rng_snapshot()
        train_flags = _module_training_flags(self.model)
        compile_flag = getattr(self.model, "_use_compile_predict_velocity", None)
        ft_aug = None
        visual_cache_saved = None
        result = None
        try:
            # ① 强制 eager：避免拿到训练用的编译产物；也避免 inference tensor 逃逸进 compile cache
            if compile_flag is not None:
                self.model._use_compile_predict_velocity = False
                self.model._compiled_predict_velocity = None
            # ② 固定 seed（整次 eval 一致 ⇒ 不同 step 可比）
            torch.manual_seed(EVAL_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(EVAL_SEED)
            # ③ 逐模块 eval（不是 model.eval() 一刀切，但那也可以：下面是等价写法）
            self.model.eval()
            # ③b 释放训练侧遗留的碎片/缓存 —— 单卡余量本来就紧，不清的话
            #     sample_actions 自己的激活 + KV cache 可能装不下
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # ③c 🔴 清掉 Qwen3-VL 视觉塔的「预计算网格缓存」（2026-10-04 micro=10 实测）
            #   `modeling_lingbot_vla_v2.py::get_image_features` 里这 5 个属性是
            #   **首次调用时按当时的 grid_thw 缓存**的（`precompute_grid_thw: true`）：
            #       if precompute_grid_thw and self.position_embeddings is None:  <-- 只在这里重算
            #           ... = self.qwenvl.visual.preprcess_grid_thw(grid_thw)
            #       split_sizes = self.visual_split_sizes
            #   训练 batch 的网格 = micro 张图 × 相机数（micro=10 ⇒ 30），
            #   评测单样本 = 1 × 3 ⇒ 缓存的 split_sizes 是 `[64]*30`，
            #   而 `torch.split(image_embeds(192), [64]*30)` 直接
            #       RuntimeError: split_with_sizes expects split_sizes to sum exactly to 192
            #   （micro=1 时训练网格恰好也是 3 ⇒ 巧合躲过，所以只在 micro>1 暴露）
            #   做法：评测前全部置 None（强制按评测网格重算），评测后原样恢复。
            visual_cache_saved = _visual_grid_cache_clear(self.model)
            if visual_cache_saved is None:
                self.logger.warning(
                    "[open_loop] ⚠️ 没找到视觉塔预计算网格缓存（config.precompute_grid_thw）；"
                    "训练 micro>1 时评测可能因 visual_split_sizes 陈旧而报 split_with_sizes")
            elif not self._grid_cache_logged:
                self._grid_cache_logged = True
                _old = visual_cache_saved["values"].get("visual_split_sizes")
                self.logger.info_rank0(
                    f"[open_loop] 已清空视觉网格缓存（{type(visual_cache_saved['owner']).__name__}）："
                    f"旧 visual_split_sizes={_old}")
            # ④ 关掉图像增强（保险；本项目 image_augment 本来就是 false）
            if self._ft is not None and hasattr(self._ft, "image_augment"):
                ft_aug = self._ft.image_augment
                self._ft.image_augment = False
            result = self._run(global_step)
        except Exception as exc:  # noqa: BLE001
            # ⚠️ 只打一行 type+msg 会让 smoke 阶段无法定位（2026-10-04 实测：
            #    split_with_sizes 形状错只报一行，栈全丢）。这里必须打完整栈。
            self.logger.warning(f"[open_loop] ⚠️ step {global_step} 评测失败: "
                                f"{type(exc).__name__}: {exc}")
            self.logger.warning("[open_loop] ---- traceback ----\n" + traceback.format_exc())
        finally:
            # 全部临时状态恢复（顺序与设置相反）
            _visual_grid_cache_restore(self.model, visual_cache_saved)
            if ft_aug is not None:
                self._ft.image_augment = ft_aug
            _module_training_restore(train_flags)
            if compile_flag is not None:
                self.model._use_compile_predict_velocity = compile_flag
                self.model._compiled_predict_velocity = None
            _rng_restore(rng_snap)
        return result

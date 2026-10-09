"""save / restore（文档 §43.9）。

resume 时**只恢复模型 checkpoint 是不够的**。要恢复到「走出同一条路线」，
至少还需要：current task / status / attempt_count / round / best_nmse /
pass snapshot / hardness version / RNG state / 世界状态。

另外还存了一份**配置指纹**：如果 resume 时关键语义参数变了（`pass_nmse` /
`max_attempts_per_task` / 任务列表 …），恢复出来的路线就没法解释了 ——
这种情况必须**报错要求显式确认**，不能悄悄继续（测试方案 §G04）。
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..sampling.hardness_scan import HardnessScan

STATE_VERSION = 4

#: 事件流最多存档这么多条（够重建学习顺序，又不至于让 state.json 无限膨胀）
MAX_EVENTS = 2000

#: 这些参数一变，历史状态的含义就变了 ⇒ resume 前必须显式确认
SEMANTIC_KEYS = (
    "pass_nmse",
    # 判定口径（"nmse" / "mse"）与阈值表路径 —— 换了它们，历史状态里"通过"的含义就变了。
    # ⚠️ 新增这两项会让**旧存档**的指纹对不上 ⇒ resume 时要显式确认一次（这是有意的：
    #    宁可提醒，也不要带着新口径跑旧状态）。
    "pass_metric",
    "pass_thresholds_file",
    "min_lp50",
    "max_attempts_per_task",
    "max_reopens_per_task",
    "eval_interval_steps",
    "min_steps_before_defer",
    "batch_size",
    "new_slots",
    "replay_slots",
    "hardness_probe_fraction",
    "hardness_weight_min",
    "hardness_weight_max",
    "hardness_alpha",
    "review_after_task_transitions",
    "forget_relative_threshold",
    "continue_after_pass",
    "post_pass_max_steps",
    "post_pass_min_lp",
    "early_defer_on_overfit",
    "overfit_train_lp_min",
    "overfit_val_lp_max",
    "overfit_gap_growth_threshold",
    "difficulty_unscored_default",
    "refresh_hardness_on_continue",
    "defer_resample_retry",
    "defer_retry_steps",
    "replay_task_policy",
    "replay_sample_policy",
    "rescan_candidates_after_transition",
    "max_new_tasks_attempted_this_run",
    "max_new_tasks_passed_this_run",
    "min_new_tasks_passed_this_run",
    "target_total_passed_tasks",
    "global_scout_val_trajs",
    "active_val_probe_trajs",
    "active_train_probe_trajs",
)


def config_fingerprint(al: Any, task_names: Sequence[str]) -> Dict[str, Any]:
    # 兼容旧 DCP：新增的可选总目标为 None 时，不能凭空多出一个指纹键。
    # 否则只升级代码不换策略也会触发 resume 的语义不一致错误。
    fp: Dict[str, Any] = {k: getattr(al, k, None) for k in SEMANTIC_KEYS
                          if k != "target_total_passed_tasks"
                          or getattr(al, k, None) is not None}
    fp["task_names"] = list(task_names)
    if getattr(al, "pass_metric", "nmse") == "gmean_mse":
        # Fresh GMean experiment controls how Scout can produce PASS. Changing
        # Confirm behavior must invalidate exact resume, without touching legacy
        # NMSE/MSE checkpoint fingerprints.
        fp["gmean_scout_confirm_enabled"] = bool(getattr(al, "scout_confirm_enabled", True))
        # A same-path threshold rewrite must invalidate exact Resume: historical
        # PASS/DEFER/REOPEN semantics depend on the *contents*, not just its path.
        # Only for this new mode, so old NMSE/MSE DCP fingerprints stay identical.
        import hashlib
        from ..decision.thresholds import attached_thresholds

        table = attached_thresholds(al)
        blob = (json.dumps(table.to_dict(), ensure_ascii=False, sort_keys=True,
                           allow_nan=False, separators=(",", ":"))
                if table is not None else None)
        fp["gmean_thresholds_sha256"] = (
            hashlib.sha256(blob.encode("utf-8")).hexdigest() if blob is not None else None)
    return fp


def diff_config(old: Dict[str, Any], new: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """返回 [(键, 存档值, 当前值), …]，只列有变化的。"""
    out: List[Tuple[str, Any, Any]] = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if a != b:
            out.append((key, a, b))
    return out


# --------------------------------------------------------------------------- #
def _rng_to_state(rng: random.Random) -> List[Any]:
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def _rng_from_state(raw: List[Any]) -> random.Random:
    rng = random.Random()
    rng.setstate((int(raw[0]), tuple(int(x) for x in raw[1]), raw[2]))
    return rng


# --------------------------------------------------------------------------- #
def collect_state(sched) -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "config": config_fingerprint(sched.al, sched.registry.names()),
        "rng_state": _rng_to_state(sched.rng),
        # 额外状态（Stage A 的假世界）；Stage B 为 None（模型状态在 checkpoint 里）
        "extra_state": sched.extra_state.to_state() if sched.extra_state else None,
        "registry": sched.registry.to_state(),
        "scheduler": sched.state.to_state(),
        "scans": {k: v.to_state() for k, v in sched.scans.items()},
        "metrics_rows": sched.metrics_rows,
        "heatmap_rows": sched.heatmap_rows,
        # 🔴 事件流也必须随状态走：否则 resume 之后内存里的 events 只剩后半段，
        # 「用 trace 重建学习顺序」这条性质就断了（`compare_resume_run` 抓出来的）。
        "events": sched.events[-MAX_EVENTS:],
    }


def restore_state(sched, raw: Dict[str, Any], *, allow_config_change: bool = False) -> None:
    version = int(raw.get("version", 0))
    if version != STATE_VERSION:
        raise ValueError(
            f"state version 不匹配：文件 {version} vs 代码 {STATE_VERSION}（不要混用旧存档）"
        )

    # ---- 配置兼容性（测试方案 §G04）----
    old_cfg = raw.get("config")
    if old_cfg:
        changes = diff_config(old_cfg, config_fingerprint(sched.al, sched.registry.names()))
        if changes and not allow_config_change:
            lines = "\n".join(f"    {k}: 存档={a!r} → 当前={b!r}" for k, a, b in changes)
            raise ValueError(
                "resume 的关键配置与存档不一致，拒绝静默继续：\n"
                f"{lines}\n"
                "  ⇒ 确认要改的话，显式传 allow_config_change=True / --allow-config-change。\n"
                "     注意：改了 pass_nmse / max_attempts 这类参数后，历史状态的含义会变，"
                "恢复出来的路线不可解释。"
            )

    sched.rng = _rng_from_state(raw["rng_state"])
    # 让所有持有 rng 引用的组件指向恢复出来的同一个对象（唯一权威随机源）
    if hasattr(sched, "rebind_rng"):
        sched.rebind_rng()

    if sched.extra_state is not None and raw.get("extra_state") is not None:
        sched.extra_state.load_state(raw["extra_state"])
    sched.registry.load_state(raw["registry"])
    sched.state.load_state(raw["scheduler"])
    # ---- 兼容（审查 D7，用户 10-08 18:5x 口径）：`bootstrap_passed` 是补丁新增字段。
    # 旧 DCP 里**没有这个键** ⇒ **不推断、不伪造**（`auto_passed` 可能含 Rescan 免费 PASS，
    # 拿它当 Bootstrap 会让 `bootstrap_pass_count` 失真）⇒ 只把来源标记为不可用，
    # 由统计端显式输出 unavailable。收工判定走 registry 当前状态，不受影响。
    _raw_sched = raw.get("scheduler") or {}
    if "bootstrap_passed" not in _raw_sched:
        sched.state.bootstrap_passed_available = False
    sched.scans = {k: HardnessScan.from_state(v) for k, v in (raw.get("scans") or {}).items()}
    sched.metrics_rows = list(raw.get("metrics_rows") or [])
    sched.heatmap_rows = list(raw.get("heatmap_rows") or [])
    sched.events = list(raw.get("events") or [])


def save_state(sched, path: str) -> None:
    payload = collect_state(sched)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, path)


def load_state(sched, path: str, *, allow_config_change: bool = False) -> None:
    with open(path, "r", encoding="utf-8") as fh:
        restore_state(sched, json.load(fh), allow_config_change=allow_config_change)

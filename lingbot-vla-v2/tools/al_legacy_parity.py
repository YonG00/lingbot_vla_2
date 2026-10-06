#!/usr/bin/env python3
"""§4 + §7–§10 —— Legacy vs Integration 的**数据/标签/norm-stats/collator 对拍**。

用法（在**任意** checkout 里跑，把结果写成 JSON）::

    python tools/al_legacy_parity.py --repo /data/code/lingbot-vla-v2 \
        --config /data/outputs/single/click_bell/lingbotvla_cli.yaml \
        --episode-ids /data/train/task_splits/click_bell.val_ids.json \
        --out /data/tmp/parity_integration.json

然后::

    python tools/al_legacy_parity.py --diff /data/tmp/parity_legacy.json \
        /data/tmp/parity_integration.json

对拍内容（规范 §7/§8/§9/§10）：
  * **effective config**（§8）：norm_stats_file 路径+内容哈希、chunk_size、cameras、
    action/state 维数、img_size、chunk 键名
  * **dataset item**（§7）：固定 (episode, 绝对帧号) 上 item 的 key set / shape / dtype /
    actions·state·joint_mask·action_is_pad 的**逐值哈希**
  * **batch schema**（§10）：把 N 个 item 过一遍真实 collator，dump 出的 batch 的
    key set / shape / dtype / valid 计数

⚠️ 对拍**不允许用数值容差**（规范 §28）：数据/ID/label/mask 必须**严格一致**。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path


def _sha(a) -> str:
    """张量的逐值哈希（先转 numpy 再 hash bytes）。"""
    import numpy as np

    x = a
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(x.tobytes()).hexdigest()[:16] + f":{x.shape}:{x.dtype}"


def _describe(item: dict) -> dict:
    out = {"keys": sorted(str(k) for k in item)}
    for k in ("actions", "state", "joint_mask", "action_is_pad",
              "lang_tokens", "lang_masks", "img_masks", "images",
              "action_joint_mask", "state_joint_mask", "image_grid_thw"):
        if k not in item:
            out[k] = None
            continue
        v = item[k]
        try:
            out[k] = _sha(v)
        except Exception as exc:  # noqa: BLE001
            out[k] = f"<{type(v).__name__}: {exc}>"
    return out


def _effective_config(args_ns, cfg, ft) -> dict:
    """§8：以**运行时最终解析出来的** effective config 为准（不是比 YAML 文字）。"""
    from lingbotvla.auto_learning.baseline import file_sha256

    ns = getattr(ft, "normalizer", None)
    stats = getattr(ns, "norm_stats", None) or {}
    key = "observation.state.arm.position"
    mean_fp = None
    try:
        import numpy as np
        m = stats.get(key, {}).get("mean")
        if m is not None:
            mean_fp = [round(float(x), 6) for x in np.asarray(m).reshape(-1)[:3]]
    except Exception:  # noqa: BLE001
        pass
    return {
        "norm_stats_file": getattr(args_ns, "norm_stats_file", None),
        "norm_stats_content_sha": file_sha256(getattr(args_ns, "norm_stats_file", None) or ""),
        "norm_stats_count": (stats.get("observation.state.arm.position", {}) or {}).get("count"),
        "norm_stats_mean_fp": mean_fp,
        "chunk_size": int(getattr(cfg, "chunk_size", 0)),
        "n_action_steps": int(getattr(cfg, "n_action_steps", 0)),
        "max_action_dim": int(getattr(cfg, "max_action_dim", 0)),
        "max_state_dim": int(getattr(cfg, "max_state_dim", 0)),
        "action_dim": int(getattr(cfg, "action_dim", 0)),
        "cameras": list(getattr(args_ns, "cameras", None) or []),
        "joints": list(getattr(args_ns, "joints", None) or []),
        "img_size": getattr(args_ns, "img_size", None),
        "image_augment": bool(getattr(args_ns, "image_augment", False)),
        "action_keys": sorted(getattr(ft, "actions", []) or []),
        "state_keys": sorted(getattr(ft, "states", []) or []),
        "image_keys": sorted(getattr(ft, "images", []) or []),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="要测的 checkout（默认本脚本所在仓库）")
    ap.add_argument("--config", default=None, help="训练用 lingbotvla_cli.yaml")
    ap.add_argument("--episode-ids", default=None, help="回合白名单 json")
    ap.add_argument("--frames", type=int, default=3, help="每个回合取几个帧（首/中/尾）")
    ap.add_argument("--batch", type=int, default=4, help="collator 对拍的样本数")
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", nargs=2, default=None, metavar=("LEGACY", "INTEG"))
    a = ap.parse_args()

    if a.diff:
        return _diff(a.diff[0], a.diff[1])

    repo = Path(a.repo or Path(__file__).resolve().parents[1]).resolve()
    sys.path.insert(0, str(repo))
    print(f"[parity] repo = {repo}", flush=True)

    import yaml

    from lingbotvla.auto_learning.model_config import load_config_and_processor
    from lingbotvla.data.dataset import build_vla_dataset
    from lingbotvla.utils.open_loop_validation import _find_feature_transform

    cfg, processor, raw = load_config_and_processor(a.config)
    ns = types.SimpleNamespace(**dict(raw.get("data", {})))
    if not hasattr(ns, "chunk_size"):
        ns.chunk_size = int(cfg.chunk_size)
    if not hasattr(ns, "num_episode"):
        ns.num_episode = None
    if a.episode_ids:
        ns.episode_ids_file = a.episode_ids

    ds = build_vla_dataset(dataset_config=ns, model_config=cfg, config=cfg,
                           processor=processor)
    ft = _find_feature_transform(ds)
    print(f"[parity] dataset len = {len(ds)}；config = {a.config}", flush=True)

    # ---- §7：固定 (episode, 绝对帧号) 上的 item 对拍 ----
    hf = ds._datasets[0].dataset.hf_dataset if getattr(ds, "_datasets", None) else ds.dataset.hf_dataset
    eps = [int(x) for x in hf["episode_index"]]
    frames = [int(x) for x in hf["index"]]
    order: dict[int, list[int]] = {}
    for i, e in enumerate(eps):
        order.setdefault(e, []).append(i)
    items = {}
    for ep, locs in sorted(order.items()):
        if a.frames == 1:
            picks = [locs[len(locs) // 2]]
        else:
            picks = [locs[0], locs[len(locs) // 2], locs[-1]][: a.frames]
        for li in picks:
            items[f"ep{ep}_abs{frames[li]}"] = _describe(ds[li])

    # ---- §10：真实 collator 后的 batch schema ----
    from lingbotvla.data.data_collator import CollatePipeline, DataCollatorWithPadding
    collate = CollatePipeline([DataCollatorWithPadding()])
    idx0 = [locs[0] for _, locs in sorted(order.items())][: a.batch]
    raw_items = [ds[i] for i in idx0]
    batch = collate(raw_items)
    try:
        from lingbotvla.auto_learning.real.sanity import audit_batch_schema
    except Exception:  # noqa: BLE001 —— legacy 树里没有这个模块 ⇒ 用等价的内联版
        audit_batch_schema = _audit_inline
    schema = audit_batch_schema(batch)

    out = {
        "repo": str(repo),
        "commit": _git(repo),
        "config": a.config,
        "episode_ids_file": a.episode_ids,
        "dataset_len": len(ds),
        "effective_config": _effective_config(ns, cfg, ft),
        "items": items,
        "batch_schema": schema,
    }
    p = a.out or "parity.json"
    Path(p).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
    print(f"[parity] 写出 {p}", flush=True)
    print(f"[parity] norm_stats mean[:3] = {out['effective_config']['norm_stats_mean_fp']}")
    print(f"[parity] item 数 = {len(items)}；batch 键数 = {len(schema.get('_keys_present', []))}")
    return 0


def _audit_inline(batch: dict) -> dict:
    """`real/sanity.py` 不可用时的等价内联版（legacy 树里没有那个模块）。"""
    import numpy as np

    keys = ("actions", "action", "state", "states", "joint_mask", "action_is_pad",
            "input_ids", "lang_tokens", "lang_masks", "img_masks", "images",
            "image_grid_thw")
    rep = {}
    for k in keys:
        if k not in batch:
            continue
        v = batch[k]
        try:
            x = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        except Exception:  # noqa: BLE001
            rep[k] = {"present": True, "kind": type(v).__name__}
            continue
        info = {"present": True, "shape": tuple(int(t) for t in np.shape(x)),
                "dtype": str(getattr(x, "dtype", "?")), "numel": int(np.size(x))}
        if x.size and np.issubdtype(x.dtype, np.number):
            info["nonzero"] = int(np.count_nonzero(x))
            if "mask" in k or k == "action_is_pad":
                info["valid"] = info["nonzero"]
        rep[k] = info
    rep["_keys_present"] = sorted(str(k) for k in batch)
    return rep


def _git(repo: Path):
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(repo), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _diff(legacy_path: str, integ_path: str) -> int:
    """§28：**严格一致**（不给数值容差）。"""
    lg = json.loads(Path(legacy_path).read_text(encoding="utf-8"))
    it = json.loads(Path(integ_path).read_text(encoding="utf-8"))
    bad: list[str] = []

    print(f"legacy      : {lg['repo']} @ {lg.get('commit')}")
    print(f"integration : {it['repo']} @ {it.get('commit')}")

    # §8 effective config
    for k, v in lg["effective_config"].items():
        w = it["effective_config"].get(k)
        if v != w:
            bad.append(f"[§8 config] {k}: legacy={v!r} vs integ={w!r}")
    # §7 item
    if set(lg["items"]) != set(it["items"]):
        bad.append(f"[§7] 抽到的 item 集合不同: "
                   f"{sorted(set(lg['items']) ^ set(it['items']))[:5]}")
    for name in sorted(set(lg["items"]) & set(it["items"])):
        a, b = lg["items"][name], it["items"][name]
        if a["keys"] != b["keys"]:
            bad.append(f"[§7 {name}] key set 不同: "
                       f"legacy-only={sorted(set(a['keys']) - set(b['keys']))[:5]} "
                       f"integ-only={sorted(set(b['keys']) - set(a['keys']))[:5]}")
        for k in a:
            if k == "keys":
                continue
            if a[k] != b[k]:
                bad.append(f"[§7 {name}] {k}: {a[k]} vs {b[k]}")
    # §10 batch schema
    la, lb = lg["batch_schema"], it["batch_schema"]
    if la.get("_keys_present") != lb.get("_keys_present"):
        bad.append(f"[§10] batch key 集合不同: {la.get('_keys_present')} vs {lb.get('_keys_present')}")
    for k in sorted(set(la) & set(lb)):
        if k.startswith("_"):
            continue
        for f in ("shape", "dtype"):
            if la[k].get(f) != lb[k].get(f):
                bad.append(f"[§10] {k}.{f}: {la[k].get(f)} vs {lb[k].get(f)}")

    print()
    if bad:
        print(f"❌ 不一致 {len(bad)} 处（**P0**：数据/标签/mask/norm-stats 必须严格一致）")
        for b in bad[:40]:
            print("  -", b)
        return 1
    print("✅ 完全一致（§7 item / §8 effective config / §10 batch schema）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

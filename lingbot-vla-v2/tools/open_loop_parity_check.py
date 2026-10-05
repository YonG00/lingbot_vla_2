#!/usr/bin/env python3
"""开环评测「与官方口径一致」自检 —— 纯 CPU、无卡可跑、**不加载模型**

为什么需要它
------------
`OpenLoopValidator` 的指标要和官方 `scripts/open_loop_eval.py` 可比，必须同时满足：

  ① **评测帧集**一致：每个回合各自从**自己的首帧**跳步（官方
     `range(start_id, end_id, action_horizon)`）
  ② **聚合口径**一致：按**完整 trajectory** 把 chunk 拼起来算 MSE，再对轨迹简单平均
  ③ padding **不被额外排除**（官方也不排除）
  ④ 前向链路一致（数值层 —— 需 GPU，见 `tools/open_loop_parity_test.sh`）

本脚本只用**数据集元信息** + **被测模块的纯函数**验证 ①②③，秒级完成、不碰 GPU。
④ 的逐值对拍由 `tools/open_loop_parity_test.sh`（需 GPU + 一个 ckpt）负责。

设计要点：**判定用的"官方参考实现"在本脚本里独立重写**，不调用被测模块 ——
否则就是自己跟自己比。P4 还会同时算出「按 chunk 平均」这个**错误口径**，
以证明本测试**有区分力**（不是恒真的空转）。

用法
----
    python tools/open_loop_parity_check.py --task click_bell
    python tools/open_loop_parity_check.py --task click_bell --horizon 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATASET = "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
SPLIT_DIR = Path("/data/train/task_splits")

_RESULTS: list[tuple[str, bool, str]] = []


def run(cid: str, title: str, fn) -> None:
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    _RESULTS.append((f"{cid} {title}", bool(ok), str(detail)))
    print(f"[parity] {'OK  ' if ok else 'FAIL'} {cid} {title}")
    for line in str(detail).splitlines():
        print(f"           {line}")
    print()


# ---------------------------------------------------------------------------
# 官方参考实现（本脚本独立重写，**不调用被测模块**）
# ---------------------------------------------------------------------------
def official_abs_starts(meta, traj_ids: list[int], horizon: int) -> list[int]:
    """官方 `evaluate_single_trajectory` 的步进：
    `range(meta.episodes[traj]["dataset_from_index"], [...,"dataset_to_index"], action_horizon)`。
    """
    out: list[int] = []
    for t in traj_ids:
        a = int(meta.episodes[t]["dataset_from_index"])
        b = int(meta.episodes[t]["dataset_to_index"])
        out.extend(range(a, b, horizon))
    return out


def official_mse_from_chunks(chunks: list[tuple]) -> tuple[float, int]:
    """官方聚合：每条轨迹先把 chunk 沿时间拼成整条，算一个 MSE，再对轨迹简单平均。"""
    by_ep: dict = {}
    for k, g, p in chunks:
        by_ep.setdefault(k, []).append((g, p))
    per = []
    for lst in by_ep.values():
        g = np.concatenate([x for x, _ in lst], axis=0)
        p = np.concatenate([y for _, y in lst], axis=0)
        per.append(float(np.mean((p - g) ** 2)))
    return float(np.mean(per)), len(per)


def wrong_per_chunk_mse(chunks: list[tuple]) -> float:
    """**错误口径**：把每个 chunk 当一条独立 trajectory 再平均（旧实现）。"""
    return float(np.mean([float(np.mean((p - g) ** 2)) for _, g, p in chunks]))


# ---------------------------------------------------------------------------
# P1 评测帧集一致（真数据、非循环）
# ---------------------------------------------------------------------------
def p1(task: str, horizon: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset as RepoDs
    from lingbotvla.utils.open_loop_validation import per_episode_starts

    ids = json.load(open(SPLIT_DIR / f"{task}.val_ids.json"))
    meta = LeRobotDatasetMetadata(Path(DATASET).name, root=DATASET)

    ds = RepoDs(Path(DATASET).name, root=DATASET, episodes=list(ids))
    # local_idx → 绝对帧号（数据集的 `index` 列）；与 `_episode_index_map` 读的同一份 hf_dataset
    abs_col = np.asarray([int(x) for x in ds.hf_dataset["index"]], dtype=np.int64)
    ep_col = np.asarray([int(x) for x in ds.hf_dataset["episode_index"]], dtype=np.int64)
    if len(abs_col) != len(ds):
        return False, f"index 列长度 {len(abs_col)} ≠ len(ds) {len(ds)}"

    ours_local = per_episode_starts(ep_col, horizon)
    ours_abs = [int(abs_col[i]) for i in ours_local]
    ref_abs = official_abs_starts(meta, list(ids), horizon)

    if sorted(ours_abs) != sorted(ref_abs):
        only_ours = sorted(set(ours_abs) - set(ref_abs))[:6]
        only_ref = sorted(set(ref_abs) - set(ours_abs))[:6]
        return False, (f"帧集不一致：我们 {len(ours_abs)} 个起点 / 官方 {len(ref_abs)} 个；"
                       f"仅我们有 {only_ours}；仅官方有 {only_ref}")

    # 逐回合 chunk 数
    per_ep_ours: dict = {}
    for i in ours_local:
        per_ep_ours[int(ep_col[i])] = per_ep_ours.get(int(ep_col[i]), 0) + 1
    per_ep_ref: dict = {}
    for t in ids:
        a = int(meta.episodes[t]["dataset_from_index"])
        b = int(meta.episodes[t]["dataset_to_index"])
        per_ep_ref[int(t)] = len(range(a, b, horizon))
    bad = {k: (per_ep_ours.get(k, 0), per_ep_ref[k])
           for k in per_ep_ref if per_ep_ours.get(k, 0) != per_ep_ref[k]}
    if bad:
        return False, f"逐回合 chunk 数不一致：{bad}"

    # 有多少起点落在回合首帧（旧口径下这个数会掉到 1）
    ep_first = {int(t): int(meta.episodes[t]["dataset_from_index"]) for t in ids}
    n_at_first = sum(1 for a in ours_abs if a in set(ep_first.values()))
    return True, (f"起点集合完全一致（{len(ours_abs)} 个）；逐回合 chunk 数一致；"
                  f"其中 {n_at_first}/{len(ids)} 个回合的**首帧**被覆盖（旧口径只有 1 个）")


# ---------------------------------------------------------------------------
# P2 旧口径会分叉（证明 P1 有区分力）
# ---------------------------------------------------------------------------
def p2(task: str, horizon: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset as RepoDs

    ids = json.load(open(SPLIT_DIR / f"{task}.val_ids.json"))
    meta = LeRobotDatasetMetadata(Path(DATASET).name, root=DATASET)
    ds = RepoDs(Path(DATASET).name, root=DATASET, episodes=list(ids))
    abs_col = np.asarray([int(x) for x in ds.hf_dataset["index"]], dtype=np.int64)

    flat = list(range(0, len(ds), horizon))               # 旧口径
    flat_abs = [int(abs_col[i]) for i in flat]
    ref_abs = official_abs_starts(meta, list(ids), horizon)
    overlap = len(set(flat_abs) & set(ref_abs))
    if overlap == len(ref_abs):
        return False, "旧口径竟然与官方一致 —— P1 失去区分力，需重新审视"
    return True, (f"旧口径（拼接序列上跳）：{len(flat_abs)} 个起点，"
                  f"与官方只有 **{overlap}** 个重合（官方 {len(ref_abs)} 个）"
                  f" ⇒ P1 确实能抓到分叉")


# ---------------------------------------------------------------------------
# P3 聚合口径（纯函数单测 + 区分力）
# ---------------------------------------------------------------------------
def p3(horizon: int):
    from lingbotvla.utils.open_loop_validation import aggregate_chunks

    rng = np.random.default_rng(0)
    chunks: list[tuple] = []
    for ep, n_ch in ((51, 2), (52, 2), (56, 1)):
        for c in range(n_ch):
            g = rng.normal(size=(horizon, 14)).astype(np.float32)
            p = g + 0.05 * rng.normal(size=(horizon, 14)).astype(np.float32)
            chunks.append((ep, g, p))

    got = aggregate_chunks(chunks)
    ref, n_traj = official_mse_from_chunks(chunks)
    wrong = wrong_per_chunk_mse(chunks)

    problems = []
    if abs(got["mse"] - ref) > 1e-12:
        problems.append(f"mse {got['mse']:.8f} ≠ 官方参考 {ref:.8f}")
    if got["n"] != n_traj:
        problems.append(f"轨迹数 {got['n']} ≠ {n_traj}")
    if got["n_chunks"] != len(chunks):
        problems.append(f"chunk 数 {got['n_chunks']} ≠ {len(chunks)}")
    if abs(wrong - ref) < 1e-9:
        problems.append("错误口径与官方口径数值相同 ⇒ 本项测试没有区分力")
    if problems:
        return False, "；".join(problems)
    return True, (f"按轨迹聚合 == 官方参考（{ref:.8f}）；轨迹数 {n_traj}、chunk 数 {len(chunks)}；"
                  f"错误口径（按 chunk 平均）= {wrong:.8f}，差 {abs(wrong - ref):.8f} ⇒ 有区分力")


# ---------------------------------------------------------------------------
# P4 padding 不被额外排除
# ---------------------------------------------------------------------------
def p4(task: str, horizon: int):
    src = (REPO / "lingbotvla/utils/open_loop_validation.py").read_text(encoding="utf-8")
    # 指标路径里若出现按 is_pad 过滤，说明我们比官方多做了排除（官方不排除）
    bad_tokens = ["action_is_pad]", "mask = ~", "~pad", "is_pad)"]
    hits = [t for t in bad_tokens if t in src]
    if hits:
        return False, f"疑似按 is_pad 过滤（官方不排除 padding）：命中 {hits}"

    # 实测 padding 占比（我们 vs 官方）
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset as RepoDs
    from lingbotvla.utils.open_loop_validation import per_episode_starts

    ids = json.load(open(SPLIT_DIR / f"{task}.val_ids.json"))
    meta = LeRobotDatasetMetadata(Path(DATASET).name, root=DATASET)
    ds = RepoDs(Path(DATASET).name, root=DATASET,
                delta_timestamps={"action": [t / meta.fps for t in range(horizon)]},
                episodes=list(ids), load_image=False)
    ep_col = np.asarray([int(x) for x in ds.hf_dataset["episode_index"]], dtype=np.int64)

    def pad_fraction(starts_local):
        n = pad = 0
        for i in starts_local:
            it = ds[i]
            n += horizon
            pad += int(np.asarray(it["action_is_pad"]).sum())
        return pad, n

    ours = pad_fraction(per_episode_starts(ep_col, horizon))
    flat = pad_fraction(list(range(0, len(ds), horizon)))
    return True, (f"指标路径未按 is_pad 过滤（与官方一致）；"
                  f"padding 占比：新口径 {ours[0]}/{ours[1]} = {ours[0]/ours[1]:.1%}，"
                  f"旧口径 {flat[0]}/{flat[1]} = {flat[0]/flat[1]:.1%}"
                  f"（⚠️ 偏差方向未实测，见 --flat-stride）")


# ---------------------------------------------------------------------------
# P5 残余差异清单（只打印，不判定）
# ---------------------------------------------------------------------------
def p5(horizon: int):
    return True, (
        "以下是**代码无法静态证明**的差异，必须靠 GPU 逐值对拍（tools/open_loop_parity_test.sh）：\n"
        "  1. `max_infer_time`：官方每条轨迹最多推理 10 次（默认），超过就截断；我们无此上限。\n"
        f"     本数据集回合 ≤80 帧 ⇒ 每回合 ≤2 次 < 10 ⇒ 当前不分叉；**长回合会分叉**。\n"
        "  2. 预测 chunk 长度：官方 `policy.infer` 返回长度须 == horizon；若实际短于 GT 会静默错位。\n"
        "  3. 动作键顺序：官方按 org_features['actions'] 顺序拼接；MSE 与顺序无关，但**逐值对拍**要在同一顺序下比。\n"
        "  4. 前向链路：官方 deploy 强制 `attention_implementation='eager'` + `use_cache=True`，\n"
        "     我们评测期间临时改、finally 恢复（已加恢复审计）。\n"
        "  5. 噪声：flow-matching 的初始噪声必须**显式喂同一份**才能逐值对拍。"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="click_bell")
    ap.add_argument("--horizon", type=int, default=50, help="= chunk_size")
    a = ap.parse_args()

    print(f"[parity] repo={REPO}  task={a.task}  horizon={a.horizon}  dataset={DATASET}")
    print()
    run("P1", "评测帧集与官方一致", lambda: p1(a.task, a.horizon))
    run("P2", "旧口径确实分叉（证明 P1 有区分力）", lambda: p2(a.task, a.horizon))
    run("P3", "聚合口径 == 官方（按完整轨迹）", lambda: p3(a.horizon))
    run("P4", "padding 未被额外排除", lambda: p4(a.task, a.horizon))
    run("P5", "残余差异清单（需 GPU 对拍）", lambda: p5(a.horizon))

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"[parity] {len(_RESULTS) - n_fail}/{len(_RESULTS)} 通过"
          + ("" if n_fail == 0 else f"，{n_fail} 项 FAIL"))
    if n_fail == 0:
        print("[parity] ✅ CPU 侧口径一致；数值层请跑 tools/open_loop_parity_test.sh（需 GPU）")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

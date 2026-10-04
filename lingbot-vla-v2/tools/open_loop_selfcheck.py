#!/usr/bin/env python3
"""训练中 open-loop validation 的自检（**纯 CPU，不加载模型权重**）。

一条命令跑完这些断言，任何一项 FAIL 就非 0 退出：

  C1 划分文件自洽        train ∩ val = ∅、train ∪ val = 每任务回合数、monitor ⊆ train
  C2 块假设 vs 真实标注  `task = TASK_ORDER[episode_index // EPISODES_PER_TASK]`
                         是否与数据集里 `meta.episodes[*]['tasks']` 的指令族一致
  C3 白名单数据非 padding 回归测试：白名单过滤后 state/action **不能**全是 padding
                         （2026-10-04 那个「整段 chunk 全判 padding」的 bug）
  C4 use_cache 机制      训练配置 `use_cache` 取值 + `handle_kv_cache` 在
                         `use_cache=False, fill_kv_cache=True` 下是否**不写缓存**
  C5 _as_frames 单测     0-d→(1,1)、1-d→(1,D)、2-d/3-d→(N,D)
  C6 _episode_index_map  用真实数据集验证 local_idx → episode_index 映射
  C7 strict_getitem      开关存在、默认 False（不影响正式训练）
  C8 **审计本身的自测**   故意破坏状态，验证 `_audit_restore` 能抓到

用法
----
    cd /data/code/lingbot-vla-v2
    HF_HUB_OFFLINE=1 /data/miniconda3/envs/lingbotvla/bin/python -u \\
        tools/open_loop_selfcheck.py --task click_bell
    ... --skip-dataset          # 只跑 C1/C4/C5/C7/C8（秒级，不碰数据集）
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASET = "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
SPLIT_DIR = "/data/train/task_splits"
TRAIN_CFG = "/data/train/configs/robotwin_official_paths.yaml"
MODELING = REPO / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
MONITOR_IDS = [50, 63, 76, 87, 99]

_RESULTS: list[tuple[str, bool, str]] = []


def run(cid: str, title: str, fn):
    """跑一项检查并收集结果；异常也算 FAIL，不影响后续检查。"""
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    _RESULTS.append((f"{cid} {title}", bool(ok), str(detail)))
    print(f"[selfcheck] {'OK  ' if ok else 'FAIL'} {cid} {title}")
    print(f"            {detail}")


def _meta():
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(Path(DATASET).name, root=DATASET)


# ---------------------------------------------------------------------------
# C1 划分文件自洽
# ---------------------------------------------------------------------------
def c1(task: str, eps_per_task: int):
    man = json.load(open(f"{SPLIT_DIR}/manifest.json"))
    tr = json.load(open(f"{SPLIT_DIR}/{task}.train_ids.json"))
    va = json.load(open(f"{SPLIT_DIR}/{task}.val_ids.json"))
    n = len(set(tr) | set(va))
    ok = (not (set(tr) & set(va))) and n == eps_per_task \
        and set(MONITOR_IDS) <= set(tr) and not (set(MONITOR_IDS) & set(va))
    return ok, (f"train={len(tr)} val={len(va)} 交={sorted(set(tr) & set(va))} "
                f"并={n}/{eps_per_task} monitor⊆train={set(MONITOR_IDS) <= set(tr)} "
                f"monitor∩val={sorted(set(MONITOR_IDS) & set(va))} "
                f"(manifest strategy={man.get('strategy')})")


# ---------------------------------------------------------------------------
# C2 块假设 vs 真实标注
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return s.lower().replace("-", "").replace(" ", "").replace("_", "")


def _tok_variants(tok: str):
    n = _norm(tok)
    out = {n}
    out.add(n[:-1] if n.endswith("s") else n + "s")
    return out


def _hit(texts_norm, tok) -> float:
    vs = _tok_variants(tok)
    return sum(1 for x in texts_norm if any(v in x for v in vs)) / max(1, len(texts_norm))


def _best(texts_norm, name):
    """返回 (命中率, 最佳词)。任务名与指令用词常常不同形
    （`blocks` vs `block`、`alarmclock` vs `alarm-clock`、`pick` vs `catch`），
    所以归一化去连字符/空格 + 去复数，再对每个 token 取最大命中率。"""
    toks = [t for t in name.split("_") if len(t) >= 3] or [name]
    scored = [(_hit(texts_norm, t), t) for t in toks]
    return max(scored, key=lambda x: x[0])


def c2(task: str, eps_per_task: int):
    """块假设核对：`task = TASK_ORDER[episode_index // EPISODES_PER_TASK]`。

    这是**词表启发式**（任务名 ≠ 指令原文用词），所以：
      * 硬门槛：**我们实际训练的那个 task 所在块**必须 >= 70%
      * 全体：报告通过比例 + 失败清单（真错位会表现为「另一个任务名匹配得更好」）
    """
    sys.path.insert(0, str(REPO / "tools"))
    import robotwin_curriculum as rc

    order = list(rc.TASK_ORDER)
    meta = _meta()

    def instr(ep):
        t = meta.episodes[ep]["tasks"]
        return [str(x) for x in t] if not isinstance(t, str) else [t]

    per_block = []
    for b, name in enumerate(order):
        eps = list(range(b * eps_per_task, (b + 1) * eps_per_task))
        if eps[-1] >= len(meta.episodes):
            break
        tn = [_norm(x) for e in eps for x in instr(e)]
        per_block.append((b, name, tn))

    rows, fails, suspicious = [], [], []
    for b, name, tn in per_block:
        own_rate, own_tok = _best(tn, name)
        rows.append((b, name, own_rate, own_tok))
        if own_rate < 0.7:
            fails.append(f"block{b} {name} 最佳词'{own_tok}' {own_rate:.0%}")
        # 真错位信号：**另一个**任务名（最佳词不同）明显匹配得更好
        best_other, best_other_tok, best_other_name = 0.0, None, None
        for b2, name2, tn2 in per_block:
            if name2 == name:
                continue
            r2, t2 = _best(tn, name2)
            if t2 != own_tok and r2 > best_other:
                best_other, best_other_tok, best_other_name = r2, t2, name2
        if best_other > own_rate + 0.2:
            suspicious.append(f"block{b} {name}({own_tok} {own_rate:.0%}) "
                              f"更像 {best_other_name}({best_other_tok} {best_other:.0%})")

    target_blocks = [r for r in rows if r[1] == task]
    target_ok = bool(target_blocks) and target_blocks[0][2] >= 0.7
    pass_ratio = 1.0 - len(fails) / max(1, len(rows))
    ok = target_ok and pass_ratio >= 0.6 and not suspicious
    detail = (f"目标 task '{task}' 所在块 = "
              f"{target_blocks[0] if target_blocks else '未找到'}；"
              f"50 块中 {len(rows) - len(fails)}/{len(rows)} 达到 70%（{pass_ratio:.0%}）")
    if fails:
        detail += "\n            未达 70% 的块（多为任务名用词与指令不一致，非错位）:\n              " \
            + "\n              ".join(fails[:6])
    if suspicious:
        detail += "\n            🔴 疑似错位:\n              " + "\n              ".join(suspicious)
    return ok, detail


# ---------------------------------------------------------------------------
# C3 白名单数据非 padding（回归测试）
# ---------------------------------------------------------------------------
def _repo_ds():
    """必须用**仓库自己的子类** —— 索引空间修复只打在
    `lingbotvla/data/vla_data/base_dataset.py::LeRobotDataset`，
    上游 `lerobot.datasets.LeRobotDataset` 仍然是坏的（2026-10-04 踩过）。"""
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset as RepoDs

    return RepoDs


def c3(task: str):
    RepoDs = _repo_ds()
    ep = json.load(open(f"{SPLIT_DIR}/{task}.train_ids.json"))[0]
    meta = _meta()
    fps = meta.fps
    delta = {"observation.state": [t / fps for t in range(51)],
             "action": [t / fps for t in range(50)]}
    ds = RepoDs(Path(DATASET).name, root=DATASET, delta_timestamps=delta, episodes=[ep])
    n = len(ds)

    bad, lines = [], []
    for k in (0, 1, n // 2, n - 1):
        it = ds[k]
        sp = int(it["observation.state_is_pad"].sum())
        ap = int(it["action_is_pad"].sum())
        st, ac = it["observation.state"], it["action"]
        nz = float(np.abs(st[0].numpy()).sum() + np.abs(ac[0].numpy()).sum())
        must = k < 2                      # 前两帧必须完全非 pad（回合末 pad 是合理的）
        good = (sp == 0 and ap == 0 and nz > 0) if must else True
        if not good:
            bad.append(k)
        lines.append(f"k={k:<3} pad={sp}/{ap} |x|={nz:.4f}{'  ❌' if not good else ''}")
    return not bad, (f"回合 {ep}，len={n}；" + " | ".join(lines)
                     + "（修复前这里全是 pad=51/50 且全 0）")


def c3b_worker(task: str):
    """真正的跨数据集对照（由 `--only-c3b` 单独调用；**必须独立进程**，
    父进程若已持有数据集，两者合计会超 cgroup 2GiB 被 OOM 杀掉）。"""
    RepoDs = _repo_ds()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    ep = json.load(open(f"{SPLIT_DIR}/{task}.train_ids.json"))[0]
    meta = LeRobotDatasetMetadata(Path(DATASET).name, root=DATASET)
    fps = meta.fps
    delta = {"observation.state": [t / fps for t in range(51)],
             "action": [t / fps for t in range(50)]}
    rid = Path(DATASET).name
    k = 3

    f = RepoDs(rid, root=DATASET, delta_timestamps=delta, episodes=[ep])[k]
    filt = [round(float(torch.abs(f["observation.state"][0]).sum()), 6),
            round(float(torch.abs(f["action"][0]).sum()), 6)]
    del f
    import gc

    gc.collect()
    abs_i = meta.episodes[ep]["dataset_from_index"] + k
    g = RepoDs(rid, root=DATASET, delta_timestamps=delta)[abs_i]
    full = [round(float(torch.abs(g["observation.state"][0]).sum()), 6),
            round(float(torch.abs(g["action"][0]).sum()), 6)]
    ok = (filt == full) and filt[0] > 0
    return ok, (f"白名单 idx={k} {filt} vs 不过滤绝对帧 {abs_i} {full} "
                f"⇒ {'一致' if ok else '❌ 不一致（索引空间又坏了）'}")


# ---------------------------------------------------------------------------
# C4 use_cache 机制
# ---------------------------------------------------------------------------
def c4():
    import yaml

    raw = yaml.safe_load(open(TRAIN_CFG))
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config

    cfg = LingbotVLAV2Config(**raw["model"])

    # AST 抽真实源码执行（直接 import 建模模块会连带 ops.group_gemm 调 CUDA，无卡必炸）
    src = MODELING.read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "handle_kv_cache")
    fn_src = textwrap.dedent(ast.get_source_segment(src, node))
    ns = {"torch": torch, "Optional": __import__("typing").Optional,
          "Union": __import__("typing").Union, "List": list, "Cache": object}
    exec(compile(fn_src, "<handle_kv_cache>", "exec"), ns)
    hkc = ns["handle_kv_cache"]

    class _S:
        pass

    k = torch.zeros(1, 8, 4, 8)
    v = torch.zeros(1, 8, 4, 8)
    off = hkc(_S(), k, v, 0, past_key_values=None, use_cache=False, fill_kv_cache=True)[2]
    on = hkc(_S(), k, v, 0, past_key_values=None, use_cache=True, fill_kv_cache=True)[2]
    ok = (cfg.use_cache is False) and (off is None) and (on is not None)
    return ok, (f"训练配置 use_cache={cfg.use_cache}（yaml 里有该键: "
                f"{'use_cache' in raw['model']}）；fill+use_cache=False -> past_kv={off}；"
                f"fill+use_cache=True -> past_kv={'dict' if on is not None else None}"
                f" ⇒ eval 期间**必须**临时置 True")


# ---------------------------------------------------------------------------
# C5 _as_frames 单测
# ---------------------------------------------------------------------------
def c5():
    from lingbotvla.utils.open_loop_validation import _as_frames

    cases = {"0-d": (np.float32(1.0), (1, 1)),
             "1-d(14,)": (np.zeros(14, np.float32), (1, 14)),
             "2-d(50,14)": (np.zeros((50, 14), np.float32), (50, 14)),
             "3-d(50,1,14)": (np.zeros((50, 1, 14), np.float32), (50, 14))}
    bad = {n: (got.shape, want) for n, (a, want) in cases.items()
           if (got := _as_frames(a, "action")).shape != want}
    return not bad, (f"{len(cases)} 例全过" if not bad else f"失败: {bad}")


# ---------------------------------------------------------------------------
# C6 _episode_index_map
# ---------------------------------------------------------------------------
def c6(task: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as RawDs

    from lingbotvla.utils.open_loop_validation import _episode_index_map

    ids = json.load(open(f"{SPLIT_DIR}/{task}.train_ids.json"))[:2]
    repo_id = Path(DATASET).name
    ds = RawDs(repo_id, root=DATASET, episodes=ids)

    class _Shim:
        def __init__(self, inner):
            self._datasets = [types.SimpleNamespace(dataset=inner)]

        def __len__(self):
            return len(self._datasets[0].dataset)

    m = _episode_index_map(_Shim(ds))
    if m is None:
        return False, "返回 None（拿不到映射 ⇒ 聚合会退化成按 chunk）"
    uniq, cnt = np.unique(m, return_counts=True)
    ok = len(m) == len(ds) and set(uniq.tolist()) == set(ids)
    return ok, f"len={len(m)}（数据集 {len(ds)}）映射计数={dict(zip(uniq.tolist(), cnt.tolist()))}"


# ---------------------------------------------------------------------------
# C7 strict_getitem
# ---------------------------------------------------------------------------
def c7():
    from lingbotvla.data.vla_data.multi_vla_dataset import MultiVLADataset

    src = (REPO / "lingbotvla/data/vla_data/multi_vla_dataset.py").read_text(encoding="utf-8")
    has_init = "self.strict_getitem = False" in src
    has_guard = "if getattr(self, \"strict_getitem\", False):" in src
    return has_init and has_guard, (
        f"类属性初始化={'有' if has_init else '缺'}；__getitem__ 早返回={'有' if has_guard else '缺'}"
        f"（默认 False ⇒ 正式训练的随机重试行为不变）")


# ---------------------------------------------------------------------------
# C8 审计本身的自测
# ---------------------------------------------------------------------------
def c8():
    from lingbotvla.utils.open_loop_validation import _audit_restore, _rng_snapshot

    class _Cfg:
        use_cache = False

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

    m, cfg = _M(), _Cfg()
    snap = _rng_snapshot()
    flags = [(x, x.training) for x in m.modules()]
    kw = dict(rng_snap=snap, train_flags=flags, compile_flag=None, model=m)

    clean = _audit_restore(use_cache_saved=[(cfg, False)], ft_aug_orig={}, **kw)

    # 故意破坏 3 处：子模块 training / use_cache / numpy RNG
    m.lin.training = not m.lin.training
    cfg.use_cache = True
    np.random.rand()
    dirty = _audit_restore(use_cache_saved=[(cfg, False)], ft_aug_orig={}, **kw)

    ok = (clean == []) and len(dirty) >= 3
    return ok, (f"干净状态报告 {len(clean)} 个问题（应为 0）；"
                f"故意破坏 3 处后报告 {len(dirty)} 个：{dirty}")


def c9():
    """归一化统计来源必须与 deploy 一致（2026-10-04 那个「训练/评测两套 stats」的坑）。

    ⚠️ 这里**不构造 FeatureTransform** —— 源 yaml 的 `joints`/`norm_type` 是 yaml dict，
    而运行时拿到的是字符串，用源 yaml 构造不忠实（会报 malformed node）。
    运行时的实证由 `open_loop_validation` 打的那行「归一化统计指纹」负责。
    """
    import json

    src = (REPO / "lingbotvla/data/vla_data/base_dataset.py").read_text(encoding="utf-8")
    static_ok = 'norm_stats_path=getattr(dataset_config, "norm_stats_file", None)' in src

    raw = yaml.safe_load(open(TRAIN_CFG))
    nsp = raw.get("data", {}).get("norm_stats_file")
    robot_cfg = yaml.safe_load(open(REPO / "configs/robot_configs/robotwin.yaml"))
    fallback = robot_cfg.get("norm_stats")

    if not nsp:
        return static_ok, f"base_dataset 透传={static_ok}；但 data.norm_stats_file 为空（无法比较）"

    a = json.load(open(nsp))
    b_path = REPO / str(fallback)
    b = json.load(open(b_path))
    ca, cb = a.get("count"), b.get("count")
    differ = ca != cb
    key = "observation.state.arm.position"
    ma = np.asarray(a["norm_stats"][key]["mean"]).reshape(-1)[:3]
    mb = np.asarray(b["norm_stats"][key]["mean"]).reshape(-1)[:3]
    ok = static_ok and differ
    return ok, (
        f"base_dataset 透传={static_ok}；"
        f"norm_stats_file={Path(nsp).name}(count={ca}) vs robot_config={Path(str(fallback)).name}(count={cb}) "
        f"两者不同={differ}；mean[:3] 新={np.round(ma, 5).tolist()} 旧={np.round(mb, 5).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="click_bell")
    ap.add_argument("--eps-per-task", type=int, default=50)
    ap.add_argument("--skip-dataset", action="store_true",
                    help="只跑 C1/C4/C5/C7/C8（秒级，不碰数据集）")
    ap.add_argument("--only-c3b", action="store_true",
                    help="只跑 C3b 跨数据集对照（会加载全量数据集，请**单独**调用）")
    a = ap.parse_args()

    if a.only_c3b:                      # 子进程模式：只做全量对照
        run("C3b", "白名单 vs 不过滤 数值一致", lambda: c3b_worker(a.task))
        return 0 if all(ok for _, ok, _ in _RESULTS) else 1

    print(f"[selfcheck] repo={REPO}  task={a.task}  dataset={DATASET}")
    print()

    run("C1", "划分文件自洽", lambda: c1(a.task, a.eps_per_task))
    run("C4", "use_cache 机制", c4)
    run("C5", "_as_frames 单测", c5)
    run("C7", "strict_getitem 开关", c7)
    run("C8", "恢复审计自测", c8)
    run("C9", "归一化统计来源一致", c9)
    if not a.skip_dataset:
        run("C2", "块假设 vs 真实标注", lambda: c2(a.task, a.eps_per_task))
        run("C3", "白名单数据非 padding（回归）", lambda: c3(a.task))
        run("C6", "_episode_index_map", lambda: c6(a.task))

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print()
    print(f"[selfcheck] {len(_RESULTS) - n_fail}/{len(_RESULTS)} 通过"
          + ("" if n_fail == 0 else f"，{n_fail} 项 FAIL"))
    print("[selfcheck] 提示：C3b（白名单 vs 不过滤 逐值对照）要**单独**跑，"
          "否则两个数据集叠加会超 cgroup 2GiB：")
    print(f"[selfcheck]   python tools/open_loop_selfcheck.py --only-c3b --task {a.task}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

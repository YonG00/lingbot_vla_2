"""TaskCatalog —— 复用 `tools/task_split.py` 的 manifest，**不重新划分**。

数据来源（`tools/task_split.py` 的产物）
---------------------------------------
    <out>/manifest.json         各任务明细 + 参数 + sha256
    <out>/<task>.train_ids.json 裸列表
    <out>/<task>.val_ids.json   裸列表

本模块**只依赖 stdlib**，可以脱离 torch/lerobot 导入与测试。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .ports import TaskEntry

DEFAULT_EPISODES_PER_TASK = 50


@dataclass(frozen=True)
class SplitProblem:
    """划分自检发现的问题（人话 + 可定位）。"""

    task: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.task}] {self.kind}: {self.detail}"


class TaskCatalog:
    """任务 → 固定 train/val 划分。

    典型用法::

        cat = TaskCatalog.from_manifest("/data/train/task_splits/manifest.json")
        cat.entry("click_bell").ids_for("train")   # 40 条
        cat.entry("click_bell").ids_for("val")     # 10 条
    """

    def __init__(
        self,
        tasks: Mapping[str, TaskEntry],
        *,
        meta: Optional[Dict] = None,
        episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
        task_order: Optional[Sequence[str]] = None,
    ):
        if not tasks:
            raise ValueError("TaskCatalog 不能为空（manifest 里一个任务都没有）")
        self._tasks: Dict[str, TaskEntry] = dict(tasks)
        self.meta: Dict = dict(meta or {})
        self.episodes_per_task = int(episodes_per_task)
        self.task_order: Optional[List[str]] = list(task_order) if task_order else None

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        *,
        task_order: Optional[Sequence[str]] = None,
        episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
        strict: bool = True,
    ) -> "TaskCatalog":
        """从 `task_split.py` 的 `manifest.json` 构造。

        ``strict=True``（默认）会在**任何**划分自检失败时直接抛错 ——
        训练前的 fail-fast 比跑一半发现漏了 val 强。
        """
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        raw_tasks = man.get("tasks")
        if not isinstance(raw_tasks, dict) or not raw_tasks:
            raise ValueError(f"manifest 里没有 tasks 字典: {manifest_path}")

        root = os.path.dirname(os.path.abspath(manifest_path))
        tasks: Dict[str, TaskEntry] = {}
        for name, rec in raw_tasks.items():
            if not isinstance(rec, dict):
                raise ValueError(f"manifest.tasks[{name!r}] 不是字典")
            train = rec.get("train_ids")
            val = rec.get("val_ids")
            # manifest 没内联 ids 时，退回读同目录的 <task>.train_ids.json
            if train is None:
                train = _read_ids(os.path.join(root, f"{name}.train_ids.json"))
            if val is None:
                val = _read_ids(os.path.join(root, f"{name}.val_ids.json"))
            tasks[name] = TaskEntry(
                name=name,
                train_ids=tuple(int(x) for x in train),
                val_ids=tuple(int(x) for x in val),
                n_total=int(rec.get("n_total", len(train) + len(val))),
                sha256_train=rec.get("sha256_train"),
                sha256_val=rec.get("sha256_val"),
                val_ratio=rec.get("val_ratio"),
                strategy=rec.get("strategy"),
            )

        cat = cls(
            tasks,
            meta={k: v for k, v in man.items() if k != "tasks"},
            episodes_per_task=int(man.get("episodes_per_task", episodes_per_task)),
            task_order=task_order,
        )
        if strict:
            problems = cat.verify()
            if problems:
                raise ValueError(
                    "task split 自检未通过：\n  - "
                    + "\n  - ".join(str(p) for p in problems)
                )
        return cat

    # -- 查询 ---------------------------------------------------------------
    def names(self) -> List[str]:
        return list(self._tasks.keys())

    def entry(self, task: str) -> TaskEntry:
        try:
            return self._tasks[task]
        except KeyError:
            raise KeyError(
                f"未知任务 {task!r}；可用：{self.names()[:8]}{' ...' if len(self) > 8 else ''}"
            ) from None

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task: object) -> bool:
        return task in self._tasks

    def __iter__(self):
        return iter(self._tasks.values())

    # -- 自检 ---------------------------------------------------------------
    def verify(self) -> List[SplitProblem]:
        """划分自检；返回问题列表（空 = 全过）。**不需要数据集**。"""
        problems: List[SplitProblem] = []
        seen_episodes: Dict[int, str] = {}
        for name, e in self._tasks.items():
            tr, va = list(e.train_ids), list(e.val_ids)
            if not tr:
                problems.append(SplitProblem(name, "空 train", "train_ids 为空，训练集会是空的"))
            if not va:
                problems.append(SplitProblem(name, "空 val", "val_ids 为空，评测集会是空的"))
            if len(set(tr)) != len(tr):
                problems.append(SplitProblem(name, "train 有重复", f"{len(tr)} → {len(set(tr))}"))
            if len(set(va)) != len(va):
                problems.append(SplitProblem(name, "val 有重复", f"{len(va)} → {len(set(va))}"))
            dup = sorted(set(tr) & set(va))
            if dup:
                problems.append(SplitProblem(
                    name, "train/val 泄漏", f"{len(dup)} 个回合同时出现在两边: {dup[:5]}"))
            if e.n_total and e.n_total != e.n_train + e.n_val:
                problems.append(SplitProblem(
                    name, "数量对不上",
                    f"n_total={e.n_total} != n_train({e.n_train}) + n_val({e.n_val})"))
            neg = [i for i in tr + va if i < 0]
            if neg:
                # ⚠️ 回合号是**全局**的（turn_switch 就是 50..99），所以不能拿 n_total
                #    当上界；上界只能靠 TASK_ORDER 的分块结构来查（见下）。
                problems.append(SplitProblem(name, "回合号为负", f"{neg[:5]}"))
            # 跨任务：同一回合号不能属于两个任务
            for i in tr + va:
                prev = seen_episodes.get(i)
                if prev is not None and prev != name:
                    problems.append(SplitProblem(
                        name, "跨任务重叠", f"回合 {i} 已属于任务 {prev!r}"))
                    break
                seen_episodes[i] = name

        # 若给了 TASK_ORDER：核对「分块结构」block_id = episode_index // EPISODES_PER_TASK
        if self.task_order is not None:
            for name, e in self._tasks.items():
                if name not in self.task_order:
                    problems.append(SplitProblem(name, "不在 TASK_ORDER", "任务名对不上数据集分块顺序"))
                    continue
                b = self.task_order.index(name)
                lo = b * self.episodes_per_task
                hi = lo + self.episodes_per_task
                bad = [i for i in list(e.train_ids) + list(e.val_ids) if not (lo <= i < hi)]
                if bad:
                    problems.append(SplitProblem(
                        name, "分块结构不符",
                        f"block {b} 应为 [{lo}, {hi})，但有 {bad[:5]} 落在外面"))
                elif e.n_train + e.n_val != self.episodes_per_task:
                    problems.append(SplitProblem(
                        name, "回合数不符",
                        f"应为 {self.episodes_per_task}，实际 {e.n_train + e.n_val}"))
        return problems

    def task_of_episode(self, episode: int) -> str:
        """回合号 → 任务名（`block_id = episode_index // EPISODES_PER_TASK`）。"""
        if self.task_order is None:
            for name, e in self._tasks.items():
                if episode in e.train_ids or episode in e.val_ids:
                    return name
            raise KeyError(f"回合 {episode} 不属于任何已知任务")
        b = int(episode) // self.episodes_per_task
        if b >= len(self.task_order):
            raise KeyError(f"回合 {episode} 的 block_id={b} 超出 TASK_ORDER（{len(self.task_order)} 个）")
        return self.task_order[b]


# --------------------------------------------------------------------------- #


def _read_ids(path: str) -> List[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到回合白名单: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"回合白名单必须是**非空** JSON 数组: {path}")
    bad = [x for x in raw if isinstance(x, bool) or not isinstance(x, int)]
    if bad:
        raise ValueError(f"回合白名单含非整数项 {bad[:5]} ... : {path}")
    return raw


def load_task_order(curriculum_module_path: Optional[str] = None) -> List[str]:
    """读取 `tools/robotwin_curriculum.py` 的 `TASK_ORDER`（**懒加载**，避免引入 pandas/yaml）。

    没给路径时按仓库布局找 `tools/robotwin_curriculum.py`。
    """
    import importlib.util

    path = curriculum_module_path
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))          # lingbotvla/auto_learning
        repo = os.path.dirname(os.path.dirname(here))              # 仓库根
        path = os.path.join(repo, "tools", "robotwin_curriculum.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 robotwin_curriculum.py: {path}")
    spec = importlib.util.spec_from_file_location("_al_robotwin_curriculum", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 该模块 import pandas/yaml，故放在函数里
    return list(mod.TASK_ORDER)


def episodes_per_task(curriculum_module_path: Optional[str] = None) -> int:
    """读取 `robotwin_curriculum.EPISODES_PER_TASK`（**懒加载**）。"""
    import importlib.util

    path = curriculum_module_path
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.dirname(os.path.dirname(here))
        path = os.path.join(repo, "tools", "robotwin_curriculum.py")
    spec = importlib.util.spec_from_file_location("_al_robotwin_curriculum2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(getattr(mod, "EPISODES_PER_TASK", DEFAULT_EPISODES_PER_TASK))


def catalog_from_task_split(
    manifest_path: str,
    *,
    curriculum_module_path: Optional[str] = None,
    strict: bool = True,
) -> TaskCatalog:
    """便捷构造：自动带上 `TASK_ORDER` / `EPISODES_PER_TASK` 做分块自检。"""
    try:
        order = load_task_order(curriculum_module_path)
        ept = episodes_per_task(curriculum_module_path)
    except Exception:  # noqa: BLE001  —— 环境里没有 pandas 时降级为「不做分块自检」
        order, ept = None, DEFAULT_EPISODES_PER_TASK
    return TaskCatalog.from_manifest(
        manifest_path, task_order=order, episodes_per_task=ept, strict=strict
    )


__all__ = ["TaskCatalog", "SplitProblem", "catalog_from_task_split",
           "load_task_order", "episodes_per_task", "DEFAULT_EPISODES_PER_TASK"]

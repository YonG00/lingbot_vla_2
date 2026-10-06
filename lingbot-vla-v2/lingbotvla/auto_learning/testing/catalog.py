"""Stage A 的 `TaskCatalog` / `SampleResolver` 实现。

这两个类是**唯一**知道「假世界的 sample_id 是怎么编号的」的地方。
Stage B 只要写一份查数据集索引的实现，其余模块不用动。

编号规则（**任务内唯一、跨任务不重叠**）：

    sample_id = task_index * TASK_STRIDE + traj_id * SAMPLE_STRIDE + frame_id

跨任务不重叠很重要 —— 真实数据集里帧号本来就是全局唯一的；
如果所有任务共用同一段编号，「外来 sample_id」这类错误就检测不出来
（`test_I05_new_probs_must_belong_to_the_task` 抓出来过）。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..config import TaskSpecConfig
from ..ports import TaskEntry
from ..types import SampleRef

SAMPLE_STRIDE = 1000
TASK_STRIDE = 1_000_000


def sample_ids_of(spec: TaskSpecConfig, traj_ids: Sequence[int], task_index: int = 0) -> List[int]:
    """把轨迹 id 展开成样本 id。"""
    base = task_index * TASK_STRIDE
    out: List[int] = []
    for t in traj_ids:
        start = base + t * SAMPLE_STRIDE
        out.extend(start + f for f in range(spec.samples_per_traj))
    return out


def traj_of_sample(sample_id: int, task_index: int = 0) -> int:
    return (int(sample_id) - task_index * TASK_STRIDE) // SAMPLE_STRIDE


def frame_of_sample(sample_id: int) -> int:
    return int(sample_id) % SAMPLE_STRIDE


class SimSampleResolver:
    """`sample_id` → `SampleRef`（Stage A：按编号规则解码）。"""

    def __init__(self, specs: Dict[str, TaskSpecConfig], order: Sequence[str] = ()) -> None:
        self._specs = dict(specs)
        names = list(order) or list(specs.keys())
        self._index = {name: i for i, name in enumerate(names)}

    def task_index(self, task: str) -> int:
        return self._index.get(task, 0)

    def resolve(self, task: str, sample_id: int) -> SampleRef:
        idx = self.task_index(task)
        return SampleRef(
            task=task,
            sample_id=int(sample_id),
            episode_id=traj_of_sample(sample_id, idx),
            frame_id=frame_of_sample(sample_id),
        )


class SimTaskCatalog:
    """`TaskCatalog`（Stage A：从 `SimConfig.tasks` 构建）。"""

    def __init__(self, specs: Sequence[TaskSpecConfig]) -> None:
        self._entries: Dict[str, TaskEntry] = {}
        self._order: List[str] = []
        for i, spec in enumerate(specs):
            base = i * TASK_STRIDE
            self._entries[spec.name] = TaskEntry(
                name=spec.name,
                train_traj_ids=list(spec.train_traj_ids),
                val_traj_ids=list(spec.val_traj_ids),
                train_sample_ids=sample_ids_of(spec, spec.train_traj_ids, i),
                samples_by_traj={
                    t: [base + t * SAMPLE_STRIDE + f for f in range(spec.samples_per_traj)]
                    for t in spec.train_traj_ids
                },
                baseline_mse=spec.baseline_mse,
            )
            self._order.append(spec.name)

    def task_names(self) -> List[str]:
        return list(self._order)

    def entry(self, task: str) -> TaskEntry:
        return self._entries[task]

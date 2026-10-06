"""日志：events.jsonl + history.csv + **真正的 TensorBoard event 文件**（文档 §40）。

TensorBoard 是可选的：能写就写真的 event 文件，不能写就只落 `tb_scalars.jsonl`
（`{step, tag, value}` 的 JSONL，用 `tools/tb_from_jsonl.py` 事后也能补成 event 文件）。
这样 Stage A 在纯标准库环境下照样能跑。
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

TB_TAGS_DOC = """
Panel A Training:        training/loss, training/vla_loss, training/grad_norm, training/lr
Panel B Current Skill:   current_skill/{task_id,attempt,attempt_step,train_nmse,val_nmse,lp50,overfit,train_val_gap_ratio}
Panel C Skill Overview:  skill_overview/{pass_count,candidate_count,defer_count,exhausted_count,coverage,
                                         mean_scout_nmse,median_scout_nmse,worst_scout_nmse,current_task_id,current_round,reopen_count}
Panel D Memory:          memory/{pass_pool_size,forgotten_count,replay_slots,replay_unique_tasks_per_batch,reopen_count}
Panel E System:          system/{open_loop_eval_seconds,global_scout_seconds,hardness_scan_seconds,peak_memory_gb,global_samples_seen}
Per-task Detail:         task/<name>/{scout_nmse,train_nmse,val_nmse,best_nmse,lp50,overfit,status,attempt_count,reopen_count,forgotten}
Debug:                   debug/<task>/{train_mse,val_mse,baseline_mse}
""".strip()


class TensorboardSink:
    """把标量写进**真正的 TensorBoard event 文件**。

    优先级：
      ① `tensorboard` 包自带的 `EventFileWriter` —— **不需要 torch**（首选）
      ② `torch.utils.tensorboard.SummaryWriter` —— 环境里有 torch 时可用
      ③ 都没有 ⇒ 不写 event 文件，只保留 `tb_scalars.jsonl`（不报错）

    落盘位置：`<outdir>/tb/events.out.tfevents.*`
    """

    def __init__(self, logdir: str) -> None:
        self.logdir = logdir
        self._writer: Any = None
        self._kind: Optional[str] = None
        self._Event = None
        self._Summary = None
        try:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter

            self._writer = EventFileWriter(logdir)
            self._kind = "event_file_writer"
            return
        except Exception:
            pass
        try:  # pragma: no cover - 取决于环境
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=logdir)
            self._kind = "summary_writer"
        except Exception:
            self._writer = None

    @property
    def ok(self) -> bool:
        return self._writer is not None

    @property
    def backend(self) -> str:
        return self._kind or "none"

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            return
        if self._kind == "event_file_writer":
            if self._Event is None:
                from tensorboard.compat.proto.event_pb2 import Event
                from tensorboard.compat.proto.summary_pb2 import Summary

                self._Event, self._Summary = Event, Summary
            self._writer.add_event(
                self._Event(
                    step=int(step),
                    wall_time=time.time(),
                    summary=self._Summary(
                        value=[self._Summary.Value(tag=tag, simple_value=float(value))]
                    ),
                )
            )
        else:  # pragma: no cover
            self._writer.add_scalar(tag, float(value), int(step))

    def close(self) -> None:
        if self._writer is None:
            return
        for fn in ("flush", "close"):
            try:
                getattr(self._writer, fn)()
            except Exception:
                pass


class EventLogger:
    def __init__(self, outdir: str, append: bool = False, tensorboard: bool = True) -> None:
        """`append=True` 用于 resume：**不能截断**上一次 run 已经写好的事件流。"""
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.events_path = os.path.join(outdir, "events.jsonl")
        self.tb_path = os.path.join(outdir, "tb_scalars.jsonl")
        self.history_path = os.path.join(outdir, "history.csv")
        mode = "a" if append else "w"
        self._events = open(self.events_path, mode, encoding="utf-8")
        self._tb = open(self.tb_path, mode, encoding="utf-8")
        self.tb_dir = os.path.join(outdir, "tb")
        # resume 时新建一个 event 文件（TB 支持一个 logdir 下多个 event file），
        # 所以不需要 append 语义，也不会覆盖历史。
        self.tb = TensorboardSink(self.tb_dir) if tensorboard else TensorboardSink("")
        self.n_events = 0
        self.n_scalars = 0

    # ---------------------------------------------------------------- #
    def log_event(self, event: Dict[str, Any]) -> None:
        self._events.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events.flush()
        self.n_events += 1

    def log_metrics(self, step: int, tag: str, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        self._tb.write(json.dumps({"step": int(step), "tag": tag, "value": val}) + "\n")
        self.n_scalars += 1
        self.tb.add_scalar(tag, val, int(step))

    # ---------------------------------------------------------------- #
    def write_history(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            open(self.history_path, "w", encoding="utf-8").close()
            return
        keys: List[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(self.history_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def close(self) -> None:
        self._events.close()
        self._tb.close()
        self.tb.close()


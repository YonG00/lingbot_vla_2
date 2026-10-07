"""★ 主循环（文档 §42 的运行流程）。

设计要点：把整个流程拆成**原子动作**，一次 `advance()` 只做一件事：

    bootstrap（2-val 扫描 → 4-val 确认）
    select   （挑 scout NMSE 最低的 candidate → 建 baseline → hardness 扫描）
    train_unit（50 step + 4train/4val open-loop + 判定）
    review   （每 N 次 transition 复查 PASS pool）
    rollover （本轮候选处理完，给 DEFER 第二次机会）
    finish

好处：**存档 / 恢复天然正确** —— 每个原子动作之间都是一个一致的断点，
所以「中途存档再恢复」必然走出同一条路线（见 `tests/test_resume.py`）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import AutoLearningConfig
from ..decision.metrics import (
    is_finite_metric,
    is_overfit,
    learning_progress,
    median,
    train_val_gap,
)
from ..decision.review import Reviewer
from ..decision.thresholds import PassCheck, check_pass, is_pass
from ..decision.state_machine import (
    apply_defer,
    apply_pass,
    decide_after_unit,
    is_terminal,
    rollover_round,
    should_rescue,
    start_attempt,
)
from ..ports import Backend, ReplayPlan, TrainRequest, TrainResult
from ..sampling.hardness_scan import HardnessScan, HardnessScanner
from ..state.registry import TaskRecord, TaskRegistry
from ..types import Decision, EvalSplit, ReasonCode, TaskStatus

def _true_nmse(extra_state, task: str):
    """真值 NMSE 只用于 debug 列（Stage B 没有假世界时返回 None）。"""
    if extra_state is None or not hasattr(extra_state, "true_nmse"):
        return None
    try:
        return round(extra_state.true_nmse(task), 5)
    except KeyError:
        return None


MAX_METRIC_ROWS = 20000
MAX_TRANSITION_ROWS = 2000


@dataclass
class SchedulerState:
    global_step: int = 0
    global_samples_seen: int = 0
    round: int = 1
    transition_count: int = 0
    current_task: Optional[str] = None
    attempt_step: int = 0
    rescued: bool = False
    #: 本次 run 训练过的**全部**任务（诊断用）
    trained_tasks: List[str] = field(default_factory=list)
    #: **真正的新任务**：第一次进入训练、且此前从未 PASS 过 —— 只有它占 `max_new_tasks_attempted`
    new_tasks_attempted: List[str] = field(default_factory=list)
    #: 首次通过（此前从未 PASS 过）—— 占 `max_new_tasks_passed`
    newly_passed: List[str] = field(default_factory=list)
    #: 旧任务恢复通过（曾经 PASS 过、因遗忘回炉后又过了）—— 属维护，**不占**任何新增额度
    repassed: List[str] = field(default_factory=list)
    #: 免费 PASS（bootstrap 扫描 / rescan 被 transfer 带起来）
    auto_passed: List[str] = field(default_factory=list)
    bootstrap_queue: List[str] = field(default_factory=list)
    pending_review: bool = False
    finished: bool = False
    stop_reason: str = ""
    units_run: int = 0
    eval_events: int = 0
    transitions: List[Dict[str, Any]] = field(default_factory=list)

    def to_state(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def load_state(self, raw: Dict[str, Any]) -> None:
        for key, value in raw.items():
            if hasattr(self, key):
                setattr(self, key, value)


# --------------------------------------------------------------------------- #
class Scheduler:
    """主循环（文档 §42）。

    **只依赖 `ports.Backend` 里的五个 adapter** —— 不认识 `SimConfig`、
    不认识 `SimulatedWorld`、不知道 `sample_id` 是怎么编码的。
    Stage B 换一份 backend 实现即可，这个文件一行都不用改
    （`tests/test_contracts.py::test_scheduler_has_no_sim_dependencies` 会守这条）。
    """

    def __init__(
        self,
        backend: Backend,
        al: AutoLearningConfig,
        *,
        seed: int = 0,
        logger: Any = None,
    ) -> None:
        self.backend = backend
        self.al = al
        self.logger = logger

        self.rng = random.Random(seed)
        self.evaluator = backend.evaluator
        self.trainer = backend.trainer
        self.registry = TaskRegistry.from_catalog(backend.catalog, al)
        #: B1：True 时 `_train_unit()` **不**自己调 trainer，只发布 request，
        #: 等外层真实训练循环跑完后用 `complete_train_unit(result)` 回填。
        #: 默认 False ⇒ Stage A / 既有测试的行为**逐位不变**。
        self.defer_train = False
        self._pending_train = None
        self.scanner = HardnessScanner(al, backend.scorer)
        self.reviewer = Reviewer(al, self.evaluator)
        self.rebind_rng()

        self.state = SchedulerState(bootstrap_queue=list(self.registry.names()))
        self.scans: Dict[str, HardnessScan] = {}
        self.metrics_rows: List[Dict[str, Any]] = []
        self.heatmap_rows: List[Dict[str, Any]] = []
        self.unit_budget: Optional[int] = None
        self.checkpoint_path: Optional[str] = None
        self.events: List[Dict[str, Any]] = []
        self.last_decision: Dict[str, Any] = {}

    # ---------------------------------------------------------------- #
    def rebind_rng(self) -> None:
        """把 scheduler 的 RNG 注入 trainer / sampler。

        🔴 全流程只允许**一个权威随机源**。解耦时曾经踩过：scheduler 自己建了一个
        `Random(seed)`，而 `build_backend` 里的 sampler 也建了一个 —— resume 时把
        scheduler 的 RNG 灌给 sampler，抽样流就和「不中断」对不上了
        （`test_resume_reproduces_identical_route` 抓出来的）。
        """
        sampler = getattr(getattr(self.backend, "trainer", None), "sampler", None)
        if sampler is not None and hasattr(sampler, "rng"):
            sampler.rng = self.rng

    @property
    def extra_state(self):
        """需要随状态一起存档的额外对象（Stage A 的假世界；Stage B 为 None）。"""
        return self.backend.extra_state

    def replay_plan(self) -> ReplayPlan:
        """PASS 池 = registry 里 status 为 PASS 的任务（含各自 PASS 时的采样分布）。

        不再需要单独的 ReplayPool 对象 —— PASS 状态本来就在 registry 里，
        多一份副本只会多一个可能不同步的地方。
        """
        tasks: List[str] = []
        probs: Dict[str, Optional[Dict[int, float]]] = {}
        sample_ids: Dict[str, List[int]] = {}
        for rec in self.registry.by_status(TaskStatus.PASS):
            tasks.append(rec.task_name)
            probs[rec.task_name] = rec.pass_sampling_snapshot
            sample_ids[rec.task_name] = rec.sample_ids
        return ReplayPlan(tasks=tasks, probs=probs, sample_ids=sample_ids)

    # ================================================================ #
    # 对外主循环
    # ================================================================ #
    def run(self, max_actions: Optional[int] = None) -> SchedulerState:
        n = 0
        while not self.state.finished:
            if max_actions is not None and n >= max_actions:
                break
            self.advance()
            n += 1
        return self.state

    def resume(self) -> None:
        """从存档继续跑（文档 §43.9）。

        存档里的 `finished` / `stop_reason` 是**上一次运行的结论**，恢复时不能沿用 ——
        否则「恢复」会变成「立刻结束」。其余状态（current_task / attempt_step /
        round / RNG / world / replay pool）全部原样保留。
        """
        self.state.finished = False
        self.state.stop_reason = ""

    def next_action(self) -> str:
        st = self.state
        if st.finished:
            return "finished"
        if not self.al.enabled:
            # 文档 §44「LingBot Regression」：关掉 auto_learning 时不能有任何副作用
            self._finish("auto_learning_disabled")
            return "finished"
        reason = self._budget_exhausted()
        if reason:
            self._finish(reason)
            return "finished"
        if st.bootstrap_queue:
            return "bootstrap"
        if st.pending_review:
            return "review"
        if st.current_task is None:
            return "select"
        return "train_unit"

    def advance(self) -> Dict[str, Any]:
        action = self.next_action()
        if action == "bootstrap":
            event = self._bootstrap_one()
        elif action == "review":
            event = self._review()
        elif action == "select":
            event = self._select()
        elif action == "train_unit":
            event = self._train_unit()
        else:
            event = {"action": "finish", "stop_reason": self.state.stop_reason}
        self.events.append(event)
        if self.logger is not None:
            self.logger.log_event(event)
        if self.checkpoint_path:
            from ..state.persistence import save_state

            save_state(self, self.checkpoint_path)
        return event

    # ---------------------------------------------------------------- #
    def _budget_exhausted(self) -> Optional[str]:
        st = self.state
        if self.unit_budget is not None and st.units_run >= self.unit_budget:
            return f"unit_budget_reached({self.unit_budget})"
        if self.al.max_global_steps is not None and st.global_step >= self.al.max_global_steps:
            return f"max_global_steps({self.al.max_global_steps})"
        if self.al.max_transitions is not None and st.transition_count >= self.al.max_transitions:
            return f"max_transitions({self.al.max_transitions})"
        return None

    def _finish(self, reason: str) -> None:
        self.state.finished = True
        self.state.stop_reason = reason
        if self.state.current_task is not None:
            # 没跑完的 attempt 不算 transition，退回候选池，保持状态可解释
            rec = self.registry.get(self.state.current_task)
            if rec.status == TaskStatus.TRAINING.value:
                rec.set_status(TaskStatus.CANDIDATE, f"finished_mid_attempt: {reason}")
            self.state.current_task = None

    # ================================================================ #
    # 原子动作
    # ================================================================ #
    def _bootstrap_one(self) -> Dict[str, Any]:
        al = self.al
        name = self.state.bootstrap_queue.pop(0)
        rec = self.registry.get(name)

        scout = self.evaluator.evaluate(name, EvalSplit.SCOUT.value, rec.scout_val_ids)
        rec.scout_nmse = scout.nmse
        rec.metric_valid = scout.metric_valid
        rec.last_eval_step = self.state.global_step
        self._record_eval(rec, scout, kind="scout")

        event: Dict[str, Any] = {
            "action": "bootstrap",
            "task": name,
            "scout_nmse": scout.nmse,
            "scout_trajs": scout.n_trajs,
            "metric_valid": scout.metric_valid,
            "note": scout.note,
        }

        # NaN / Inf 不能进排序（测试方案 §I01）—— 明确标 invalid 并记录，不 fail-fast
        # （一个坏任务不该把整轮跑挂掉），但它从此不会出现在 candidate 池里。
        if not is_finite_metric(scout.nmse):
            rec.metric_valid = False
            rec.set_status(
                TaskStatus.CANDIDATE,
                f"metric_invalid: scout nmse={scout.nmse!r}",
                ReasonCode.METRIC_INVALID.value,
            )
            event["result"] = "metric_invalid"
            event["note"] = f"nmse={scout.nmse!r} 非有限值，已排除出候选池"
            return event

        if not scout.metric_valid:
            rec.set_status(TaskStatus.CANDIDATE, "metric_invalid: baseline≈0，不参与排序")
            event["result"] = "metric_invalid"
            return event

        # 没有可用通过线（pass_metric="mse" 且阈值表里该任务是 null）⇒ 不训练、不消耗 attempt。
        # 与 metric_invalid 的区别：metric 本身是好的，只是没标定出线 ⇒ needs_calibration。
        if check_pass(al, name, nmse=scout.nmse, mse=scout.mse) == PassCheck.NO_THRESHOLD:
            rec.pass_line_usable = False
            rec.set_status(
                TaskStatus.CANDIDATE,
                "no_pass_threshold: 该任务没有可用通过线（needs_calibration），不参与训练",
            )
            event["result"] = "no_threshold"
            event["note"] = "没有可用通过线，已排除出候选池（不消耗 attempt）"
            return event

        # 判定走统一入口 thresholds.is_pass（口径由 cfg.pass_metric 决定；
        # mse 模式下任务没有可用阈值 ⇒ 不判 PASS，不会误放行）
        if is_pass(al, name, nmse=scout.nmse, mse=scout.mse):
            confirm = self.evaluator.evaluate(name, EvalSplit.CONFIRM.value, rec.confirm_val_ids)
            self._record_eval(rec, confirm, kind="confirm")
            event["confirm_nmse"] = confirm.nmse
            event["confirm_trajs"] = confirm.n_trajs
            if confirm.metric_valid and is_pass(al, name, nmse=confirm.nmse, mse=confirm.mse):
                apply_pass(
                    rec,
                    al,
                    f"bootstrap scout={scout.nmse:.4f} → confirm={confirm.nmse:.4f}",
                    ReasonCode.BOOTSTRAP_PASS.value,
                )
                rec.best_nmse = confirm.nmse
                rec.current_val_nmse = confirm.nmse
                rec.best_mse = confirm.mse
                rec.current_val_mse = confirm.mse
                self._mark_auto_pass(name)
                event["result"] = "pass"
                return event
            # 4 条不通过 ⇒ 用更可信的 4-val 值当 scout 估计
            rec.scout_nmse = confirm.nmse
            rec.set_status(TaskStatus.CANDIDATE, "scout 疑似达标但 4-val 确认未通过")
            event["result"] = "confirm_failed"
            return event

        rec.set_status(TaskStatus.CANDIDATE, "scout 未达标")
        event["result"] = "candidate"
        return event

    # ---------------------------------------------------------------- #
    def _select(self) -> Dict[str, Any]:
        al = self.al
        st = self.state

        cands = self.registry.candidate_records()
        if not cands:
            promoted = rollover_round(self.registry, al)
            if promoted:
                st.round += 1
                if al.rescan_candidates_after_transition:
                    self._rescan(promoted)
                return {
                    "action": "round_rollover",
                    "round": st.round,
                    "promoted": promoted,
                }
            counts = self.registry.counts()
            if counts.get("CANDIDATE", 0) == 0 and counts.get("DEFER", 0) == 0:
                self._finish("all_tasks_resolved")
            else:
                self._finish("no_eligible_candidate")
            return {"action": "finish", "stop_reason": st.stop_reason}

        cap_passed = al.max_new_tasks_passed_this_run
        if cap_passed is not None and len(st.newly_passed) >= cap_passed:
            self._finish(f"max_new_tasks_passed_reached({cap_passed})")
            return {"action": "finish", "stop_reason": st.stop_reason}

        cap_attempted = al.max_new_tasks_attempted_this_run
        if cap_attempted is not None and len(st.new_tasks_attempted) >= cap_attempted:
            # 🔴 这个额度只限制「**真正的新任务**」（第一次训练且此前从未 PASS 过）。
            # 已经 PASS 过的任务因为遗忘回炉 ⇒ 那是 **maintenance**，不吃这个额度 ——
            # 否则「启动时免费 PASS 的任务」一旦被忘掉就再也修不回来
            # （它 attempt_count==0，会被旧口径排除掉；review v0.2 P1 指出的）。
            maintenance = [r for r in cands if r.ever_passed or r.attempt_count > 0]
            if not maintenance:
                self._finish(f"max_new_tasks_attempted_reached({cap_attempted})")
                return {"action": "finish", "stop_reason": st.stop_reason}
            cands = maintenance

        pick = min(cands, key=lambda r: (r.scout_nmse, r.task_name))
        name = pick.task_name
        if name not in st.trained_tasks:
            st.trained_tasks.append(name)
        # 「新任务」= 第一次进入训练 **且** 此前从未 PASS 过（首次 attempt 才登记）
        if pick.attempt_count == 0 and not pick.ever_passed:
            if name not in st.new_tasks_attempted:
                st.new_tasks_attempted.append(name)

        start_attempt(pick)
        st.current_task = name
        st.attempt_step = 0
        st.rescued = False

        # 4 train + 4 val 的 attempt 基线（文档 §42）
        tm = self.evaluator.evaluate(name, EvalSplit.TRAIN_MONITOR.value, pick.train_monitor_ids)
        vm = self.evaluator.evaluate(name, EvalSplit.ACTIVE_VAL.value, pick.active_val_ids)
        pick.current_train_nmse = tm.nmse
        pick.current_val_nmse = vm.nmse
        pick.current_val_mse = vm.mse
        pick.prev_train_nmse = tm.nmse
        pick.prev_val_nmse = vm.nmse
        pick.train_val_gap_ratio = train_val_gap(vm.nmse, tm.nmse)
        pick.last_eval_step = st.global_step
        self._record_eval(pick, tm, kind="active_train")
        self._record_eval(pick, vm, kind="active_val")

        # hardness 扫描（一个 attempt 入口扫一次，文档 §26）
        scan = self.scanner.scan(pick, self.backend.catalog.entry(name))
        self.scans[name] = scan
        pick.sample_probs = scan.probs
        pick.hardness_version = scan.version

        return {
            "action": "select",
            "task": name,
            "step": st.global_step,
            "scout_nmse": pick.scout_nmse,
            "attempt": pick.attempt_count,
            "round": st.round,
            "baseline_train_nmse": tm.nmse,
            "baseline_val_nmse": vm.nmse,
            "hardness_version": scan.version,
            "hardness_scanned_trajs": len(scan.scanned_traj_ids),
            "hardness_coverage": round(scan.coverage, 3),
            "hardness_mean_loss": round(scan.mean_loss_scanned, 4),
            "n_samples": scan.n_total,
        }

    # ---------------------------------------------------------------- #
    def _prepare_train_request(self):
        """构造本 unit 的 `TrainRequest`（B1 的 hook 用它驱动真实 sampler）。"""
        al = self.al
        st = self.state
        name = st.current_task
        assert name is not None
        rec = self.registry.get(name)
        steps = al.eval_interval_steps

        # 🔴 一个 learning unit = `steps` 个 optimizer step，**每步重新采样一个 batch**。
        # 这件事由 trainer 负责（契约见 ports.Trainer）；scheduler 只说「跑几步」。
        request = TrainRequest(
            task=name,
            probs=dict(rec.sample_probs),
            replay=self.replay_plan(),
            start_step=st.global_step,
            batch_size=al.batch_size,
            new_slots=al.new_slots,
            replay_slots=al.replay_slots,
        )
        return request, steps

    def _train_unit(self) -> Dict[str, Any]:
        request, steps = self._prepare_train_request()
        if self.defer_train:
            # B1：训练由**外层真实循环**跑 ⇒ 只发布 request，等 `complete_train_unit()`
            self._pending_train = (request, steps)
            return {"action": "train_unit", "deferred": True, "task": request.task,
                    "steps": steps, "request": request, "step": self.state.global_step}
        result = self.trainer.train_steps(request, steps)
        return self._apply_train_result(request, steps, result)

    # -- B1：外层训练循环的回填接口 ------------------------------------------
    @property
    def pending_train_request(self):
        """当前待完成的 `TrainRequest`（没有则 None）。"""
        return None if self._pending_train is None else self._pending_train[0]

    @property
    def pending_train_steps(self) -> int:
        return 0 if self._pending_train is None else self._pending_train[1]

    def complete_train_unit(self, result: TrainResult, *,
                            allow_partial: bool = False) -> Dict[str, Any]:
        """把外层跑完的 `TrainResult` 交回来，继续做 unit 级决策。

        ``allow_partial=True``（**只在训练收尾时用**，见 hook 的 `flush_partial_unit`）：
        unit 没跑满 ⇒ 按 `result.steps` 的**实际步数**记账，让 Scheduler 的账与
        模型权重对齐（模型已经做了这 k 次 update，不能当成没发生）。
        默认 `False` ⇒ 行为与既有调用**逐位不变**。
        """
        if self._pending_train is None:
            raise RuntimeError("当前没有待完成的 train_unit（需 defer_train=True）")
        request, steps = self._pending_train
        self._pending_train = None
        if allow_partial and 0 < int(result.steps) < int(steps):
            steps = int(result.steps)
        return self._apply_train_result(request, steps, result)

    def cancel_pending_train_unit(self) -> None:
        """撤回尚未跑动（0 步）的 pending train_unit。

        用途：训练收尾 / epoch 切换时，unit 刚发布但一步都没跑 ⇒ 模型没有被更新，
        直接撤回即可（不改账）。**不要**用它丢一个已经跑过步的 unit。
        """
        self._pending_train = None

    def _apply_train_result(self, request, steps: int,
                            result: TrainResult) -> Dict[str, Any]:
        """unit 结束后的记账 + 评测 + 判定（原 `_train_unit` 的后半段）。"""
        al = self.al
        st = self.state
        name = request.task
        rec = self.registry.get(name)
        self._check_train_result(result, steps)

        st.global_step += steps
        # 真实消费的样本数 = steps × batch_size（不是 1 个 batch）
        st.global_samples_seen += result.samples_seen
        st.attempt_step += steps
        st.units_run += 1
        rec.attempt_step += steps
        rec.total_task_steps += steps

        tm = self.evaluator.evaluate(name, EvalSplit.TRAIN_MONITOR.value, rec.train_monitor_ids)
        vm = self.evaluator.evaluate(name, EvalSplit.ACTIVE_VAL.value, rec.active_val_ids)
        self._record_eval(rec, tm, kind="active_train")
        self._record_eval(rec, vm, kind="active_val")

        rec.prev_train_nmse = rec.current_train_nmse
        rec.prev_val_nmse = rec.current_val_nmse
        rec.current_train_nmse = tm.nmse
        rec.current_val_nmse = vm.nmse
        rec.current_val_mse = vm.mse
        rec.lp50 = learning_progress(rec.prev_val_nmse, rec.current_val_nmse)
        rec.lp_train = learning_progress(rec.prev_train_nmse, rec.current_train_nmse)
        gap_prev = rec.train_val_gap_ratio
        rec.train_val_gap_ratio = train_val_gap(rec.current_val_nmse, rec.current_train_nmse)
        rec.overfit = is_overfit(
            rec.lp_train,
            rec.lp50,
            rec.train_val_gap_ratio,
            gap_prev,
            train_lp_min=al.overfit_train_lp_min,
            val_lp_max=al.overfit_val_lp_max,
            gap_growth_threshold=al.overfit_gap_growth_threshold,
        )
        rec.last_eval_step = st.global_step

        # rescue 会把本 attempt 的 DEFER 阈值往后推 defer_retry_steps
        # （所以 defer_retry_steps 是**真的**在执行，不是写多少都只多跑一个单元）
        defer_after = al.min_steps_before_defer + (al.retry_budget_steps if st.rescued else 0)
        verdict = decide_after_unit(
            rec, al, attempt_step=st.attempt_step, defer_after_steps=defer_after
        )
        decision, reason = verdict.decision, verdict.detail

        row = {
            "step": st.global_step,
            "unit": st.units_run,
            "round": st.round,
            "task": name,
            "attempt": rec.attempt_count,
            "attempt_step": st.attempt_step,
            "loss": round(result.loss, 6),
            "train_nmse": rec.current_train_nmse,
            "val_nmse": rec.current_val_nmse,
            "lp50": None if rec.lp50 is None else round(rec.lp50, 6),
            "lp_train": None if rec.lp_train is None else round(rec.lp_train, 6),
            "gap": None if rec.train_val_gap_ratio is None else round(rec.train_val_gap_ratio, 4),
            "overfit": rec.overfit,
            "n_new": sum(result.new_slot_counts.values()),
            "n_old": sum(result.old_slot_counts.values()),
            "old_tasks": ",".join(sorted(result.old_slot_counts)),
            "decision": decision.value,
            "reason": reason,
            "true_nmse": _true_nmse(self.extra_state, name),
            "samples_seen": result.samples_seen,
            "batches_built": result.batches_built,
            "unique_batches": result.unique_batches,
        }
        event: Dict[str, Any] = {
            "action": "train_unit",
            "task": name,
            "step": st.global_step,
            "attempt": rec.attempt_count,
            "attempt_step": st.attempt_step,
            "loss": round(result.loss, 6),
            "train_nmse": rec.current_train_nmse,
            "val_nmse": rec.current_val_nmse,
            "lp50": None if rec.lp50 is None else round(rec.lp50, 6),
            "overfit": rec.overfit,
            "decision": decision.value,
            "reason": reason,
            "old_tasks": sorted(result.old_slot_counts),
            "samples_seen": result.samples_seen,
            "batches_built": result.batches_built,
            "unique_batches": result.unique_batches,
        }

        # ---- 先把判定落地，再把这一行写盘（这样 coverage 反映的是判定之后的状态）----
        if decision == Decision.CONTINUE:
            if al.refresh_hardness_on_continue:
                self._refresh_hardness(rec, event)
        elif decision == Decision.PASS:
            was_ever_passed = rec.ever_passed
            apply_pass(rec, al, reason, verdict.code)
            if was_ever_passed:
                # 旧任务因遗忘回炉后**恢复**通过 ⇒ 维护，不占新增额度
                if name not in st.repassed:
                    st.repassed.append(name)
            elif name not in st.newly_passed:
                st.newly_passed.append(name)
            event["snapshot_saved"] = len(rec.sample_probs)
            event["pass_sampling_version"] = rec.pass_sampling_version
            event["code"] = verdict.code
            self._after_transition(name, "PASS", reason, verdict.code)
        elif should_rescue(al, st.rescued):
            st.rescued = True
            self._refresh_hardness(rec, event)
            event["decision"] = "RESCUE"
            event["reason"] = (
                f"准备 DEFER，先换一组 hardness 分布再给 {al.defer_retry_steps} step（{reason}）"
            )
        else:
            status = apply_defer(rec, al, reason, verdict.code)
            event["status_after"] = status.value
            event["code"] = verdict.code
            self._after_transition(name, status.value, reason, verdict.code)

        summary = self.registry.summary()
        row["coverage"] = round(summary["coverage"], 6)
        row["pass_count"] = summary["pass"]
        self.metrics_rows.append(row)
        if len(self.metrics_rows) > MAX_METRIC_ROWS:
            del self.metrics_rows[: len(self.metrics_rows) - MAX_METRIC_ROWS]
        self._log_unit_metrics(row)
        return event

    # ---------------------------------------------------------------- #
    @staticmethod
    def _check_train_result(result: TrainResult, num_steps: int) -> None:
        """**Trainer 契约的运行时校验**。

        Stage B 换上真 Trainer 时，这里会第一时间抓住两类错误：
          * 没有「每步重新采样」（`batches_built != num_steps`）
          * 计数不自洽（NEW + OLD ≠ 真实消费样本数）
        """
        if result.batches_built != num_steps:
            raise RuntimeError(
                f"Trainer 只抽了 {result.batches_built} 个 batch，应为 {num_steps} —— "
                "没有做到「每个 optimizer step 重新采样」"
            )
        if result.steps != num_steps:
            raise RuntimeError(f"Trainer 报告 steps={result.steps}，应为 {num_steps}")
        n_new = sum(result.new_slot_counts.values())
        n_old = sum(result.old_slot_counts.values())
        if n_new + n_old != result.samples_seen:
            raise RuntimeError(
                f"Trainer 计数不自洽：NEW({n_new}) + OLD({n_old}) "
                f"!= samples_seen({result.samples_seen})"
            )

    def _refresh_hardness(self, rec: TaskRecord, event: Dict[str, Any]) -> None:
        """换一组 hardness 分布（文档 §15 rescue / §26 refresh_hardness_on_continue）。"""
        scan = self.scanner.scan(rec, self.backend.catalog.entry(rec.task_name))
        self.scans[rec.task_name] = scan
        rec.sample_probs = scan.probs
        rec.hardness_version = scan.version
        event["hardness_refreshed"] = scan.version

    # ---------------------------------------------------------------- #
    def _mark_auto_pass(self, task: str) -> None:
        """免费的 PASS（bootstrap 扫描 / rescan 被 transfer 带起来）—— **不占**新增技能预算。"""
        if task not in self.state.auto_passed:
            self.state.auto_passed.append(task)

    def _review(self) -> Dict[str, Any]:
        st = self.state
        st.pending_review = False
        outcomes = self.reviewer.review(self.registry, st.global_step)
        for o in outcomes:
            if o.forgotten:
                self.state.transition_count += 1
                kind = "REOPEN" if o.action == "reopen" else "REOPEN_EXHAUSTED"
                self._record_transition(o.task, kind, o.action, o.code, {"review": o.to_row()})
        return {
            "action": "review",
            "outcomes": [o.to_row() for o in outcomes],
            "reopened": [o.task for o in outcomes if o.forgotten],
            "reviewed": len(outcomes),
        }

    # ================================================================ #
    # 辅助
    # ================================================================ #
    def _after_transition(self, task: str, kind: str, reason: str, code: str = "") -> None:
        st = self.state
        st.current_task = None
        st.attempt_step = 0
        st.rescued = False
        st.transition_count += 1
        self._record_transition(task, kind, reason, code, {})
        if self.al.rescan_candidates_after_transition:
            self._rescan()
        if (
            self.al.review_after_task_transitions > 0
            and st.transition_count % self.al.review_after_task_transitions == 0
        ):
            st.pending_review = True

    def _record_transition(
        self,
        task: str,
        kind: str,
        reason: str,
        code: str,
        extra: Dict[str, Any],
    ) -> None:
        row = {
            "transition": self.state.transition_count,
            "step": self.state.global_step,
            "round": self.state.round,
            "task": task,
            "kind": kind,
            "code": code,
            "reason": reason,
        }
        row.update(extra)
        self.state.transitions.append(row)
        if len(self.state.transitions) > MAX_TRANSITION_ROWS:
            del self.state.transitions[: len(self.state.transitions) - MAX_TRANSITION_ROWS]

    def _rescan(self, only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """重新做 candidate scout（文档 §12）：训练一个任务会改变其他任务的排序。

        顺带做一次「免费 PASS 检测」：EXHAUSTED 任务如果被 transfer 带起来了，
        直接 PASS（不再消耗训练预算）。
        """
        al = self.al
        rows: List[Dict[str, Any]] = []
        for rec in self.registry:
            if rec.task_name == self.state.current_task:
                continue
            if rec.status not in (
                TaskStatus.CANDIDATE.value,
                TaskStatus.DEFER.value,
                TaskStatus.EXHAUSTED.value,
            ):
                continue
            if not rec.pass_line_usable:
                continue
            if only is not None and rec.task_name not in only:
                continue
            scout = self.evaluator.evaluate(
                rec.task_name, EvalSplit.SCOUT.value, rec.scout_val_ids
            )
            rec.scout_nmse = scout.nmse
            # 🔴 不能用裸的 `scout.metric_valid` —— 评测器可能把 metric_valid 置 True
            # 却给出 NaN/Inf。rescan 若把这种任务「重新洗白」回候选池，
            # 它就会带着 inf 进 `min(...)`（测试方案 §I01 抓出来的真 bug）。
            rec.metric_valid = scout.metric_valid and is_finite_metric(scout.nmse)
            rec.last_eval_step = self.state.global_step
            self._record_eval(rec, scout, kind="rescan")
            row = {"task": rec.task_name, "nmse": scout.nmse, "status": rec.status}
            if (
                rec.metric_valid
                and is_finite_metric(scout.nmse)
                # 判定走统一入口（口径由 cfg.pass_metric 决定；
                # mse 模式下任务没有可用阈值 ⇒ 不判 PASS）
                and is_pass(al, rec.task_name, nmse=scout.nmse, mse=scout.mse)
                # 🔴 churn guard 已触发的任务**不能被 rescan 复活**：否则会出现
                # 「EXHAUSTED → 免费 PASS → 又遗忘 → 回炉 → EXHAUSTED → …」的循环，
                # 每次都不消耗 attempt（Monte Carlo 压力测试抓出来的）。
                and not is_terminal(rec, al)
            ):
                confirm = self.evaluator.evaluate(
                    rec.task_name, EvalSplit.CONFIRM.value, rec.confirm_val_ids
                )
                self._record_eval(rec, confirm, kind="rescan_confirm")
                row["confirm_nmse"] = confirm.nmse
                if confirm.metric_valid and is_pass(
                    al, rec.task_name, nmse=confirm.nmse, mse=confirm.mse
                ):
                    apply_pass(
                        rec,
                        al,
                        f"rescan 自动达标 scout={scout.nmse:.4f} → confirm={confirm.nmse:.4f}",
                        ReasonCode.RESCAN_PASS.value,
                    )
                    rec.best_nmse = confirm.nmse
                    rec.current_val_nmse = confirm.nmse
                    rec.best_mse = confirm.mse
                    rec.current_val_mse = confirm.mse
                    self._mark_auto_pass(rec.task_name)
                    row["auto_pass"] = True
            rows.append(row)
        return rows

    def _record_eval(self, rec: TaskRecord, metrics, kind: str) -> None:
        self.state.eval_events += 1
        rec.note_eval(
            {
                "step": self.state.global_step,
                "kind": kind,
                "split": metrics.split,
                "n_trajs": metrics.n_trajs,
                "nmse": metrics.nmse,
                "mse": metrics.mse,
                "baseline_mse": metrics.baseline_mse,
            }
        )
        self.heatmap_rows.append(
            {
                "event": self.state.eval_events,
                "step": self.state.global_step,
                "task": rec.task_name,
                "kind": kind,
                "nmse": metrics.nmse,
            }
        )
        if self.logger is not None:
            self.logger.log_metrics(self.state.global_step, f"task/{rec.task_name}/{kind}_nmse", metrics.nmse)
            self.logger.log_metrics(
                self.state.global_step, f"debug/{rec.task_name}/{kind}_mse", metrics.mse
            )
            self.logger.log_metrics(
                self.state.global_step, f"debug/{rec.task_name}/baseline_mse", metrics.baseline_mse
            )

    def _log_unit_metrics(self, row: Dict[str, Any]) -> None:
        if self.logger is None:
            return
        step = row["step"]
        lg = self.logger
        lg.log_metrics(step, "training/loss", row["loss"])
        lg.log_metrics(step, "current_skill/task_id", self._task_index(row["task"]))
        lg.log_metrics(step, "current_skill/attempt", row["attempt"])
        lg.log_metrics(step, "current_skill/attempt_step", row["attempt_step"])
        lg.log_metrics(step, "current_skill/train_nmse", row["train_nmse"])
        lg.log_metrics(step, "current_skill/val_nmse", row["val_nmse"])
        lg.log_metrics(step, "current_skill/lp50", row["lp50"])
        lg.log_metrics(step, "current_skill/train_val_gap_ratio", row["gap"])
        lg.log_metrics(step, "current_skill/overfit", 1.0 if row["overfit"] else 0.0)
        lg.log_metrics(step, "system/global_samples_seen", self.state.global_samples_seen)
        lg.log_metrics(step, "system/units_run", self.state.units_run)
        lg.log_metrics(step, "system/replay_slots", row["n_old"])
        lg.log_metrics(step, "system/replay_unique_tasks", len(set(row["old_tasks"].split(","))) if row["old_tasks"] else 0)
        summary = self.registry.summary()
        lg.log_metrics(step, "skill_overview/pass_count", summary["pass"])
        lg.log_metrics(step, "skill_overview/candidate_count", summary["candidate"])
        lg.log_metrics(step, "skill_overview/defer_count", summary["defer"])
        lg.log_metrics(step, "skill_overview/exhausted_count", summary["exhausted"])
        lg.log_metrics(step, "skill_overview/coverage", summary["coverage"])
        lg.log_metrics(step, "skill_overview/median_scout_nmse", summary["median_val_nmse"])
        lg.log_metrics(step, "skill_overview/worst_scout_nmse", summary["worst_val_nmse"])
        lg.log_metrics(step, "skill_overview/current_round", self.state.round)
        lg.log_metrics(step, "memory/pass_pool_size", len(self.registry.by_status(TaskStatus.PASS)))
        lg.log_metrics(step, "memory/forgotten_count", sum(1 for r in self.registry if r.forgotten))
        lg.log_metrics(step, "memory/reopen_count", sum(r.reopen_count for r in self.registry))

    def _task_index(self, task: str) -> float:
        try:
            return float(self.registry.names().index(task))
        except ValueError:
            return -1.0

    # ================================================================ #
    def final_report(self) -> Dict[str, Any]:
        summary = self.registry.summary()
        vals = [r.current_val_nmse for r in self.registry if r.current_val_nmse is not None]
        return {
            "stop_reason": self.state.stop_reason,
            "global_step": self.state.global_step,
            "units_run": self.state.units_run,
            "global_samples_seen": self.state.global_samples_seen,
            "rounds": self.state.round,
            "transitions": self.state.transition_count,
            "trained_tasks": list(self.state.trained_tasks),
            "new_tasks_attempted": list(self.state.new_tasks_attempted),
            "newly_passed": list(self.state.newly_passed),
            "repassed": list(self.state.repassed),
            "auto_passed": list(self.state.auto_passed),
            "registry": summary,
            "median_val_nmse": None if not vals else round(median(vals) or 0.0, 5),
            "worst_val_nmse": None if not vals else round(max(vals), 5),
            "true_state": self.extra_state.snapshot_table() if self.extra_state else [],
        }

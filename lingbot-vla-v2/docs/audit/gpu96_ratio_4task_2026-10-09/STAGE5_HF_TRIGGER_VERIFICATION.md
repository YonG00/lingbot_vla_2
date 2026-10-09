# 阶段 5（BF16 HF 导出）触发条件核实（无卡）+ 最小补丁

基线：`cf81d04`（本轮补丁后提交见文末）

## 一、生产触发链（逐行）

| 环节 | 位置 | 规则 |
|---|---|---|
| 参数校验 | `tasks/vla/train_lingbotvla.py:829-845` | `hf_pass_interval` 需 AL 开启、`smoke_no_checkpoint=false`、`save_hf_weights=false`、`async=false`、`dcp_save_mode=always`、`save_steps>0`、`save_epochs=0`，否则启动即报错 |
| **Bootstrap 排除** | `lingbotvla/utils/al_checkpoint_policy.py::learned_nonbootstrap_count` | `len((newly_passed ∪ auto_passed) − bootstrap_passed)` ⇒ **Bootstrap PASS 明确不计入** ✓（旧 DCP 无法区分时只计"显式训练出的首次 PASS"） |
| 里程碑序号 | `pass_milestone_index(state, interval)` | `learned_nonbootstrap_count(state) // interval` |
| 是否导出 | `milestone_action(observed, committed, final_due, dcp_due)` | `observed <= committed ⇒ none`；`final_due or dcp_due ⇒ covered_by_dcp`；否则 **`hf`** |
| 导出调用 | `train_lingbotvla.py:1639-1653` | `action=="hf"` ⇒ 日志 `[ckpt] N 个新增 PASS ⇒ step X 直接导出 HF（不写 DCP）` ⇒ `export_model_hf_direct(...)` ⇒ **成功后**才 `_hf_milestone_index = observed` |
| DCP 覆盖 | `:1654-1655, 1709-1716` | `covered_by_dcp` ⇒ 待 DCP 存档成功后提交里程碑序号（final DCP 同样覆盖） |
| Resume | `:889-890`（保存 `hf_pass_milestone_index`）、`:1132-1138`（恢复并取 max） | 精确续训不重复导出同一里程碑 |

## 二、为什么"自然触发"不可保证（实测支持）

隔离配置原为 `click_bell / click_alarmclock / adjust_bottle / press_stapler`（历史 NMSE 0.034 / 0.102 / ~0.1–0.7 / 0.089，**全部 < 1.66**）且 `target_total_passed_tasks=None`
⇒ Bootstrap 会把 4 个任务全部 PASS ⇒ `all_tasks_resolved` ⇒ **零训练步收工**（与 2026-10-09 08:09 那次 `ratio-gpu` 实测完全同型：2 任务 Bootstrap 2/2 PASS ⇒ `zero-step`）。
而里程碑判定点位于**逐步存档块内**（`_save_due` 逻辑，`train_lingbotvla.py:1597-1636`）⇒ **没有优化步就永远到不了判定点** ⇒ 不会导出 ✗
⇒ 结论：**不能依赖随机任务 PASS**；必须同时满足「≥1 优化步」与「一次性的确定性触发」。

## 三、合法的确定性触发入口（已有，无需碰 Scheduler/PASS）

`tools/hf_direct_export_acceptance.py:37-48` `_install_decision_patch()`：

```python
def one_shot_action(**kwargs):
    if not FIRED["done"]:
        FIRED["done"] = True
        return "hf"          # 第一次里程碑判定 → 强制走真实生产导出
    return "none"            # 之后一律 none ⇒ 恰好一次
policy.milestone_action = one_shot_action
```
- 只替换**"要不要导"这一个决策**；**Scheduler / Registry / PASS 事件完全不动**（不伪造任何 PASS ✓）
- 导出实现用生产原版 `export_model_hf_direct`（外层仅加计时与快照取证）
- 出口 `_verify_export()`：逐分片分块比对活模型快照，要求 **无缺/无多/形状一致/最大绝对差 == 0**，否则退出码 **1** ✓

## 四、本轮最小补丁（已提交，含 CPU 测试）

`tools/gpu96_acceptance.py` 新增三个纯函数并接线到 `run_hf`：

1. `build_hf_smoke_config(base, *, step_offset, steps)`：隔离配置改为
   `task_names = [click_bell, click_alarmclock, turn_switch, put_object_cabinet]`
   （**含历史不通过任务**：turn_switch 1.762~1.763 四次稳定、put_object_cabinet 2.167）+
   `pass_metric='nmse'`、`pass_nmse=1.66`（显式）、`pass_thresholds_file=None`、
   **`target_total_passed_tasks = len(tasks)`** ⇒ 调度器必须尝试未通过任务 ⇒ **至少消费 1 个 Train Unit** ⇒ 判定点可达 ✓
2. `hf_step_evidence(out_dir)`：从 `auto_learning_events.jsonl` 的 `system/units_run` 与 `train_hf.log` 的 `Step:` 取证（读不到即 0，fail-closed）
3. `hf_verdict(...)`：零步 ⇒ FAIL 并给出 **`no_optimizer_step_so_milestone_decision_point_never_reached`**；
   里程碑 ≠1 / 无分片 / `5 < size_gib < 19` / DCP 泄漏 / 出口非零 ⇒ 各自 FAIL 原因；全部满足才 `verified` ✓
   `result.json` 新增 `verdict_reason`、`units_run`、`max_global_step_seen`

**CPU 测试**：`tests/test_gpu96_acceptance.py` 新增 9 例（配置强制训练且不改源、判定矩阵 6 例、证据解析 2 例）⇒
本文件 **35 passed**；相关专项 127 passed；全量 **743 passed / 18 skipped / 0 failed**；CPU Gate ✅(9/0/9)。

## 五、补丁后的保证与残余边界

- **保证**：判定点必定可达（≥1 Train Unit）✓；触发恰好一次（一次性 latch ✓）；导出必为真实生产路径 ✓；数值不一致或尺寸越界或 DCP 泄漏或零步 ⇒ **FAIL 而非假 PASS** ✓
- **边界（如实）**：① 不能保证导出发生在第 1 步（发生在**首个可达的判定步**，通常为第 1–3 步）；② 若出现未预料的提前结束路径，工具会以明确原因 FAIL，不会静默通过 ✓；③ 仍建议保留 **~24 GiB** 磁盘余量（防 Scheduler 提前结束写 final DCP）✓；④ 正式 `pass_metric='nmse'`、正式 YAML、DCP/HF 策略**均未改动** ✓

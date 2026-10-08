# B. Replay 对账（Bootstrap 两个 PASS 有没有进 Replay 池）

事件行数: 114

## bootstrap（谁 PASS 了）
- click_bell: result=pass pass_source=scout_direct scout_nmse=0.03415933310667733
- click_alarmclock: result=pass pass_source=scout_direct scout_nmse=0.06870544595554436
- turn_switch: result=candidate pass_source=None scout_nmse=1.7629511377779599
- place_container_plate: result=candidate pass_source=None scout_nmse=0.15683109648522375

## 每个 unit / select 的全部相关字段

```json
{
 "action": "select",
 "task": "place_container_plate",
 "step": 0,
 "tb_step": 500
}
```

```json
{
 "action": "train_unit",
 "task": "place_container_plate",
 "step": 0,
 "tb_step": 500,
 "steps": 5,
 "deferred": true
}
```

```json
{
 "action": "select",
 "task": "turn_switch",
 "step": 5,
 "tb_step": 505
}
```

```json
{
 "action": "train_unit",
 "task": "turn_switch",
 "step": 5,
 "tb_step": 505,
 "steps": 5,
 "deferred": true
}
```

## `memory/pass_pool_size` 等 TB 指标
- `memory/pass_pool_size`: [(505, 2.0), (510, 2.0)]
- `memory/reopen_count`: [(505, 0.0), (510, 0.0)]
- `memory/forgotten_count`: [(505, 0.0), (510, 0.0)]

## 结论（自动判定，人工复核）
- unit task=place_container_plate step=500: n_old=None old_tasks/replay_plan=None
- unit task=turn_switch step=505: n_old=None old_tasks/replay_plan=None

⇒ 若所有 unit 的 `n_old` 都为 0 或 `old_tasks` 为空，则**Bootstrap PASS 的 click_bell/click_alarmclock 未被采样**；下一步按 design 追查：`Scheduler.replay_plan()` 是否要求 PASS 时已存在 snapshot（`replay_sample_policy=pass_snapshot`），以及 `build.py` 里 replay 池的填充时机（初始化 vs PASS 时）。**发现 Bug 只报告，不改 Scheduler。**

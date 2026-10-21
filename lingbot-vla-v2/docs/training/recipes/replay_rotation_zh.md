# 10%旧任务复习：每N步轮换一个任务

90%主任务样本搭配10%复习样本，每次只复习一个旧任务。
R11启动时44项本地/服务器CPU契约、19任务950条实际CPU装配及三步混合GPU短测通过，正式1200步进行中。
短测只覆盖首个主任务阶段，完整多阶段和最终闭环仍待完成；历史R10未使用复习。

## 参数

```json
"replay": {
  "fraction": 0.1,
  "mode": "round_robin",
  "switch_every_steps": 10,
  "pool_file": "refs/training_records/replay_pool_9680_through_R10.json"
}
```

- fraction按全局batch样本数计算，不改变micro/GBS/GAS。GBS56时每步5或6条复习，其余50或51条主任务。
- switch_every_steps单位是优化器更新步数；1表示每步换，10表示同一任务连续10步，每块恰好56条复习样本。
- 每步复习只来自一个任务；任务内均匀抽帧、有放回。任务按池中顺序循环，不按数据量分配；末尾可能不足完整一轮。
- 排除当前主任务，不排除后续课程的其他任务。切换主任务时若与当前复习任务冲突，立即跳到下一个候选。
- 只支持sequential/task_units；mixed模式拒绝轮换规则。候选池不能被当前主任务全部排空。
- round_robin要求任务weight=1；省略mode仍使用旧sample模式和样本权重，旧行为不变。

复习资格与补课独立，框架目前仍拒绝“复习与自动补课同时开启”，本次未取消该限制。
默认闭环只测主课程任务；需要复测旧任务时显式配置evaluation.tasks，不隐式增加测评开销。

## 当前14个候选及证据

历史任意一次完整闭环≥1/3即可，整数比较为successes×3≥episodes。
4回合中的1/4不达标；历史成功只代表复习资格，不代表当前模型成功率，后续退化不会自动移出池。

| 顺序 | 任务 | 纳入依据 |
|---|---|---|
| 1 | click_alarmclock | 用户提供9680截图：4/4 |
| 2 | press_stapler | 用户提供9680截图：4/4 |
| 3 | click_bell | 用户提供9680截图：3/4 |
| 4 | open_microwave | 用户提供9680截图：3/4 |
| 5 | shake_bottle | 用户提供9680截图：3/4 |
| 6 | open_laptop | 用户提供9680截图：2/4 |
| 7 | place_object_basket | 用户提供9680截图：2/4 |
| 8 | shake_bottle_horizontally | 用户提供9680截图：2/4 |
| 9 | adjust_bottle | R3：3/3 |
| 10 | place_container_plate | 9680截图1/4不达标；后续R6达2/3 |
| 11 | place_empty_cup | R6：1/3 |
| 12 | turn_switch | R9：2/3 |
| 13 | move_playingcard_away | R10：1/3 |
| 14 | rotate_qrcode | R10：1/3 |

place_a2b_right仅1/4，保留证据但不纳入。截图标记为用户会话图片，不冒充服务器原始报告。
[最新累计证据注册表](../../../refs/training_records/replay_pool_9680_through_R10.json)；
[截至R9的12任务旧快照](../../../refs/training_records/replay_pool_9680_and_R9.json)保留不改。
R10的place_phone_stand、place_object_scale均0/3，证据保留但不进入候选池。

## 冻结与累计

启动时加载pool_file，按证据生成tasks，将完整pool_snapshot写入冻结recipe与哈希。
外部注册表后续更新不改变已经准备好的run，不热改训练中的比例或任务池。
每轮闭环后可导入完整报告生成下一版注册表，再随文档更新提交；不会覆盖原证据：

```bash
python -m experiment.task_recipes.replay_pool \
  --registry refs/training_records/replay_pool_9680_through_R10.json \
  --summary <完整闭环summary.json> \
  --output <下一版注册表.json>
```

相同报告重复导入保持不变，同一来源冲突或未完整评测均报错。
下一轮配方指定新注册表路径，当前不会在训练后台自动修改Git注册表。

示例：[四主任务＋10%轮换复习](../../../configs/training_recipes/sequential_four_tasks_replay10pct_rr10_example.json)。
默认预览，不启动；正式开跑仍必须通过真实CPU整链与三步GPU检查。
本例800步共44800样本，复习4480条；14候选在无当前主任务冲突时每140步轮完。
本例主任务playingcard/qrcode也在候选池，各自主训阶段会排除自身。
主任务配额仍按更新步数计数，不能将复习比例另算成完整训练步数。

验证入口：`python3 -m unittest experiment.task_recipes.test_replay_rotation`。

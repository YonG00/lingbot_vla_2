# 旧任务轮换复习：比例与每步任务数

原有默认是90%主任务搭配10%复习样本、每次一个旧任务。
R13显式改为80%主任务＋两个不同旧任务各10%，每10步轮换一对，排除当前主任务。
R11完整1200步与七任务闭环、R12完整1350步与十任务闭环均已完成；历史R10未使用复习。
最新累计资格池19任务，through_R13已合入R13完整闭环。R12实际冻结14任务、R13实际冻结17任务，不因事后评测修改历史记录。

## 参数

```json
"replay": {
  "fraction": 0.2,
  "mode": "round_robin",
  "tasks_per_step": 2,
  "switch_every_steps": 10,
  "pool_file": "refs/training_records/replay_pool_9680_through_R13.json"
}
```

- fraction按全局batch样本数计算，不改变micro/GBS/GAS。GBS56时10%为每步5/6条；20%为11/12条复习，其余44/45条主任务。
- tasks_per_step默认1，保持旧冻结计划；本轮显式2，两个不同任务平分复习比例，每个10%。样本数按累计整数配额分配，每步各5/6条，每10步各56条；本轮1250步每槽7000条，共14000。
- switch_every_steps单位是优化器更新步数；1每步换，10表示固定当前任务或任务对10步再换下一组。
- 任务内均匀抽帧、有放回。任务按池中顺序循环，不按数据量分配；双任务逐对推进，末尾可能不足完整一轮。
- 排除当前主任务，不排除后续课程的其他任务。切换主任务时若与当前复习任务冲突，立即跳到下一个候选。
- 只支持sequential/task_units；mixed模式拒绝轮换规则。排除主任务后候选数至少为tasks_per_step；每步复习样本足以覆盖每个选中任务。
- round_robin要求任务weight=1；省略mode仍使用旧sample模式和样本权重，旧行为不变。

R13双任务复习：本地及服务器55项启动CPU、真实1250条280122帧CPU、三步GPU及完整1250步/十任务正式闭环完成；结果读流水。
旧R12冻结完整计划与改动前相同（hash见改动台账），不追改既有训练记录。

复习资格与补课独立，框架目前仍拒绝“复习与自动补课同时开启”，本次未取消该限制。
默认闭环只测主课程任务；需要复测旧任务时显式配置evaluation.tasks，不隐式增加测评开销。

## 当前19个候选及证据

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
| 8 | shake_bottle_horizontally | 用户提供9680截图：2/4；R12：3/3 |
| 9 | adjust_bottle | R3：3/3 |
| 10 | place_container_plate | 9680截图1/4不达标；后续R6达2/3 |
| 11 | place_empty_cup | R6：1/3 |
| 12 | turn_switch | R9：2/3 |
| 13 | move_playingcard_away | R10：1/3；R11：3/3 |
| 14 | rotate_qrcode | R10：1/3；R11：2/3 |
| 15 | dump_bin_bigbin | R12：2/3 |
| 16 | stack_blocks_two | R12：2/3 |
| 17 | stack_bowls_two | R12：1/3 |
| 18 | place_can_basket | R13：1/3 |
| 19 | place_bread_basket | R13：2/3 |

place_a2b_right仅1/4，保留证据但不纳入。截图标记为用户会话图片，不冒充服务器原始报告。
[最新累计证据注册表](../../../refs/training_records/replay_pool_9680_through_R13.json)；
[R12训练实际冻结的through_R11旧池](../../../refs/training_records/replay_pool_9680_through_R11.json)；
[截至R9的12任务旧快照](../../../refs/training_records/replay_pool_9680_and_R9.json)保留不改。
R10的place_phone_stand、place_object_scale均0/3，证据保留但不进入候选池。

## 冻结与累计

启动时加载pool_file，按证据生成tasks，将完整pool_snapshot写入冻结recipe与哈希。
外部注册表后续更新不改变已经准备好的run，不热改训练中的比例或任务池。
每轮闭环后可导入完整报告生成下一版注册表，再随文档更新提交；不会覆盖原证据：

```bash
python -m experiment.task_recipes.replay_pool \
  --registry refs/training_records/replay_pool_9680_through_R13.json \
  --summary <完整闭环summary.json> \
  --output <下一版注册表.json>
```

相同报告重复导入保持不变，同一来源冲突或未完整评测均报错。
下一轮配方指定新注册表路径，当前不会在训练后台自动修改Git注册表。

示例：[四主任务＋10%轮换复习](../../../configs/training_recipes/sequential_four_tasks_replay10pct_rr10_example.json)。
默认预览，不启动；正式开跑仍必须通过真实CPU整链与三步GPU检查。
本例保留through_R10的14候选历史配置，800步共44800样本，复习4480条；无当前主任务冲突时每140步轮完。
若新配方使用through_R12的17候选且仍每10步轮换，无冲突时每170步轮完。
使用through_R13的19候选、双任务每10步轮换，在无当前主任务冲突时190步覆盖两次候选槽位（每个候选两次）。
本例主任务playingcard/qrcode也在候选池，各自主训阶段会排除自身。
主任务配额仍按更新步数计数，不能将复习比例另算成完整训练步数。

验证入口：`python3 -m unittest experiment.task_recipes.test_replay_rotation`。

最新R16冻结池为`refs/training_records/replay_pool_9680_through_R15.json`，20个历史≥1/3任务，新增R14的place_shoe（2/3）；R15两个0/3不增加资格。R16按用户指定10%比例、tasks_per_step=1、switch_every_steps=5，20任务一轮100更新；800步每个旧任务预计224样本，仍以实际完成步数为准。

R16完成后，累计资格池更新为`refs/training_records/replay_pool_9680_through_R16.json`，20→24，新增beat_block_hammer、hanging_mug、place_a2b_left、place_object_stand；本轮冻结池仍through_R15。

R16后置抽检完成后资格池另存`refs/training_records/replay_pool_9680_through_R16_retention3.json`，三个旧任务均1/3，无新增资格，仍24任务；今后需要最新测评来源时使用此文件，不追改旧训练冻结池。

R17首批5项完整测后，登记表另存`refs/training_records/replay_pool_9680_through_R17_first5.json`。历史≥1/3资格仍24项，曾成功任务仍25项；最新0/3不删除旧资格，后20项没有本轮测评记录。

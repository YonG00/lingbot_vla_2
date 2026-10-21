# 任务训练步数与评测对照

任务按本轮实验中首次参与训练的顺序排列。每次训练及其评测占一列，补课另占一列。
取证截止2026-10-21：R15训练及正式闭环完成，两任务均0/3；完整时间、来源和路径见[流水记录](training_history_zh.md)。
不计原始9680内未知的逐任务步数；CPU检查及三步GPU探针不计入正式训练。
“累计”沿模型真实继承链计算，不把独立实验分支相加。只测未训记为`+0`；未测不能当失败。

## 早期继承链与放弃的补课分支

**9680 → R3-adjust200 → R5-plate200 → R6-顺序500 → R7-补课300（已完成，后续放弃）**。
当前主链为**9680 → R3 → R5 → R6 → R8 → R9 → R10 → R11 → R12 → R13**；R7更新不计入当前模型。
单元格为“本次该任务步数 / 沿继承链累计步数；该模型测评”。

| 首训顺序 / 任务 | R3：adjust200 | R5：plate200 | R6：顺序500 | R7：补课1（已300/300） |
|---|---|---|---|---|
| 1 adjust_bottle | +200 / **200**；3/3 | +0 / 200；未测 | +0 / 200；未测 | +0 / **200**；本轮未安排测 |
| 2 place_container_plate | +0 / 0；未测 | +200 / **200**；1/3 | +100 / **300**；2/3 | +0 / **300**；0/3 |
| 3 place_empty_cup | +0 / 0；未测 | +0 / 0；未测 | +200 / **200**；1/3 | +100 / **300**；0/3 |
| 4 move_can_pot | +0 / 0；未测 | +0 / 0；未测 | +200 / **200**；0/3 | +200 / **400**；0/3 |

R7已完成累计adjust200、plate300、cup300、pot400；三个复测任务均0/3，用户放弃R7模型。后续R8回到R6，而不是继承R7。

**R7对应run**：`/data/outputs/sequential_plate100_cup200_pot200_from_plate200_bf16_m56_20261020/remediation_round_1`。
闭环汇总在其`evaluation_300/summary.json`，评测模型在`checkpoints/global_step_300/hf_ckpt`。
逐任务日志/视频及R6与R7路径区别见[流水中的R7路径说明](training_history_zh.md#r7评测复核)。
R7仅cup/pot训练，最终复测plate/cup/pot用于检查plate是否遗忘，不包含adjust_bottle。

## 独立对照与废弃分支

这些结果保留用于分析，但不计入当前模型的继承累计。
R1/R2的250与500列是同一次训练的中间/最终快照，500已包含前250，不能相加成750。

| 任务 | R1标准FP32 step250 | R1标准FP32 step500 | R2增强FP32 step250 | R2增强FP32 step500 | R4三任务混训400（来源R3） |
|---|---|---|---|---|---|
| adjust_bottle | 累计250；2/3 | 累计500；2/3 | 累计250；1/3 | 累计500；2/3 | 本轮+0、继承200；2/3 |
| place_container_plate | 未训/未测 | 未训/未测 | 未训/未测 | 未训/未测 | 约133.3等效步（期望）；0/3 |
| place_empty_cup | 未训/未测 | 未训/未测 | 未训/未测 | 未训/未测 | 约133.3等效步（期望）；0/3 |
| move_can_pot | 未训/未测 | 未训/未测 | 未训/未测 | 未训/未测 | 约133.3等效步（期望）；完整评测中断 |
| click_bell（仅评测） | 未训/未测 | 未训/未测 | 未训/未测 | 未训/未测 | 本轮+0；3/3 |

混训400步每个batch可含多个任务；每任务样本概率1/3。
`400×1/3≈133.3`只是按GBS56折算的期望曝光步数，不是实测独立优化器更新次数，不能与单任务步数混作精确累计。

## 后续续训分支

### R8 → R9续训分支

R8已完成600步及三任务闭环，R9从其最终600续训并完成500步及闭环。

| 首训顺序 / 任务 | R6继承累计 | R8本轮 / 累计；最终评测 | R9本轮 / 累计；最终评测 |
|---|---:|---|---|
| 1 adjust_bottle | 200 | +0 / 200；未测 | +0 / 200；未安排 |
| 2 place_container_plate | 300 | +0 / 300；未测 | +0 / 300；未安排 |
| 3 place_empty_cup | 200 | +0 / 200；未测 | +0 / 200；未安排 |
| 4 move_can_pot | 200 | +0 / 200；未测 | +0 / 200；未安排 |
| 5 turn_switch | 0 | **+200 / 200；1/3** | **+100 / 300；2/3** |
| 6 move_pillbottle_pad | 0 | **+200 / 200；0/3** | **+200 / 400；0/3** |
| 7 move_stapler_pad | 0 | **+200 / 200；0/3** | **+200 / 400；0/3** |

R7的cup+100/pot+200不计入R8继承累计。未完成的配额只记计划。
未来补课：2/3与1/3各100步，2/3优先；0/3、3/3不补；默认最多一轮。R8未执行自动补课，R9继续关闭。
R9 run：`/data/outputs/20261020_2058_stapler200_pillbottle200_switch100_from_R8_bf16_m56`。
R9按任务保存200/400/500，最终仅测上述三个实际训练任务各3回合。

### R10 四任务新课程（800步与闭环已完成）

2026-10-21终态复核：来源R9-step500，四任务各200步已完成；未继承被替换三任务流程的更新。
R9已训练的任务累计保持不变：adjust200、plate300、cup200、pot200、switch300、pillbottle400、stapler400。

| 首训顺序 / 新任务 | R9继承累计 | R10本轮实际 / 当前累计 | 本轮配额 | 最终评测 |
|---|---:|---|---:|---|
| 8 move_playingcard_away | 0 | **200 / 200** | 200 | **1/3** |
| 9 place_phone_stand | 0 | **200 / 200** | 200 | 0/3 |
| 10 place_object_scale | 0 | **200 / 200** | 200 | 0/3 |
| 11 rotate_qrcode | 0 | **200 / 200** | 200 | **1/3** |

run：`/data/outputs/20261020_2247_playingcard_phone_scale_qrcode_from_R9_bf16_m56_200each`。
按任务200/400/600/800四份存档均已校验，补课未执行。[存档保留清单](checkpoint_inventory_zh.md)。
playingcard与qrcode已加入历史≥1/3复习池，累计候选由12变14；phone与scale仅0/3，不加入。

### R11完成列（来源R10-step800，1200步已完成）

| 主任务 | R10继承累计 | R11实际主训 | 当前主训累计 | 最终评测 |
|---|---:|---:|---:|---|
| rotate_qrcode | 200 | 100 | 300 | 2/3 |
| move_playingcard_away | 200 | 100 | 300 | 3/3 |
| pick_dual_bottles | 0 | 200 | 200 | 0/3 |
| pick_diverse_bottles | 0 | 200 | 200 | 0/3 |
| grab_roller | 0 | 200 | 200 | 0/3 |
| lift_pot | 0 | 200 | 200 | 0/3 |
| stamp_seal | 0 | 200 | 200 | 0/3 |

另有14个历史任务轮换复习，实际合计6720样本曝光；不增加这些任务的“主训优化器累计步数”。
本轮主训只占90%样本，不能与以往100%主任务batch的200步简单视为相同曝光量。
run路径与阶段顺序见[流水R11](training_history_zh.md#r11r10-step800续训七任务10轮换复习)。

### R12完成列（来源R11-step1200，1350步与闭环完成）

| 任务 | R11完成主训累计；评测 | R12实际主训 / 累计；评测 |
| --- | --- | --- |
| shake_bottle_horizontally | 0；本轮未测，9680历史2/4 | +150 / 150；3/3 |
| pick_dual_bottles | 200；0/3 | +100 / 300；0/3 |
| pick_diverse_bottles | 200；0/3 | +100 / 300；0/3 |
| put_bottles_dustbin | 0；未测 | +150 / 150；0/3 |
| dump_bin_bigbin | 0；未测 | +150 / 150；2/3 |
| place_container_plate | 300；本轮未测，R6历史2/3 | +100 / 400；0/3 |
| stack_bowls_two | 0；未测 | +150 / 150；1/3 |
| stack_bowls_three | 0；未测 | +150 / 150；0/3 |
| blocks_ranking_size | 0；未测 | +150 / 150；0/3 |
| stack_blocks_two | 0；未测 | +150 / 150；2/3 |

0表示9680之后本实验链中可归属的主训步数；不含9680未知曝光及复习样本。
R12完成1350步，7560复习样本；本轮冻结资格池使用累计历史及R11结果，14任务。
最终仅1350模型测上述10任务各3回合，6＋4分批，共享一份模型，合计8/30成功。
累计资格池已另存through_R12，14→17任务，新增dump_bin_bigbin、stack_bowls_two、stack_blocks_two；历史合格任务不因本次失败移除。
R12实际使用的through_R11冻结池仍保留，不把事后新增任务冒充已参加本轮复习。

## 全部已登记实验的训练投入

### R13完成列（1250步与正式闭环完成）

| 任务 | R12完成主训累计；评测 | R13实际主训 / 累计；最终评测 |
| --- | --- | --- |
| blocks_ranking_size | 150；0/3 | +100 / 250；0/3 |
| move_stapler_pad | 400；R12未测，R9历史0/3 | +100 / 500；0/3 |
| pick_dual_bottles | 300；0/3 | +100 / 400；0/3 |
| place_container_plate | 400；0/3 | +100 / 500；0/3 |
| stack_bowls_two | 150；1/3 | +100 / 250；0/3 |
| blocks_ranking_rgb | 0；未测 | +150 / 150；0/3 |
| place_bread_skillet | 0；未测 | +150 / 150；0/3 |
| place_can_basket | 0；未测 | +150 / 150；1/3 |
| place_cans_plasticbox | 0；未测 | +150 / 150；0/3 |
| place_bread_basket | 0；未测 | +150 / 150；2/3 |

已完成1250步，旧任务合计20%复习，每步两个任务各10%；复习曝光不算额外主训优化器步数。
本轮冻结R12累计17任务池，每槽7000样本；最终1250模型十任务各3回合、5＋5分批，合计3/30成功。
事后累计资格池更新为19任务，新增place_can_basket、place_bread_basket，不覆盖历史冻结池。
独立八路单回合试跑没有新增训练步数；完整来源、run路径和保存点见[流水R13](training_history_zh.md)。

### 已完成的训练投入

下表跨分支汇总计算投入，用于看花了多少训练；不表示任何一个模型继承了这些全部更新。已计入R13完成步数。

| 任务（按首次训练顺序） | 可精确归属的单任务/顺序步数 | 另有混训曝光 |
|---|---:|---|
| adjust_bottle | **1200**＝R1 500＋R2 500＋R3 200 | 无 |
| place_container_plate | **500**＝R5 200＋R6 100＋R12 100＋R13 100 | R4约133.3等效步（期望），另有轮换复习 |
| place_empty_cup | **300**＝R6 200＋R7已100 | R4约133.3等效步（期望） |
| move_can_pot | **400**＝R6 200＋R7 200 | R4约133.3等效步（期望） |
| turn_switch | **300**＝R8 200＋R9 100 | 无 |
| move_pillbottle_pad | **400**＝R8 200＋R9 200 | 无 |
| move_stapler_pad | **500**＝R8 200＋R9 200＋R13 100 | 无 |
| move_playingcard_away | **300**＝R10 200＋R11 100 | R11另有复习曝光 |
| place_phone_stand | **200**＝R10 200 | 无 |
| place_object_scale | **200**＝R10 200 | 无 |
| rotate_qrcode | **300**＝R10 200＋R11 100 | R11另有复习曝光 |
| pick_dual_bottles | **400**＝R11 200＋R12 100＋R13 100 | 无 |
| pick_diverse_bottles | **300**＝R11 200＋R12 100 | 无 |
| grab_roller | **200**＝R11 200 | 无 |
| lift_pot | **200**＝R11 200 | 无 |
| stamp_seal | **200**＝R11 200 | 无 |
| shake_bottle_horizontally | **150**＝R12 150 | 另有轮换复习 |
| put_bottles_dustbin | **150**＝R12 150 | 无 |
| dump_bin_bigbin | **150**＝R12 150 | 无 |
| stack_bowls_two | **250**＝R12 150＋R13 100 | R13另有复习曝光 |
| stack_bowls_three | **150**＝R12 150 | 无 |
| blocks_ranking_size | **250**＝R12 150＋R13 100 | 无 |
| stack_blocks_two | **150**＝R12 150 | 无 |
| blocks_ranking_rgb | **150**＝R13 150 | 无 |
| place_bread_skillet | **150**＝R13 150 | 无 |
| place_can_basket | **150**＝R13 150 | 无 |
| place_cans_plasticbox | **150**＝R13 150 | 无 |
| place_bread_basket | **150**＝R13 150 | 无 |

GBS不同，步数并不直接等价：R1/R2为28，其他为56。需要比较样本曝光时应再乘以各自GBS。
后续新训练/补课在对应链的表格右侧追加列，保留来源分叉；补课完成后以真实step与闭环报告更新，不能只凭计划累加。

R14完成取证补充：最新链实际主任务新增place_can_basket100、place_mouse_pad200、pick_dual_bottles300、stack_bowls_two300、handover_block150、place_shoe150、place_fan150。R14-step500追加评测：pick_dual_bottles0/3，stack_bowls_two0/3；500时课程只执行到pick_dual_bottles中段，尚未进入本轮双碗主课。R15计划handover_block200→stack_bowls_two200且无复习，尚不计入已完成累计。

### R14最终评测与R15状态

|任务|R14新增主课|R14最终评测|R15主课计划（未计完成）|
|---|---:|---|---:|
|place_mouse_pad|200|0/3|0|
|pick_dual_bottles|300|0/3|0|
|place_can_basket|100|0/3|0|
|handover_block|150|0/3|200|
|stack_bowls_two|300|0/3|200|
|place_shoe|150|2/3|0|
|place_fan|150|0/3|0|

### R15完成列（继承R14-step1350，无复习）

|任务|R14主训累计|R15新增 / 当前主训累计|R15最终评测|
|---|---:|---|---|
|handover_block|150|+200 / 350|0/3|
|stack_bowls_two|550|+200 / 750|0/3|

400步已完成，不包含隔离驻留接口测试的1步更新；隔离测试不保存模型、不并入继承链。

R16已启动但未完成：beat_block_hammer、handover_mic、hanging_mug、place_a2b_left、place_burger_fries、place_dual_shoes、place_object_stand、put_object_cabinet、scan_object、stack_blocks_three均计划新增80步，尚不计入已完成累计。10%单任务复习每5步轮换，使用through_R15的20个合格旧任务。

### R16完成列（继承R15-step400）

以下十任务首次专门主训，均新增80、主训累计80，R16共800步完成；原R15之前的主训累计保持不变。

|任务|R16新增 / 主训累计|R16最终评测|
|---|---|---|
|place_a2b_left|+80 / 80|1/3|
|beat_block_hammer|+80 / 80|1/3|
|hanging_mug|+80 / 80|1/3|
|place_burger_fries|+80 / 80|0/3|
|handover_mic|+80 / 80|0/3|
|place_object_stand|+80 / 80|2/3|
|scan_object|+80 / 80|0/3|
|place_dual_shoes|+80 / 80|0/3|
|put_object_cabinet|+80 / 80|0/3|
|stack_blocks_three|+80 / 80|0/3|

实际复习4480/44800样本（10%），每次一个旧任务、每5步切换，20任务各224样本。新增复习资格4任务：beat_block_hammer、hanging_mug、place_a2b_left、place_object_stand；历史资格池累计24（不覆盖R16冻结20任务池）。旧三任务遗忘抽检结果尚待完成，不能填为失败或成功。

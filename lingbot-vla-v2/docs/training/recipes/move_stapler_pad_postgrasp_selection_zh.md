# move_stapler_pad：抓取后准备放置到轨迹末尾

2026-10-23用户授权先筛选，不启动订书机训练；后续该任务训练结束按单任务三回合并行、全部录像准备。当前夹碗0818轮仍按原无视频计划结束，100步完成、1/3。

## 输入与定位依据

50条成功示范，episode150–199，7731帧、15FPS。只读关节目标/夹爪命令，以已验证ALOHA FK和TCP偏置得到末端轨迹；没有物体/接触/实时pad坐标。专家脚本先单手抓取、抬高0.1m，再以align约束到pad前置0.1m姿态、放下、松爪。高差以示范松爪时TCP中位位置为参考，不是物体底面到pad的实测距离。

每条要求唯一闭爪→松爪配对，关爪后末端抬升≥8cm；随后寻找距离示范释放TCP8–12cm、高差≥6cm、夹爪保持闭合且速度≤2.5cm/s至少2帧的最后稳定准备姿态。保留其前0.33秒上下文（不早于确认抬升），直到轨迹末帧；验证下放≥5cm、前进比例≥70%、路径横偏≤2.5cm、松爪漂移≤1.5cm，末尾不得重新闭爪。

## 筛选结果

|候选范围|片段|帧数|占比|
|---|---:|---:|---:|
|准备放置前短瞄准→下放停稳→松爪→末尾|50（左22/右28）|2282|29.52%|
|较宽参考：确认抬起8cm→搬运对准→末尾|50|3595|46.50%|

默认窄候选每段43–53帧，全部连续保留、没有抽稀。准备姿态实际距离释放TCP约9.84–10.02cm，竖直高差9.01–10.02cm。数值校验通过，但“抓稳/到pad上”只属于成功示范和运动学代理，不是接触或物体位姿独立验证；未视频抽检。

## 当前50帧窗口风险

2282个候选锚点中2278个的50帧动作窗口会超过轨迹末尾；按从当前帧起算，平均只有23.35帧真实轨迹长度。transform保留action_is_pad字段，但当前V2 FM损失只使用joint_mask，未见action_is_pad进入该损失的计算。因此不能把这套候选直接描述成“只监督真实放置动作”；若实际末尾填充参与loss，重复终态可能被放大。用户已确认任务完成后的填充可接受；本轮不修改模型窗口或padding mask。

## 文件与复现

`refs/data_selection/move_stapler_pad_postgrasp_place_v1/` 包含summary.json、events.json、selected_frames.json、segments.csv、broad_reference_frames.json。

入口 `tools/select_stapler_place_sequence.py`，配置 `configs/data_selection/move_stapler_pad_postgrasp_place_v1.json`；用 `--task move_stapler_pad --dataset-root <root> --settings configs/data_selection/move_stapler_pad_postgrasp_place_v1.json --output-dir <new-dir>`。缓存模式改为 `--frames-json <path> --fps 15`。只用NumPy/PyArrow，不导入Torch或使用GPU。

## 2026-10-23 宽片段训练启动

用户选用宽片段：确认抬起≥8cm后的搬运→轨迹结束，50段3595帧（46.50%），连续保留、不抽稀。命令增加 `--selection-scope postgrasp`；缺省仍为窄片段 `preplace`，二者分别生成目标文件，不覆盖原窄候选。主任务全7731帧均匀采样，只有目标3595帧loss×2，按样本数归一化，未增加目标采样倍率。保留10%复习，每5步切换一个复习任务；夹碗复习仍按固定7436帧×2。

来源 `20261023_0818_bowls_add100_from_original_s100_bf16_m56_original_grasp2_replay10pct_parallel3_novideo/checkpoints/global_step_100/hf_ckpt`。本轮200步，用户最新明确指定仅保存最终200；GBS56/GAS1/BF16。配方 `configs/training_recipes/move_stapler_pad_postgrasp200_parallel3_video.json`。run `20261023_0847_stapler200_from_bowls0818_s100_bf16_m56_postgrasp2_replay10pct_parallel3_video`。结束后驻留模型三回合并行评测，三个回合都录像，种子100000/100001/100002；不关机。

当前仅已生成配方、完成选择/配方单测并启动CPU整链预检；GPU及正式训练完成、闭环结果仍待核验，不计为已训练完成。最终结果写训练流水、任务统计和存档清单。

- 启动检查更新：真实CPU1300条/258357帧通过，首批21/56命中2倍；50段连续坐标恰3595且覆盖到末帧。GPU三步通过，峰值allocated77.71GiB/reserved85.72GiB。流水线进入正式200步训练，尚未核验完成或闭环成功率。证据 `refs/data_selection/move_stapler_pad_broad200_started.json` 与 `move_stapler_pad_broad200_coordinate_contract.json`。

## 2026-10-23 保留宽目标并增加白色订书机抓取（已筛选，未训练）

以用户最后指令为准：继续使用3595帧宽目标，不切换2282帧窄目标。增加用户图片所指的白色机身、银灰色上盖型号 `048_stapler/base3` 的抓取段。早先“橙色按钮”来自语言描述，不作为识别标准；已对照七型号真实示范首帧，base3外观对应用户图片。不要按任务指令里的white关键词筛选，蓝色base2也有white metallic components描述；场景 `{B}` 是垫子颜色，也不能用于订书机颜色判定。

型号来自原始scene_info.json的 `{A}`，白色对应原始episode9/12/17/19/23/29/31/33/34，训练episode159/162/167/169/173/179/181/183/184。九条的原始HDF5 `state/joint_states` 与训练缓存逐帧完全一致，避免仅凭episode偏移假设映射。左手2段、右手7段。

抓取判据复用现有grasp_sequence：夹爪朝下≤30°，张开≥0.95/闭合≤0.2；预抓6–16cm稳定对准、下降≥5cm、路径横偏≤2.5cm、闭爪位置漂移≤1.5cm，闭爪后3秒内抬升≥3cm。向前最多4秒寻找瞄准，进展≥2cm，反向比例≤30%。九条均从frame0的接近瞄准开始，经下降、闭爪，提起至原宽片段确认抬高8cm的起点；没有抽稀，和宽片段连续衔接。运动学与成功示范是代理，未独立测接触/物体是否随手移动。

|目标|片段|帧数|
|---|---:|---:|
|原抬起后搬运到结束|50|3595|
|新增白色瞄准→下降→闭爪→提起|9|772|
|重复边界去重|9|−9|
|联合目标|—|4358（56.37%）|

目标统一loss2、按原样本数归约；重复边界只计2，不叠成4。其余3373帧loss1；未增加采样倍率，未改变训练配置或启动新训练。原3595坐标全部保留。未来可显式设置 `LINGBOT_SAMPLE_WEIGHT_CONFIG` 为联合weight_config.json；自动继承只在真正启动并冻结新run时发生。

入口 `tools/select_stapler_white_grasp.py`，输入 `--frames-json`、经映射核验的 `--catalog`、原宽选择 `--broad-dir`、`--training-dataset-root` 和新 `--output-dir`。产物 `refs/data_selection/move_stapler_pad_broad_white_grasp_v1/` 含summary/events/white_grasp_frames/selected_frames/weight_config、型号目录、真实示范外观对照和CPU核验。服务器同名selection目录已生成；原数据只读。57项选择与损失测试＋72项配方测试共129项通过，真实全部目标2/全部非目标1及九条HDF5身份检查通过。

### 2026-10-23 09:28 宽目标加白色抓取续训200（已启动）

run `20261023_0928_stapler_add200_from_0847_s200_bf16_m56_broad_whitegrasp2_replay10pct_parallel3_video`，来源 `20261023_0847_stapler200_from_bowls0818_s100_bf16_m56_postgrasp2_replay10pct_parallel3_video/checkpoints/global_step_200/hf_ckpt`。用户要求先启动再整理文档。目标4358/7731帧（原3595全保留，白色772中新增763、交界9去重），统一loss2、按样本数归约，主任务均匀、不加概率倍率；其余样本1。10%单任务轮换复习、每5步切换，夹碗固定原7436帧loss2。BF16/micro56/GAS1，200步，仅最终200存档，驻留三回合并行且全录像，无补课、不关机。

服务器129项CPU回归通过；真实1300轨迹258357帧整链通过，首批28/56样本权重2；GPU三步通过（allocated77.70GiB/reserved83.56GiB）。正式已出现更新日志，完成步数及闭环待终态核验，不能记为200步完成。启动前空闲约54GiB，单份存档预算16.06GiB，无清理。配方 `configs/training_recipes/move_stapler_pad_add200_from_0847_white_grasp2_parallel3_video.json`；证据 `refs/data_selection/move_stapler_pad_white200_started.json`。

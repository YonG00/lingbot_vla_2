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

2282个候选锚点中2278个的50帧动作窗口会超过轨迹末尾；按从当前帧起算，平均只有23.35帧真实轨迹长度。transform保留action_is_pad字段，但当前V2 FM损失只使用joint_mask，未见action_is_pad进入该损失的计算。因此不能把这套候选直接描述成“只监督真实放置动作”；若实际末尾填充参与loss，重复终态可能被放大。当前没有修改模型窗口、padding mask或loss，也没有启动订书机训练。

## 文件与复现

`refs/data_selection/move_stapler_pad_postgrasp_place_v1/` 包含summary.json、events.json、selected_frames.json、segments.csv、broad_reference_frames.json。

入口 `tools/select_stapler_place_sequence.py`，配置 `configs/data_selection/move_stapler_pad_postgrasp_place_v1.json`；用 `--task move_stapler_pad --dataset-root <root> --settings configs/data_selection/move_stapler_pad_postgrasp_place_v1.json --output-dir <new-dir>`。缓存模式改为 `--frames-json <path> --fps 15`。只用NumPy/PyArrow，不导入Torch或使用GPU。

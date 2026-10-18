# 低噪 FM 与目标变化窗口采样

这是两个独立实验开关。默认仍用原 FM 时间分布和 hardness 扫描；连续抓取窗口的 CPU 规则有回归测试，新的真实数据覆盖率和 GPU 效果仍待本轮验证。

## 含义与边界

RoboTwin 的 `get_obs()` 使用 `get_*_arm_jointState()`，保存的是 drive target；夹爪值也是控制指令。
现有训练数据不能证明实际受力或关节跟踪误差。本方法只识别**控制目标变化的开始与停止**，不能称为接触检测。
腕部运动可能只是自由空间移动；抓住后保持目标不变也不会产生持续接触信号。报告中事件多寡不等于识别正确。

代码按 `robot.py` 的 `arm_joints_name[0/1]` 顺序以及 `robotwin.yaml` 的 `6+1+6+1` 原始状态布局解析名称。
必须提供真实 `assets/embodiments/aloha-agilex/config.yml`，并填写其中已核对的腕部名称；不会猜最后两维就是腕部。
本地仓库缺该资产，因此还需要在服务器核对 URDF/运行时关节含义，再填名称。数字式名称本身不能证明该关节属于腕部。

## 参数

以下均为 AL JSON 字段（进程启动时读取，更改后须重启）：

| 字段 | 默认 | 作用 |
|---|---|---|
| `low_noise_fm_enabled` | `false` | 25% 时间样本改取均匀 `[0.001,0.2]`，其余原样；固定 time 的 hardness 不受影响 |
| `hardness_scan_enabled` | `true` | 历史名称，现在是**两种样本扫描的总开关**；false 时均匀采样 |
| `sample_scan_method` | `hardness` | `hardness` 或 `target_change_window`，互斥，不相乘 |
| `target_change_cache_dir` | 空 | 可读缓存根目录；窗口模式必填 |
| `target_change_dataset_root` | 空 | 真实训练集目录；完整/便捷启动器自动填写，通用 AL 入口须显式填写 |
| `target_change_embodiment_config` | 空 | 实际机器人 `config.yml`，窗口模式必填 |
| `target_change_wrist_joint_names` | `[]` | `[左腕名称列表, 右腕名称列表]`，必须逐名匹配 |
| `target_change_window_seconds` | `0.25` | 事件左右窗口半径，各 `ceil(秒数×FPS)` 帧 |
| `target_change_gripper_speed` | `0.1` | 原始夹爪指令每秒变化量的开启阈值 |
| `target_change_wrist_speed` | `0.2` | 原始腕部弧度每秒变化量的开启阈值 |
| `target_change_release_ratio` | `0.5` | 停止阈值为开启阈值乘该值，形成滞回 |
| `target_change_persistence_frames` | `2` | 连续达到条件的帧数，事件回标第一帧 |
| `target_change_window_weight` | `2.0` | 窗口内权重，窗口外恒为 1；最终全任务归一化 |
| `target_change_grasp_lift_max_delay_seconds` | `2.0` | 闭合结束后，腕部开始运动仍可关联为初始提起的最长等待时间 |

低噪开关不改变推理积分步数，也不改变用户显式传入的 time；关闭时不增加 RNG 调用。
`t` 的定义是 `x_t=t*noise+(1-t)*action`，因此小 t 靠近专家动作。独立验证的 GMSE/成品模型倍率仍沿用原逻辑。
窗口扫描直接读取未归一化 Parquet 数字列，不经过训练 transform、反归一化或随机失败重试的数据读取路径。

## 扫描与采样流程

1. 对每条**训练轨迹**的夹爪与指定腕部目标求一阶差分，再乘 FPS，得到每秒变化量。
2. 对有符号变化量做相邻三帧中值滤波，再取绝对值；抑制孤立跳点。两帧持续与高/低阈值滞回检测开始/停止。
3. 普通事件保留前后局部窗口，不连续加权所有自由空间腕部运动。按同侧夹爪指令方向区分闭合（数值下降）与打开（数值上升）。闭合后，在可配置等待上限内寻找同侧第一段腕部运动；找到时，将闭合前扩展半径到腕部运动停止后扩展半径设为一个连续抓取阶段窗口，覆盖闭合过程和中间保持帧。找不到时只覆盖闭合及短暂保持；下一次夹爪变化截断配对，避免连到后续放置动作。
4. 原始事件窗口与连续阶段窗口统一合并；窗口外权重 1，窗口内默认 2，重叠不累乘。复用 `normalize_weights` 和现有加权采样器，不做二元分数百分位排名。
5. 采样对象仍是**动作 chunk 的锚点帧**，不是只对 chunk 中的事件时刻算 loss。窗口不会自动向前扩展整个 chunk；当前采样器每 batch 无放回，报告概率为输入权重分布，不是实测 batch 频率。

连续抓取阶段只是依据控制命令推断的候选片段，不是实际接触或抓取成功标签。打开动作不会创建抓取阶段。静态缓存 schema 已升至 2；旧缓存必须显式 `--refresh` 重建，防止沿用旧的不连续窗口。

用前一份 stand 缓存中保存的原始信号离线重算最终规则：5561帧中3707帧进入窗口（66.7%），旧规则为2772帧（49.8%）；40个闭合阶段均在等待上限内关联到腕部运动。沿用窗口权重2.0，理论概率质量约80.0%落在加权窗口内。全运动区间连续方案曾覆盖80.6%，因过宽已撤回。该结果证明规则覆盖了缓存信号，不代替视频人工检查，也不表示40次都真实抓住了物体。新的 schema2 报告仍须启动训练前重建并抽样查看。

半径 0.25 秒：20 FPS 为 11 帧，30 FPS 为 17 帧（裁边/合并前）。这是初始参数，不是已在 cabinet 上验证的最优值。
长时间抓持不会因本方法获得全程加权；如果希望覆盖整段动作，应先核对报告再调整半径，避免窗口覆盖整条轨迹后退化成均匀。

## 训练前准备与检查

在 AL JSON 填好上述路径与真实腕部名称后，先用 CPU 单独准备：

```bash
python experiment/robotwin/prepare_target_change_windows.py \
  --al-config /path/to/al_config.json \
  --dataset-root /path/to/dataset \
  --manifest refs/al_artifacts/al/task_splits_50/manifest.json \
  --tasks put_object_cabinet
```

输出 `<cache-dir>/target_change_window/put_object_cabinet.json` 和同名 `.html`。
JSON 保存明文来源、关节映射、参数、训练回合、绝对帧号、事件、合并窗口、权重与全任务概率；**不按模型或文件指纹生成路径**。
HTML 显示原始目标曲线、滤波后的变化量、高低阈值与事件按钮；LeRobot v2 独立轨迹视频存在时可点击事件同步定位多个相机。
浏览器须能访问原始视频路径；把报告拷到其他机器时要连视频一起搬并修正链接。
v3 分块视频的起止偏移尚未适配，报告保留数字曲线，不猜偏移；没有视频的报告不能替代人工视觉验收。

先看每条轨迹的抓物体、抓柜门/抽屉把手动作是否落在窗口内，再检查误检与漏检；不能用预期的“至少几次抓取”硬凑事件数。
当前任务究竟操作柜门还是抽屉应以原始轨迹为准。重点看抓取前后、释放、腕部自由移动是否被误标，记录帧号后再决定阈值。

缓存已有时只读；缺任务才补算。来源目录、参数、FPS、关节顺序或训练回合变化会明确报错，检查后用同一准备命令加 `--refresh` 重建。
运行时通过 `(episode, absolute index)` 精确映射当前 local_idx，并核对完整帧集合；不会继承其他 run 的 local_idx 概率。
同一路径下原始数值被覆盖而坐标未变，不会被自动发现：数据更换后必须人工 `--refresh`。
既有数据任务归属对拍仍使用原 `data_audit_file`，本工具不替代那项检查。

完整与便捷启动器在 `--execute` 时，会在 torchrun 前准备缺失缓存；默认预览不读轨迹、不写缓存。改动窗口算法后应先显式重建缓存：

```bash
python experiment/robotwin/prepare_target_change_windows.py \
  --al-config /path/to/al_config.json \
  --dataset-root /path/to/dataset \
  --manifest refs/al_artifacts/al/task_splits_50/manifest.json \
  --tasks place_object_stand --refresh
```
通用 AL 入口只读已有缓存，不在模型加载后补扫描。训练中的任务选择、DEFER/rescue、周期 HF 存档均复用静态概率；hardness 模式保持原模型扫描触发机制。

## 便捷续训

在已经填写窗口配置的 AL JSON 基础上：

```bash
python experiment/robotwin/continue_skill_training.py \
  --from-hf /path/to/run/checkpoints/global_step_1000/hf_ckpt \
  --steps 500 --micro 24 --gas 1 \
  --al-config /path/to/al_config.json \
  --sample-scan --sample-scan-method target_change_window \
  --low-noise-fm --no-shutdown
```

默认仅预览，加 `--execute` 才启动。先做单变量对照：可用 `--no-low-noise-fm` 只验证窗口采样，或切回 `--sample-scan-method hardness` 只测试低噪。
关闭所有扫描用 `--no-sample-scan`。其他参数可继续通过 `--al-set key=JSON值` 覆盖。
自动关机仍由原 `--shutdown/--no-shutdown` 控制，本改动不启动训练、不清理存档、不改保留机制。

## 验证边界

本地隔离 Python 3.10 环境完成以下 CPU 验证（2026-10-16）：

| 组合/链路 | 已验证行为 |
|---|---|
| 扫描关闭，方法选静态 | 不需要缓存，不调用模型 scorer，均匀采样 |
| 扫描开启，方法为 hardness | 既有 hardness、周期保存后重扫、采样/恢复回归通过 |
| 扫描开启，方法为静态 | Parquet 仅扫描训练回合、缓存复用不读数字行、绝对帧映射、构造时预载、Replay/旧池概率替换 |
| 静态模式进入存档、rescue、DEFER、恢复 | 不调用模型 scorer，继续使用静态概率，保存后的扫描 ID 不丢失 |
| 低噪关闭/开启 | 关闭保持原 sample_time 逐位结果与 RNG；开启比例、范围、复现通过，V2 继承该方法 |
| 启动器 | 开关传递、后台配置快照、CPU 准备先于启动、坏缓存不遗留半成品 run |

最终回归 **169 passed, 66 subtests passed, 2 deselected**：

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/test_target_change_window.py tests/test_low_noise_fm.py \
  tests/test_skill_mix_task_launcher.py tests/test_continue_skill_training.py \
  tests/test_al_skill_experiment.py tests/test_al_save_hardness_refresh.py \
  tests/test_hardness.py tests/test_persistence.py tests/test_auto_learning_sampler.py \
  tests/test_replay.py tests/test_al_pool_state.py tests/test_hardness_cache_cpu.py \
  tests/test_al_periodic_checkpoint.py tests/test_al_hf_only.py \
  -k 'not test_scorer_skips_cached_samples_end_to_end and not test_scorer_without_cache_keeps_old_behavior'
```

这两项在首次扩展回归中因隔离环境缺 `torchdata` 失败，未修改测试或重依赖导入链；最终选择集中显式排除，并未当作通过。`test_hardness_scan_timing.py` 同样因该缺失依赖未能收集。真实模型 scorer、完整训练依赖链需服务器继续验证。
19 个新增/修改 Python 文件编译通过；两个启动工具 `--help` 与 `git diff --check` 通过。

真实 `put_object_cabinet` 轨迹、腕部名称的物理含义、视频帧对齐、GPU 训练以及闭环收益，均需后续上机验证；不能据 CPU 测试声称效果提升。

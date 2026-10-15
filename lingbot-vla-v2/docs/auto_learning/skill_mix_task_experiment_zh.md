# 指定技能学习实验

实验分支 `learn_skill_mix_task` 基于 `Task_Auto_Learning` 的 `956782f`。已完成本地实现与 CPU 契约验证；2026-10-15 获批提交、push，并在服务器独立目录同步和做 CPU 验证。GPU 验收和训练需后续单独确认，不能由同步脚本自动启动。

## 当前实验

从 CKPT9680 的完整 BF16 HF 权重起新 run，专练 `put_object_cabinet`。模型继承权重，优化器、LR 调度和实验步数重新开始。学习任务的 train 划分全部参与采样，独立 val 保留。归档 manifest 中该任务为 **40 train / 10 val**；每次开环仍评 train_monitor 4 条、独立 active_val 4 条，不把验证数据加回训练。

| 参数/行为 | 本实验 | 旧默认 |
|---|---|---|
| `task_names` | 单任务总范围 | manifest 全体 |
| `learning_task_names` | `[put_object_cabinet]` | 动态候选；显式 Replay 任务除外 |
| `replay_task_names` | `[]` | `null`：使用动态 PASS 池 |
| `new_ratio` | `1.0`，全为当前任务 | 旧固定槽位或用户配置 |
| `pass_transition_enabled` | `false` | `true` |
| `task_scan_enabled` | `false`，不做 bootstrap/rescan/rollover 复扫 | `true` |
| `hardness_scan_enabled` | `false`，均匀采样 | `true` |
| `import_pool_state` | `false`，不继承旧状态/attempt/权重/RNG | `true` |
| `review_after_task_transitions` | `0`，关闭复查 | `2` |
| `eval_interval_steps` | `100` optimizer steps | `50` |
| `min_steps_before_defer` | `10000`，大于本次总步数 | `100` |
| `early_defer_on_overfit` | `false` | `true` |
| `defer_resample_retry` | `false` | `false` |
| 本次新增步数 | `1000`，可改 `--steps 2000` | 各配置决定 |
| `reset_step_counter` | `true`，从 0 开始 | `false`，0 offset 从路径推导 |
| `dcp_save_mode` | `hf_only` | `always` |
| HF | BF16，每 200 步及结束时 | 各配置决定 |

配置：`configs/auto_learning/learn_skill_mix_task.json`。开关均在启动时读取，修改后需重新启动。关闭 PASS 是不允许**新的自动状态转换**，不会把继承的旧 PASS 改回候选；本实验因此同时关闭旧池导入。

`min_steps_before_defer` 是最小判定预算，不是必然的硬截止；达到预算后仍根据 LP 决定是否 DEFER。关闭 hardness 后，即使以后启用 rescue 或 CONTINUE refresh，也不能调用 scorer 或使用旧概率：rescue 只增加预算、保持均匀采样。实际 DEFER 后的重选也遵守同一个扫描开关。任务级 rescan 与样本 hardness 是两个独立开关。

## 两把尺子，只观察

1. `val_to_reference`：当前 active_val gMSE / 成品模型同一组验证轨迹 gMSE。
2. `val_to_initial`：当前 active_val gMSE / 本 run 第一次训练前 active_val gMSE。

第一把使用 `pass_thresholds_gmean1_active4.json`：由归档 `refs/al_artifacts/eval_results/open_loop/ref50k/ref_per_traj.jsonl` **按每任务 val_ids 的前 4 条重新几何聚合**，倍率 1，不从裁剪后的旧 60× 表反除。`put_object_cabinet` 轨迹为 `[2350, 2351, 2354, 2367]`，参考值 `0.0007115229657801033`。文件记录源 SHA256 和轨迹列表；更换划分/验证条数时，启动器拒绝不一致的参考轨迹。

参考模型结果来自历史归档；相同轨迹不代表精度、归一化、采样窗口、源码均已在新机器上对齐。上机需要核实这些口径，必要时重新测成品基准。第二把在本 run 相同环境实测。两者进入事件文件中的 metric 和 registry 的 `eval_history`，并写 TensorBoard 的 `task/<任务>/val_to_reference`、`val_to_initial`。零或缺失分母显示空值，不制造比例。

原 `val_to_pass_threshold` 指标继续保留作为兼容显示；PASS 关闭时达标也不会毕业。开环指标用于观察学习趋势，不能证明闭环已学会。训练 loss 为 Flow Matching 目标，不等同于开环动作 MSE。

## 数据对拍

启动器生成精确的 train episode 白名单，避免从 50 个任务里误取。真实数据集建好后，在模型训练之前：

- 首次对拍任务分块顺序、train/val 不重叠、实际 episode 集合、每条训练轨迹帧数、绝对 frame 边界（metadata 提供时）、每个 local_idx 的任务/episode，以及帧表 `task_index` 与 episode 原始 instruction。
- 单柜子任务额外核对 instruction 中含 `cabinet`。完整报告保留轨迹/样本数和原始 instruction 示例，便于人工核对。其他新任务可在 `data_audit_instruction_patterns` 增加表达式。
- 后续只核对映射列摘要、split/任务顺序/检查规则，以及 metadata/帧表文件的路径、大小、mtime；身份相同复用报告。任一相关指纹变化重新完整检查。

`sample_id` 始终是**当前过滤后数据集 local_idx**。报告是元数据与索引契约验收，不会解码视频、判断画面是不是柜子；同尺寸且保留 mtime 的人为内容替换不在这套快速文件身份检查保证范围内。报告应放在 run 目录外、按数据版本共用，避免每个新 run 重做。

## 启动器

`experiment/robotwin/launch_skill_mix_task.py` 默认仅打印计划，不写输出、不启动训练。它检查 HF 分片头、索引与 payload 是否完整，BF16 dtype，来源归一化文件，HF 与选用 Qwen3VL 的 tokenizer 是否一致，并记录来源配置/索引/归一化/参考表摘要。它不替代真实模型的加载验收。

**`--train-config` 必须给 CKPT9680 来源 run 的 `lingbotvla_cli.yaml`**，保留原动作配置和归一化。脚本生成独立 `trainer_config.json`，清除 DCP 恢复及 VLM-only/部分加载入口；完整 HF 使用 `post_training=true` 严格检查参数。模型构造仍复用原有 loader，实际加载、buffer 和 GPU 精度需上机检查。Qwen3VL 路径需保留型号名称，因为现有模型 patch 选择依赖这个名字；不把 step 目录当作 Qwen 型号路径。

开机后先核实真实路径，打印计划示例（下列占位参数需替换）：

```bash
cd <仓库Python包目录>
python experiment/robotwin/launch_skill_mix_task.py \
  --model /data/outputs/al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt \
  --train-config <来源run>/lingbotvla_cli.yaml \
  --dataset-root <真实RoboTwin_lerobot_v30根目录> \
  --qwen3vl <真实Qwen3-VL目录> \
  --out <新实验输出目录> \
  --audit-report <可复用的对拍报告目录>/cabinet.json
```

默认 micro=1、GAS=10、单卡，GBS=10；这是待显卡验收的保守初值，不宣称已跑通或最快。`--steps 2000` 可延长实验，仍须低于 defer 预算。新增目标学习与复习任务可传 `--learning-tasks ... --replay-tasks ... --new-ratio 0.7`；任务名、重复和集合重叠会校验。显式 Replay 仅表示用户分配复习数据，不会写成“模型已 PASS”；已有生产自动选任务启动器保持原语义，并拒绝免任务扫描配置，避免误起预扫。

`--prepare-only` 只落配置和来源报告，不训练。`--execute` 创建新 run 后开始训练；输出目录已存在就拒绝复用。两者互斥，准备过的目录不能再直接 `--execute`，应重新选择目录。长任务仍须在服务器后台启动并重定向日志；此阶段尚未授权远端启动。

## 存档与恢复

1000 步预期只得到 HF 200/400/600/800/1000，结束与周期同一步只写一次；非 200 整数倍结束则补最终 HF。STOP_AND_SAVE 沿用已有 AL unit 边界停止机制，然后走 HF-only 收尾。最后一轮自然结束也会补 HF。

本模式无 DCP、无优化器/调度器状态，**不支持精确 Resume**。HF 可用于测评或作为下一次新实验的起点。保存失败会报错停止，不能用“没有 DCP 的 best-effort 成功”假装存好了；磁盘保护仍在写入前检查。保留所有 HF，6B BF16 五份权重按参数规模约 60 GB，真实占用/主机内存峰值需上机确认，不自动删除旧产物。

| 存档模式 × save_hf_weights | 周期 | 收尾/轮末 | 本地证据 |
|---|---|---|---|
| `always × false/true` | DCP，仍守 unit 边界 | 原路径 | 原周期 9 项回归覆盖；真实导出未重测 |
| `final_only × false` | 原无 HF 行为 | 原完整 DCP 路径 | 原周期分支 CPU 回归；真实收尾未重测 |
| `final_only × true` | 原 HF 直出 | 原完整 DCP 路径 | 原周期分支 CPU 回归；真实收尾未重测 |
| `hf_only × true` | HF 直出，不等 unit | HF，同步成功游标去重 | 周期/正常结束/AL结束/STOP复用路径/轮末/失败契约 |
| `hf_only × false` | 拒绝 | 拒绝 | 真实参数守卫 CPU 执行 |

## 本地验证与后续

34 项纯 CPU 单测通过，包括扫描 × rescue 四组合、1000 步完整模拟、指定 Replay、真实 local_idx 数据对拍/缓存失效、初始评测失效不能冒充训练前基准、HF-only 分支和启动器预检。另将当前调度器与基线 `956782f` 的调度器在 6 组旧配置上对拍，学习路线、任务状态/步数、停止原因和采样表一致。

```bash
PYTHONPATH=.:tests python -m unittest \
  test_al_skill_experiment test_al_data_audit test_al_hf_only \
  test_al_periodic_checkpoint test_skill_mix_task_launcher -v
```

本机 Python 3.13 的 pytest 在收集前退出 139，禁用插件仍复现，未把它计作测试通过；模拟测试替换 CUDA 计时同步以避免导入本机 torch。CPU 证据不包含真实张量导出、NCCL 集合通信或训练效果。

本轮服务器同步仅在独立实验目录进行，不覆盖生产分支。后续验收顺序：核实来源模型/归一化与 tokenizer → 真实 CPU 数据对拍和小模型保存契约 → 用户开卡并确认后，做 GPU 权重完整加载、独立验证与小步数 HF-only 验收 → 确认每份存档完整且没有 DCP → 再启动 1000 步实验。训练完后另行闭环比较 5 个 HF，固定 unseen、任务、seed 和推理参数；本轮不接闭环通关。

2026-10-15 同步前的机器检查：无训练进程，无 GPU，CPU 模式内存限额 2 GiB；文件系统可用约 18 GB，五份 6B BF16 HF 约需 60 GB，尚不满足完整实验存档容量要求。不自动清理历史产物；正式训练前须重新确认磁盘挂载与可用空间。生产分支 `e64388c` 上的既有 CPU 用例 `test_hf_export_dtype.py`、`test_state_machine.py`、`test_pass_thresholds.py` 共 49 项通过；这不是本次新代码的上机验收。

# 指定技能学习实验

实验分支 `learn_skill_mix_task` 基于 `Task_Auto_Learning` 的 `956782f`。主体实现已提交并推送为 `6470669`，同步至服务器独立目录 `/data/experiments/learn_skill_mix_task`；生产 `/data/code` 保持 `Task_Auto_Learning` 的 `e64388c`。GPU 验收和训练需后续单独确认，不能由同步脚本自动启动。

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
| `hardness_scan_enabled` | `true`：任务启动 + 后续仍训练的周期成功存档后各扫一次 | `true`；入口扫描，另有可选 continue/rescue 重扫 |
| `import_pool_state` | `false`，不继承旧状态/attempt/权重/RNG | `true` |
| `review_after_task_transitions` | `0`，关闭复查 | `2` |
| `eval_interval_steps` | `100` optimizer steps | `50` |
| `min_steps_before_defer` | `10000`，大于本次总步数 | `100` |
| `early_defer_on_overfit` | `false` | `true` |
| `defer_resample_retry` | `false` | `false` |
| 本次新增步数 | `1000`，可改 `--steps 2000` | 各配置决定 |
| `reset_step_counter` | `true`，从 0 开始 | `false`，0 offset 从路径推导 |
| `dcp_save_mode` | `hf_only` | `always` |
| HF | BF16，每 250 步及结束时 | 各配置决定 |

配置：`configs/auto_learning/learn_skill_mix_task.json`。开关均在启动时读取，修改后需重新启动。关闭 PASS 是不允许**新的自动状态转换**，不会把继承的旧 PASS 改回候选；本实验因此同时关闭旧池导入。

### 样本扫描的两个触发机制（2026-10-16 用户要求）

`hardness_scan_enabled=true` 且 `sample_scan_method=hardness`（默认）时，样本难度扫描有**两个触发点**：

1. **任务启动**：`Scheduler._select()` 里扫一次（原有）。
2. **周期存档后**：周期保存成功后调用 `AutoLearnLoopHook.on_save()` → `Scheduler.refresh_current_hardness()`，更新单任务长跑的采样分布。HF-only 在 `save_steps` 整数倍保存；旧 DCP 模式仍可能延迟到 unit 边界，按实际落盘时点触发。

生效条件：`hardness_scan_enabled=true`、当前任务处于 `TRAINING`、确有成功存档且后续仍会训练。保存失败、只建空目录、最大步数到顶、AL 预算耗尽、STOP_AND_SAVE/待停标记、最后一轮的最后一步均不重扫。HF-only 保存失败仍按原规则报错退出；best-effort HF 导出失败则继续训练但跳过本次扫描。多 rank 下由 rank0 广播 HF 导出成功状态，避免各 rank 走不同扫描分支。

**新 probs 在下一个 learning unit 开始时生效**：`TrainRequest` 在 unit 开始时已快照 `rec.sample_probs`（`_prepare_train_request`），采样器也只在边界 `set_request()` 时重编译 alias 表。当前 unit 的分布和 DataLoader 预取不变。`eval_interval_steps=100` 时，250/500/750 存档后立即重扫，分别从第 **301/501/801 次 optimizer 更新**使用新概率；1000 步达到结束条件则仅保存，不再白扫。任务入口一次 + 250/500/750 三次，共四次扫描。

缓存与覆盖范围：

- 存档后重扫调用真实 scorer 的 `score_fresh()`，暂时绕过 hardness 缓存读写，按**当前权重**计算；异常时也恢复缓存对象，不用新权重的分数污染来源模型缓存。实验启动器另外将子进程的 `AL_HARDNESS_CACHE_FILE` 和旧别名 `AL_HARDNESS_CACHE` 都置空，防止继承 shell 中的旧路径。其他入口的缓存策略保持原有配置。
- 探测窗口按 `hardness_version` 轮转；默认 `hardness_probe_fraction=0.10` 每次扫 4/40 条训练轨迹，每条最多 300 帧。未扫到的样本按默认难度计权，概率覆盖全训练集，但只反映**当次窗口**，不会累计历史窗口分数；验证集不参与打分或训练。
- `hardness_refresh_save` 事件的 `step` 是包含 unit 内进度的 AL 相对步，`scheduler_step` 是已完成 unit 的记账步，`checkpoint_step` 是实际存档的训练绝对步。例：250 步重扫记录 `step=250, scheduler_step=200, checkpoint_step=250`；有来源 offset 时绝对步可能不同。Scheduler 本身的记账不因记录日志而前移。

样本扫描是 Flow Matching loss 的难度代理；更新采样概率不等于保证训练 loss、开环 gMSE 或闭环成功率提升，需要对照验证。

选择 `sample_scan_method=target_change_window` 时，只在训练前准备静态窗口，任务启动、周期存档和 DEFER/rescue 均复用缓存，不执行上述模型重扫。低噪 FM 是独立开关。参数、按任务名的缓存和人工对帧报告见 [目标变化窗口说明](target_change_window_zh.md)。

启动器门控相应放宽：原 `build_plan` 强制 `hardness_scan_enabled=false`，现**允许为 true**；`task_scan_enabled` / `pass_transition_enabled` / `import_pool_state` / `review_after_task_transitions` / `rescan_candidates_after_transition` 仍必须关闭。

本地验证（2026-10-16，未同步、未做 GPU 验收）：39 项标准库 CPU 测试通过，包含真实 Hook/采样器的 1000 步模拟、实际 scorer 类的缓存命中/绕过/异常恢复，以及成功/失败/无产物/收尾存档触发、非零 rank 使用 rank0 保存结果的契约；另在已安装 `tokenizers` 的本地 Python 环境中，启动器 8 项 CPU 测试通过，覆盖两种旧缓存环境变量被清除。验证入口：

```bash
PYTHONPATH=.:tests python -m unittest \
  test_al_skill_experiment test_al_periodic_checkpoint test_al_hf_only \
  test_al_save_hardness_refresh test_skill_mix_task_launcher -v
```

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

### 从 HF 存档便捷续练

`experiment/robotwin/continue_skill_training.py` 用于从某次训练的 `global_step_N/hf_ckpt` 开始**新 run**。默认新增 500 步、micro24/GAS1、单任务沿用该实验 AL 配置；它会从来源 run 自动读取 `lingbotvla_cli.yaml`、`datasets.txt`、AL 对拍报告，并复用 `launch_skill_mix_task.py` 的 HF、tokenizer、数据划分预检。新 run 的优化器/LR/步数从头开始；HF-only 不提供精确续训。

先预览（不会占 GPU 或创建训练目录）：

```bash
cd /data/experiments/learn_skill_mix_task/lingbot-vla-v2
PY=/data/miniconda3/envs/lingbotvla/bin/python
MODEL=/data/outputs/<来源训练run>/checkpoints/global_step_1000/hf_ckpt
RUN=continue_cabinet_from1000_add500
"$PY" experiment/robotwin/continue_skill_training.py --from-hf "$MODEL" --run-name "$RUN"
```

预览确认来源、任务、AL 开关、输出目录、日志路径和剩余磁盘后，再后台启动：

```bash
"$PY" experiment/robotwin/continue_skill_training.py \
  --from-hf "$MODEL" --run-name "$RUN" --steps 500 --micro 24 --gas 1 --execute
```

默认**不关机**。想在训练成功结束且最终 step500 HF 校验通过后自动关机，增加 `--shutdown`：

```bash
"$PY" experiment/robotwin/continue_skill_training.py \
  --from-hf "$MODEL" --run-name "$RUN" --steps 500 --micro 24 --gas 1 --shutdown --execute
```

命令加 `--no-shutdown` 可显式关闭自动关机。只有训练进程返回成功，且最终 checkpoint 的 BF16 safetensors、张量索引和 tokenizer/config 文件均完整并与来源张量名一致，脚本才运行 `bash /usr/bin/shutdown`；训练失败、磁盘保护拦截、最终 HF 不完整时不关机。脚本不会删除任何 checkpoint，也不会启动“满四删一”监控。

常用 AL 参数可直接在命令行改，不需要复制 JSON。比如关闭样本扫描、把开环间隔改成 50、DEFER 预算设为 5000：

```bash
"$PY" experiment/robotwin/continue_skill_training.py \
  --from-hf "$MODEL" --run-name "$RUN" --steps 500 \
  --no-sample-scan --eval-interval-steps 50 --defer-steps 5000
```

可选布尔开关：`--sample-scan/--no-sample-scan`、`--refresh-hardness-on-continue/--no-refresh-hardness-on-continue`、`--low-noise-fm/--no-low-noise-fm`。扫描方法用 `--sample-scan-method hardness|target_change_window` 二选一；静态窗口的真实关节名称、预计算缓存和报告见 [窗口采样说明](target_change_window_zh.md)。数值选项：`--eval-interval-steps`、`--defer-steps`、`--defer-retry-steps`。任意其他 AL 字段可用重复的 `--al-set hardness_probe_fraction=0.2 --al-set hardness_weight_max=2.5` 覆盖；值按 JSON 解析，不能覆盖任务集合、对拍路径和总步数上限。单技能隔离要求关闭任务扫描、PASS 转换、旧池导入和 review；尝试开启会被预检拒绝。脚本会验证配置、显示生效值，并将最终配置快照到 launch 记录旁。

启动后日志和状态 JSON 位于 `/data/outputs/skill_launches/`；脚本启动时会打印实际路径。用 `tail -f <日志路径>` 看训练与扫描进度，用 `cat <状态JSON路径>` 检查是否完成、最终 HF 校验结果及是否发出关机命令。自定义任务可用 `--learning-tasks task_a task_b`；默认使用 AL 配置中的任务。`--out` 可指定一个不存在的新输出目录。预览和执行传相同 `--run-name` 会使用相同路径；重复启动会拒绝覆盖已有记录。

便捷入口只检查当前 NVIDIA compute 进程和输出盘余量，不会停止或清理已有任务。确认 `nvidia-smi` 无计算进程、磁盘余量足够后再加 `--execute`。自动关机是真正的服务器关机动作，使用前先确认这台机器可关。

`experiment/robotwin/launch_skill_mix_task.py` 默认仅打印计划，不写输出、不启动训练。它检查 HF 分片头、索引与文件长度是否匹配，BF16 dtype，来源归一化文件，HF 与选用 Qwen3VL 的 tokenizer 语义是否一致，并记录来源配置/索引/归一化/参考表摘要。分片预检不读取全部权重 payload，不能替代真实模型加载验收。

tokenizer 校验使用项目已依赖的 `tokenizers` 库分别加载，取消运行期 padding/truncation 后比较规范化配置，仍严格比较完整词表、ID、有序 BPE 合并规则、特殊 token、normalizer、pre-tokenizer、decoder 等。库负责兼容旧字符串与新二元数组的 merges 存储格式；不会修改 tokenizer 文件或训练长度设置。报告记录规范化 SHA256、原始长度设置及两个文件摘要。实际 CKPT9680 与 Qwen 文件的 151,387 条合并规则全部一致，规范化配置相同；原始 JSON 相等检查会因上述格式、默认 `ignore_merges=false` 和训练保存的 72 长度设置误报。该校验修正已单独获批，不是跳过兼容性检查。

**`--train-config` 必须给 CKPT9680 来源 run 的 `lingbotvla_cli.yaml`**，保留原动作配置和归一化。脚本生成独立 `trainer_config.json`，清除 DCP 恢复及 VLM-only/部分加载入口；完整 HF 使用 `post_training=true` 严格检查参数。模型构造仍复用原有 loader，实际加载、buffer 和 GPU 精度需上机检查。Qwen3VL 路径需保留型号名称，因为现有模型 patch 选择依赖这个名字；不把 step 目录当作 Qwen 型号路径。

训练子进程必须优先导入本实验源码。启动器显式把包根置于 `PYTHONPATH` 首位，再保留原值，torchrun 与 worker 均继承此环境；不能只依赖启动器的 cwd。机器中的 editable 安装仍指向生产 `/data/code`，无需修改该安装。2026-10-15 首次 GPU 启动因缺此设置，误导入旧 AL（四个新开关均不存在），在 step0 以 `no_eligible_candidate` 退出；`skill_cabinet_9680_1000` 中 step0 HF 和日志留作证据，不算训练完成。用户获知后批准修复并在新目录重启；CPU 契约使用竞争旧包与真实子进程覆盖此问题。

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

1000 步预期得到 HF 250/500/750/1000，共四份，全部保留；不启动旧的 `al_checkpoint_watch.py --apply`（满四份删最早一份）监控，启动脚本也不调用它。结束与周期同一步只写一次；非 250 整数倍结束则补最终 HF。评测仍每 100 步进行；HF-only 周期保存不等待评测单元边界，因此 250/750 步不会推迟到 300/800。STOP_AND_SAVE 沿用已有 AL unit 边界停止机制，然后走 HF-only 收尾。最后一轮自然结束也会补 HF。

本模式无 DCP、无优化器/调度器状态，**不支持精确 Resume**。HF 可用于测评或作为下一次新实验的起点。保存失败会报错停止，不能用“没有 DCP 的 best-effort 成功”假装存好了；磁盘保护仍在写入前检查。保留本次全部四份 HF；按来源每份权重 12,752,076,094 bytes 估计，四份约 47.5 GiB，另需资源文件与日志空间。真实导出占用/主机内存峰值需上卡确认。旧训练存档只留两份是一次性清理决定，不是本次训练的自动保留规则。

| 存档模式 × save_hf_weights | 周期 | 收尾/轮末 | 本地证据 |
|---|---|---|---|
| `always × false/true` | DCP，仍守 unit 边界 | 原路径 | 原周期 9 项回归覆盖；真实导出未重测 |
| `final_only × false` | 原无 HF 行为 | 原完整 DCP 路径 | 原周期分支 CPU 回归；真实收尾未重测 |
| `final_only × true` | 原 HF 直出 | 原完整 DCP 路径 | 原周期分支 CPU 回归；真实收尾未重测 |
| `hf_only × true` | HF 直出，不等 unit | HF，同步成功游标去重 | 周期/正常结束/AL结束/STOP复用路径/轮末/失败契约 |
| `hf_only × false` | 拒绝 | 拒绝 | 真实参数守卫 CPU 执行 |

## 验证与后续

本地 37 项纯 CPU 单测通过，包括扫描 × rescue 四组合、1000 步完整模拟、指定 Replay、真实 local_idx 数据对拍/缓存失效、初始评测失效不能冒充训练前基准、HF-only 分支和启动器预检。tokenizer 新增契约覆盖格式与长度差异可兼容，以及 ID、合并顺序、特殊 token、预分词规则和有效参数变化必须拒绝。另将当前调度器与基线 `956782f` 的调度器在 6 组旧配置上对拍，学习路线、任务状态/步数、停止原因和采样表一致。

```bash
PYTHONPATH=.:tests python -m unittest \
  test_al_skill_experiment test_al_data_audit test_al_hf_only \
  test_al_periodic_checkpoint test_skill_mix_task_launcher -v
```

本机 Python 3.13 的 pytest 在收集前退出 139，禁用插件仍复现，未把它计作测试通过；模拟测试替换 CUDA 计时同步以避免导入本机 torch。CPU 证据不包含真实张量导出、NCCL 集合通信或训练效果。

2026-10-15 实验目录上机证据：

- 主体提交 `6470669` 的 34 项新/周期单测通过（0.681 秒），182 项既有 CPU 回归通过（20.68 秒）；后者包括真实小张量 HF dtype 保存/读取，不代表 6B 模型保存已验证。原预检在 tokenizer 原始 JSON 比较处被拦截；只读取证后获批修正。
- 修正候选的服务器 7 项启动器契约通过，实际 CKPT9680 分片头、来源归一化与 tokenizer 预检通过。默认仅生成计划，未执行训练命令、未创建实验输出目录。
- 真实 Parquet 的 `episode_index/index/task_index` 列与 LeRobotDatasetMetadata 对拍通过：`put_object_cabinet` 为 40 train / 10 val，训练 10,769 帧；原始指令指向 cabinet。第二次命中缓存，签名 `ffd08b32b22aa7a140420ce17db86b9eab02bbc075148143c00d0f34472513f6`。报告在 `/data/al_cache/skill_mix_cabinet_audit.json`。本项未实例化完整 VLA DataLoader、未解码图像，不能视为全链训练已跑通。
- 日志：`/tmp/al_verify_skill_mix.log`（主体回归与原预检拦截）、`/tmp/al_tokenizer_candidate_check.log`（校验修正候选）、`/tmp/al_audit_skill_mix.log`（真实映射对拍）。

后续验收顺序：用户开卡并确认后，做 GPU 权重完整加载、独立验证与小步数 HF-only 验收 → 确认每份存档完整且没有 DCP → 再启动 1000 步实验。训练完后另行闭环比较四个 HF，固定 unseen、任务、seed 和推理参数；本轮不接闭环通关。

2026-10-15 后续准备：按用户确认，旧训练 HF 只保留 `al_v79_poolfix_7300/checkpoints/global_step_9460`、`global_step_9680`；已删除旧 4300/7260/7300/9240 及空 9701，释放约 59.5 GiB，可用约 76.8 GiB，两份保留 HF 的分片头/索引/关键资源复核通过。清理日志在 `/data/al_cache/cleanup_old_ckpts_20261015.log`，清理后默认启动预检退出 0，未启动训练。用户随后将新实验保存周期由 200 改为 250，CPU 契约覆盖四次保存、同一步结束去重和非评测边界的周期触发。

2026-10-15 同步前的机器检查：无训练进程，无 GPU，CPU 模式内存限额 2 GiB；文件系统可用约 18 GB，五份 6B BF16 HF 约需 60 GB，尚不满足完整实验存档容量要求。不自动清理历史产物；正式训练前须重新确认磁盘挂载与可用空间。生产分支 `e64388c` 上的既有 CPU 用例 `test_hf_export_dtype.py`、`test_state_machine.py`、`test_pass_thresholds.py` 共 49 项通过；这不是本次新代码的上机验收。

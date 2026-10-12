# al_v44 意外中断事故分析报告：`FrameTimestampError`

> 报告日期：2026-10-12
> 运行：`al_v44`（Auto Learning，7 卡 · GPU0,2,3,4,5,6,7）
> 中断时刻：**2026-10-12 06:10:15 UTC（本地 14:10）**，**Step 2500**
> 性质：**数据侧时间戳越界 + 评测严格取帧**共同导致的确定性崩溃
> 状态：**仅分析，未做任何修复**（本报告不含改动）

---

## 0. 一句话结论

数据集元数据里 `videos/<相机>/to_timestamp` **比视频真实长度多算了 1 帧**（15fps 网格上多 0.0667 s）；
训练时撞到这种样本会被"随机换帧重试"掩盖（本 run 已发生 38 次），
但**开环评测走的是严格路径（`strict_getitem`，不重试）** ⇒ 撞到即抛 `FrameTimestampError` →
rank0 评测失败 → 调度器抛错 → torchrun 退出码 1 → 训练终止。

---

## 1. 现象（现场记录）

| 项 | 实测值 |
|---|---|
| 崩溃时刻 | 06:10:15 UTC（本地 14:10） |
| 最后步数 | `Step 2500/20000`（进度 12.5%） |
| 进程状态 | `train_lingbotvla.py` / `torchrun` / `al_50task_bf16.sh` / `al_launch.py` **全部为 0**（已退出） |
| 退出码 | `TORCHRUN_RC=1`（`ChildFailedError`，首个失败 rank0 exitcode 1） |
| 最后存档 | `checkpoints/global_step_2400`（05:46，24G，233 文件，`extra_state` 完整） |
| `global_step_2500` | **不存在**（崩在存档之前） |
| `hf_milestones/` | 空 |
| 通关数 | 7/50（`target_total_passed_reached` **未**触发 ⇒ 不是正常收工） |
| GPU 状态 | GPU0/2–7 全 0%；**GPU1 仍 100%（已知坏卡，与本次无关）** |

⚠️ **判因陷阱**：日志末尾有一段"训练结束 / 收工原因：target_total_passed_reached(N)… all_tasks_resolved…"
—— 那是**固定模板文案**（脚本收尾时无条件打印），**不是**本次退出原因。
真正的判据是同一段日志上方的 `TORCHRUN_RC=1` 与 Python Traceback。

---

## 2. 直接原因（报错原文）

```
lerobot.datasets.video_utils.FrameTimestampError:
  One or several query timestamps unexpectedly violate the tolerance
  (tensor([0.0667]) > tolerance_s=0.0001)

  queried timestamps: tensor([3747.9333, 3749.2000])
  loaded  timestamps: tensor([3747.8000, 3747.8667, 3747.9333, 3748.0000, …
                              3749.0000, 3749.0667, 3749.1333])
  video: …/RoboTwin_lerobot_v30/videos/observation.images.cam_high/chunk-000/file-010.mp4
  backend: pyav
```

调用栈（自下而上）：

```
lerobot/datasets/video_utils.py:195  decode_video_frames_pyav → raise FrameTimestampError
lerobot/datasets/dataset_reader.py:281 _decode_single → decode_video_frames
lingbotvla/auto_learning/orchestration/scheduler.py:619 _select
  → raise RuntimeError(f"多卡评测在 rank0 失败：{got.get('error')}")
```

**关键数字**：

| 量 | 值 | 说明 |
|---|---|---|
| 请求的第二个 ts | **3749.2000** | = 该视频容器 `duration` |
| pyav 实际能解出的末帧 ts | **3749.1333** | = `duration − 1/15` |
| 差值 | **0.0667 s** | **正好一帧**（15 fps ⇒ 1/15 = 0.0667） |
| 容差 `tolerance_s` | **0.0001** | lerobot 默认值（**远严于半帧 0.0333**）⇒ 必然报错，不是抖动 |

---

## 3. 根因（数据侧，已定位到具体 episode）

### 3.1 触发点：episode 2354

| 字段 | 值 |
|---|---|
| `episode_index` | **2354** |
| `length` | **270** |
| `videos/…/cam_high/file_index` | 10（`file-010.mp4`） |
| `videos/…/cam_high/from_timestamp` | **3731.2667** |
| `videos/…/cam_high/to_timestamp` | **3749.2000** ← 等于视频时长 |
| 任务描述 | `"Open the drawer of the cabinet with gray sides and set the cardboard playing cards box inside."` |

该 episode 的**最后一帧**查询时间戳 = `from + (length−1)/fps` = `3731.2667 + 269/15` = **3749.2000**
⇒ 比视频真实末帧（3749.1333）**大整整一帧** ⇒ 这一帧在视频里**不存在**。

### 3.2 视频真实规格（ffprobe 实测，非推测）

```
nb_read_packets = 56238      （逐包计数，与 nb_frames 一致）
duration        = 3749.200000
r_frame_rate    = 15/1
⇒ 真实末帧 ts = (56238−1)/15 = 3749.1333
```

### 3.3 元数据写法的问题

`meta/episodes/*.parquet` 里以 `from_timestamp` / `to_timestamp` 描述"该 episode 占用视频的哪一段"：

```
file-010 的最后两段：
  ep=2353  len=273  from=3713.0667  to=3731.2667
  ep=2354  len=270  from=3731.2667  to=3749.2000   ← to == 视频时长
```

**若语义是"半开区间 [from, to)"**，则 `to = from + len/fps` 是对的，查询必须用 `from + i/fps`（`i = 0…len−1`），
此时 `to` **只是边界值、不应被当作帧时间戳使用**；但**该 episode 的末帧恰好落在 `from + (len−1)/fps`**
—— 需要确认查询侧是否把 `to` 当成了"末帧时间"（见 §6 待确认项）。

### 3.4 全局规模：不是个例

逐视频核对"所有 episode 的 `length` 之和"与"视频真实帧数"：

| 视频 | 段内帧和 | 视频真实帧数 | 差 |
|---|---|---|---|
| file-000 | 53112 | 53041 | **+71** |
| file-001 | 39603 | 39534 | **+69** |
| file-002 | 54631 | 54561 | **+70** |
| file-003 | 49192 | 49117 | **+75** |
| file-004 | 57421 | 57383 | **+38** |
| file-005 | 56507 | 56414 | **+93** |
| file-006 | 11884 | 11879 | **+5** |
| file-007 | 55023 | 54983 | **+40** |
| file-008 | 47304 | 47247 | **+57** |
| file-009 | 27374 | 27340 | **+34** |
| **file-010** | **56322** | **56238** | **+84** |
| file-011 | 40520 | 40461 | **+59** |
| **合计** | — | — | **约 +695 帧** |

**12/12 个 camera 视频都"登记的帧数比视频实际多"** ⇒ 这是**系统性元数据缺陷**，不是单条坏样本。
每个视频的 `to_timestamp` 上界都精确等于该视频 `duration`（= 真实末帧 + 1/15）。

---

## 4. 为什么训练没崩、评测崩了（决定性区别）

`lingbotvla/data/vla_data/multi_vla_dataset.py` 的 `__getitem__` 有两个分支：

```python
# [DSH] strict：评测用。直接取，失败就抛错（不做随机换帧重试）。
if getattr(self, "strict_getitem", False):
    return self.getdata(idx)

max_retries = 200
attempts = 0
cur = idx
while attempts < max_retries:
    try:
        return self.getdata(cur)
    except Exception as e:
        last_err = e
        attempts += 1
        ...
        logger.info(f"Last error: {repr(last_err)},\n Dataset: {dataset.dataset.repo_id}")
        cur = np.random.randint(0, len(self))   # ← 随机换一条重试
        continue
raise RuntimeError(f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. …")
```

| 路径 | 行为 | 结果 |
|---|---|---|
| **训练取数** | 失败 ⇒ **随机换 idx 重试**（上限 200） | 坏样本被**静默跳过**，训练继续 |
| **开环评测** | `strict_getitem=True` ⇒ **直接取，失败即抛** | 撞到坏样本 ⇒ **评测失败 ⇒ 整个 run 退出** |

**实测证据**：日志里 `Last error: FrameTimestampError` 共 **38 条**，
时间跨度 **00:13:20 → 04:36:54**（约 4.4 小时），涉及 `file-001 / file-003 / file-010` 三个视频
⇒ **训练期间这个错误一直在发生，只是每次都被重试消化掉了**。
评测侧一撞到（06:10:15）就是终结。

---

## 5. 影响面与代价

| 项 | 影响 |
|---|---|
| 训练进度 | 停在 2500/20000（12.5%） |
| **可用续训点** | **`global_step_2400`**（含 `extra_state` ⇒ 注册表/调度器状态完整） |
| **丢失** | 2400 → 2500 共 **100 步**（约 **17 分钟**，按 10.6 s/it） |
| 通关 | 7/50（2400 时点状态） |
| 数据完整性 | 12 个 cam_high 视频共约 **+695 帧**的登记冗余；受影响的 episode = **每个视频的最后一段**（约 12 条，待逐条确认） |
| 隐蔽性 | **高**：训练侧无感（仅 INFO 日志），只在评测时爆发；崩溃点与"任务难度/配置改动"均无关 |
| 与近期改动的关系 | **无关**（轨迹分片 `AL_EVAL_SHARD` 默认关闭；崩溃发生在 `_select` 的评测调用内） |

---

## 6. 待确认项（本报告未验证，修之前必须先查清）

1. **查询侧语义**：`dataset_reader._query_videos` 里 `shifted = from_timestamp + ts`，其中 `ts` 来自
   `hf_dataset[...]["timestamp"]`（数据列，非计算）。需逐值 dump `episode 2354` 的 `timestamp` 列，
   确认**最后一行的 `timestamp` 是否等于 `to − from`**（即 `len/fps` 而非 `(len−1)/fps`）：
   - 若**是** ⇒ 缺陷在**数据行的时间戳列**（写多了一帧）；
   - 若**否** ⇒ 缺陷在**查询/换算路径**（把 `to_timestamp` 当帧时间用）。
2. **受影响的 episode 全集**：遍历 2500 条 episode，找出"末帧查询 ts > 所属视频真实末帧"的全部清单
   （预计约 12 条，每个视频 1 条）。
3. **是否影响训练质量**：这 ~695 帧的登记冗余是否导致正常样本被"错位取帧"
   （即 `from_timestamp` 与实际帧位置是否逐帧对齐）—— 需抽样比对若干 episode 的首/中/末帧。
4. **其他相机**：本报告只核对了 `cam_high`；`cam_left_wrist` / `cam_right_wrist` 尚未核对。

---

## 7. 可选修法（**仅列出，均未实施**）

| 方案 | 做法 | 代价 | 风险 / 副作用 |
|---|---|---|---|
| **A. 评测侧受限容错** | 给评测路径加"仅对 `FrameTimestampError`"的受限重试或跳过 + **显式告警日志**；其他异常仍严格抛错 | 改 1 处、约 20 行 | 低。但会让评测**静默少一条轨迹的末帧** ⇒ 必须打印告警并计数，否则指标口径悄悄变化 |
| **B. 修数据（治本）** | 修正 `meta/episodes` 的 `to_timestamp`（或在源数据修正时间戳列），使末帧时间 = `from + (len−1)/fps` | 需改 2500 条 episode 元数据 / 重建该数据集 | 中。动数据 ⇒ 需备份 + 全量校验；但根治，且让训练侧不再靠重试掩盖 |
| **C. 放宽容差** | 将评测 `tolerance_s` 从 `1e-4` 调到 ≥ 1/15（≈0.0667） | 改 1 行 | ⚠️ **不建议**：会连"取错帧"一起放行（差 1–2 帧也算通过），指标口径被污染且难以察觉 |
| **D. 维持现状** | 不修，接受"评测可能随时崩" | 0 | 高。候选池里只要出现受影响任务，评测就会再次终止 |

**倾向**：先做 §6 的 1–2 项确认（纯只读），把"缺陷在数据行还是换算路径"钉死，再在 A / B 之间选。
无论选哪个，都建议**同时**把评测侧的"坏样本"变成**可见**（告警 + 计数），而不是继续静默重试。

---

## 8. 复发时的快速判读清单

出现"训练不动了"时，按此顺序看（**不要**只看日志末尾的收工模板）：

1. `pgrep -cf "[t]asks/vla/train_lingbotvla.py"` → 0 表示进程已退出（不是卡住）；
2. 日志尾部搜 `TORCHRUN_RC` / `Traceback` / `FrameTimestampError`；
3. `grep -c "Last error: FrameTimestampError"` 看是否长期被重试掩盖；
4. 确认最后一份完整 DCP 存档（`checkpoints/global_step_*` 且 `extra_state` 存在）；
5. 报错里的 `video:` 路径 + `queried timestamps` 用来定位具体 episode：
   `queried_ts − 该视频真实时长` 若等于 `1/fps`，即本篇所述的越界帧。

---

## 9. 复现与验证入口（纯只读，可在训练进行时执行）

```bash
# ① 视频真实规格（逐包计数，比 nb_frames 可信）
ffprobe -v error -select_streams v:0 -count_packets \
  -show_entries stream=nb_read_packets -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 \
  <dataset>/videos/observation.images.cam_high/chunk-000/file-010.mp4

# ② 该 episode 的登记信息
python -c "import pyarrow.parquet as pq; …"   # 读 meta/episodes/*.parquet，筛 episode_index==2354

# ③ 训练日志里该错误的历史
grep -c "Last error: FrameTimestampError" <run>.log
```

---

## 附录：本次分析用到的关键文件与坐标

| 内容 | 路径 |
|---|---|
| 崩溃日志 | `/models/robotwin-persistent/al_runs/logs/train_al_v44.log` |
| 可用存档 | `/models/robotwin-persistent/al_runs/train_al_v44/checkpoints/global_step_2400` |
| 数据集 | `/models/robotwin-persistent/data/agg_lerobot_v30/RoboTwin_lerobot_v30` |
| 数据集元数据 | `…/meta/info.json`、`…/meta/episodes/*/file-*.parquet` |
| 触发视频 | `…/videos/observation.images.cam_high/chunk-000/file-010.mp4` |
| 训练侧重试/严格分支 | `lingbotvla/data/vla_data/multi_vla_dataset.py`（`strict_getitem` / `max_retries=200`） |
| 评测调用点 | `lingbotvla/auto_learning/orchestration/scheduler.py` `_select`（多卡评测 rank0 失败即抛） |
| 解码器 | `lerobot/datasets/video_utils.py::decode_video_frames_pyav`（`tolerance_s` 默认 `1e-4`） |

# checkpoint 磁盘容量保护 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关参数：`--train.disk_guard` / `--train.disk_guard_margin` / `--train.disk_check_interval`
> 相关文件：`lingbotvla/utils/checkpoint_guard.py`、`lingbotvla/utils/async_hf_checkpoint.py`

---

## 1. 这个功能解决什么问题

一次训练会周期性产生 checkpoint（DCP + 异步 HF 转换），单份体积很大。
如果磁盘写满，会出现：

- **半截 checkpoint** —— 存了一半就失败，文件不可用
- **训练白跑** —— 前面几小时的算力全部浪费
- **拖垮同机任务** —— 磁盘写满影响其它进程

**本功能的目标**：在当前 checkpoint **完整保存之后**，判断剩余空间还够不够存下一份；
不够就**正常结束训练**，绝不写出半截文件。

---

## 2. 一句话原理

> 后台 HF 线程在 checkpoint 写完的那一刻实测它的真实体积；
> 主循环周期性用「当前剩余空间」和「历史最大体积 × 1.1」比较；
> 不够就所有 rank 一致优雅退出。

---

## 3. 快速开始

**什么都不用改** —— `disk_guard` 默认关闭，现有命令行为完全不变。

要开启，训练命令加三个参数即可：

```bash
--train.disk_guard true \
--train.disk_guard_margin 1.1 \
--train.disk_check_interval 50
```

**完整示例**（Phase 1，4×PRO6000）：

```bash
cd /data/code/lingbot-vla-v2
export PATH=/data/miniconda3/envs/lingbotvla/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash train.sh tasks/vla/train_lingbotvla.py \
  /data/train/configs/robotwin_official_paths.yaml \
  --data.train_path        /data/train/phases/datasets.txt \
  --data.episode_ids_file  /data/train/phases/phase1_L1.episode_ids.json \
  --train.output_dir       /data/outputs/phase1_L1 \
  --train.micro_batch_size 28 \
  --train.gradient_accumulation_steps 1 \
  --train.global_batch_size 112 \
  --train.num_train_epochs 1 \
  --train.max_steps        50000 \
  --train.save_steps       545 \
  --train.save_epochs      1 \
  --train.save_hf_weights  true \
  --train.async_save_hf_weights true \
  --train.enable_resume    false \
  --train.train_expert_only true \
  --data.image_augment     true \
  --train.disk_guard true \
  --train.disk_guard_margin 1.1 \
  --train.disk_check_interval 50
```

---

## 4. 工作原理

### 4.1 测量（不阻塞训练）

```
主循环                    后台 HF 线程
   │
   ├─ 采样 disk_avail_before
   ├─ Checkpointer.save()          ← DCP 写盘
   ├─ 提交异步 HF 任务  ─────────►  HF 转换 / 写盘
   │                                  │
   │  （训练继续，不等待）             ├─ HF 全部写完 ★
   │                                  ├─ 采样 disk_avail_after
   │                                  └─ checkpoint_used
   │                                       = before - after
   ▼
 下一次安全检查点 ◄─────────────── 消费上面的结果
```

- `disk_avail_after` 是在 **HF 真正写盘完成的那一刻**、**在后台线程内**采样的
- 训练主循环**完全不阻塞** —— 正常情况下 HF 转换远快于存档间隔

### 4.2 判断（rank0 计算 + 广播）

```python
max_checkpoint_used = max(max_checkpoint_used, checkpoint_used)   # 只增不减
next_checkpoint_required = max_checkpoint_used * margin          # 默认 ×1.1

if 当前剩余空间 >= next_checkpoint_required:
    继续训练
else:
    当前 checkpoint 已完整保存 → 停止训练
```

**为什么必须广播**：实测占用只在 rank0 上可得（异步 HF 只在 rank0 跑）。
如果各 rank 各自判断，会出现 rank0 认为「空间不足」而 rank1/2/3 认为
「无历史 → 放行」的分歧 —— 那样 rank0 退出循环，其余 rank 却继续进入
`Checkpointer.save()` 里的 distributed collective，**直接死锁**。

所以决策统一由 rank0 计算，再用 `dist.broadcast_object_list` 同步给所有 rank。

### 4.3 停止方式

采用**已有的多 rank 正常退出路径**，不引入信号处理 / 看门狗：

```
break 步循环 → break epoch 循环
  → hf_saver.wait_all_across_ranks()   等 rank0 的 HF 收尾
  → dist.barrier()
  → dist.destroy_process_group()
```

**当前 checkpoint 一定是完整的** —— 判断发生在写下一份**之前**。

### 4.4 作用域：单次训练

`max_checkpoint_used` 是**进程内变量，不落盘**：

- 每次启动训练都从 `0.0` 重新开始
- 不会跨 Phase 沿用
- 从 Expert-only 切到 Freeze Vision / Full FT 后，optimizer state 变大，
  新的一次训练会**重新测量**，不会被上一轮的旧数字误导

---

## 5. 日志怎么读

### 正常情况下

```
[DiskCheck] step=545 disk_avail_before=500.0GB disk_avail_after=430.0GB checkpoint_used=70.0GB max_checkpoint_used=70.0GB
[DiskCheck] (按步存档前 step 1090) max_checkpoint_used=70.0GB next_checkpoint_required=77.0GB disk_avail_now=429.9GB continue_training=true
```

| 字段 | 含义 |
|---|---|
| `disk_avail_before` | 本次 checkpoint 开始前的可用空间 |
| `disk_avail_after` | **本次 checkpoint 完整保存后**的可用空间 |
| `checkpoint_used` | 本次 checkpoint 的实际净占用 = before − after |
| `max_checkpoint_used` | 本次训练见过的最大的 checkpoint 体积 |
| `next_checkpoint_required` | 下一份需要预留的空间 = max × margin |
| `disk_avail_now` | **此刻**的实时可用空间 |
| `continue_training` | `true` 继续 / `false` 停止 |

### 首次存档（还没有历史参考）

```
[DiskCheck] (按步存档前 step 545) 尚无历史占用参考; 放行 (disk_avail_now=500.0GB)
```

### 因磁盘余量不足停止

```
[DiskCheck] (轮末存档前 epoch 1 step 779) max_checkpoint_used=71.0GB next_checkpoint_required=78.1GB disk_avail_now=69.0GB continue_training=false
[DiskCheck] 决策: 停止训练 (reason=disk)
[DiskCheck] 剩余空间不足以再存一份 checkpoint, 当前 checkpoint 已完整保存, 不再继续训练。
[DiskCheck] 训练因磁盘余量不足提前结束 (max_checkpoint_used=71.0GB, global_step=779)。已保存的 checkpoint 均完整。
```

### 因异步 HF 保存失败停止

```
[checkpoint] step=545 异步 HF 保存失败 (OSError(28, 'No space left on device')); DCP 仍有效, 停止后续训练。
[DiskCheck] 决策: 停止训练 (reason=hf_failed)
[checkpoint] 上一份异步 HF 保存失败 (DCP 仍有效), 不再继续训练。
[checkpoint] 训练因异步 HF 保存失败提前结束 (DCP 已成功保存, global_step=545)。
```

### DCP 保存失败（作业直接失败，这是预期行为）

```
[ERROR] [checkpoint] DCP checkpoint save failed at global_step=545; aborting training.
Traceback (most recent call last):
  ...
OSError: [Errno 28] No space left on device
```

---

## 6. 参数详解

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--train.disk_guard` | bool | **`False`** | 总开关。关闭时全部容量检查代码都不执行，**行为与改动前完全一致** |
| `--train.disk_guard_margin` | float | `1.1` | 安全系数。下一份需要 `max_checkpoint_used × margin` 的空间 |
| `--train.disk_check_interval` | int | `50` | 检查间隔（步）。`0` = 只在存档点前检查 |

### 关于检查时机

除了**每个存档点前必查**，还会**每 N 步检查一次**。原因：

如果不做周期检查，判断只在存档点前发生 —— 那么「空间已不足」这件事
最坏要等一整个 `save_steps` 才会被发现，白训 `save_steps × StepTime`。

| 配置 | 最坏白训时长（`save_steps=545`, `StepTime=3.31s`） |
|---|---|
| `disk_check_interval=0` | 545 步 ≈ **30 分钟** |
| `disk_check_interval=50`（默认） | 50 步 ≈ **2.8 分钟** |
| `disk_check_interval=20` | 20 步 ≈ **1.1 分钟** |

周期检查是一次 `broadcast`（毫秒级），开销可忽略。

---

## 7. 边界与失败处理

### 7.1 三条 fail-open 路径（绝不把训练搞崩）

| 情况 | 行为 |
|---|---|
| 读取磁盘空间失败 | 打 warning，跳过本轮判断，**放行** |
| `checkpoint_used <= 0`（实测无效） | 打 warning，跳过本轮判断，**放行** |
| 收集上一份结果时抛异常 | 打 warning，跳过本轮判断，**放行** |

> 设计取向：容量保护本身**不能成为新的故障点**。

### 7.2 DCP 保存失败 → 整个作业失败退出

```
logger.exception(...)   # 完整 traceback
raise                   # 直接抛出
```

- **不 catch 后继续训练**
- **不在异常路径里做任何 distributed collective** —— 此时某些 rank 可能已经卡在
  checkpoint collective 上，再插入集合操作会形成**二次死锁**
- 交给 torchrun 正常 teardown 整个多卡作业

### 7.3 异步 HF 保存失败 → 保留 DCP，正常退出

- DCP 此时**已经成功**，所以不 crash 进程，也不删除已保存的 checkpoint
- 后台线程置 `hf_save_failed = True`
- 主循环在**下一个安全检查点**收取该标记
- rank0 判定 `stop=True, reason="hf_failed"`，**广播给所有 rank**
- 所有 rank 一致正常结束训练

> 第一版**不区分** ENOSPC / IO error / 其它异常，统一按失败处理。

### 7.4 并发安全

沿用现有机制，未新造轮子：

- `AsyncHFCheckpointSaver` 用 `ThreadPoolExecutor(max_workers=1)` → 单线程，
  物理上同时只有 1 个 HF 转换任务
- `--train.async_hf_max_pending` 默认 `1`
- 新增的 `drain_pending_best_effort()` 与 `wait_all_best_effort()` 的区别是
  **不关闭 executor**，因此可在训练过程中反复调用

---

## 8. 常见问题

**Q：会不会影响正常训练？**
不会。`disk_guard` 默认 `False`，4 处容量检查全部跳过。唯一无条件改动是
DCP 保存包了一层 `try/except` —— 成功路径行为完全相同。

**Q：会拖慢训练吗？**
正常情况下不会。测量在后台线程完成；判断每 N 步一次 `broadcast`，
毫秒级开销；并且调用时 HF 任务通常早已完成，不会阻塞。

**Q：为什么 `max` 只增不减？**
保守起见。万一某一份偏小（例如 69.5G），预留仍按历史最大 71.0G 算，
避免因一次偏小就误判放行。

**Q：`max_checkpoint_used` 会存到 checkpoint 里吗？**
不会。它是纯进程内变量，不落盘，每次启动从 0 开始，也不影响 resume 兼容性。

**Q：`disk_check_interval` 设多少合适？**
默认 50 已经够用。想更保险就调小（代价是更频繁的 broadcast）。
设 `0` 表示只在存档点前检查（最坏白训一整个 `save_steps`）。

**Q：磁盘扩容后还需要这个吗？**
建议开着。它同时兜住了「别的进程把盘写满」这类外部因素。

**Q：`Checkpointer.save` 写盘过程中磁盘写满怎么办？**
会抛异常 → 走 7.2 的路径，作业失败退出。容量保护是把这类情况**提前挡掉**，
但挡不住「两次检查之间磁盘被外部写满」。

---

## 9. 测试

```bash
cd /data/code/lingbot-vla-v2
python tests/test_disk_guard.py          # 11 项, 无需 GPU
python tests/test_disk_guard.py -v       # 附带详细日志
```

覆盖范围：

| # | 测试 |
|---|---|
| T1 | DCP save 正常 |
| T2 | DCP save 抛异常 → **re-raise** |
| T3 | HF async 正常 |
| T4 | HF async 抛异常 → `hf_save_failed=True` |
| T5 | 多卡收到相同 `stop=True`（4 rank 一致性） |
| T5b | HF 失败的 `reason` 也广播给所有 rank |
| T6 | **HF 失败后不再进入下一次 `Checkpointer.save()`** |
| T6b | 余量不足时不存档 |
| T7 | 首次放行 / `max` 只增不减 |
| T8 | 三条 fail-open 路径 |
| T8b | 读盘助手边界（路径不存在） |

---

## 10. 不做什么（明确的边界）

- ❌ 不引入外部 watchdog / 守护进程
- ❌ 不使用 SIGTERM / SIGKILL
- ❌ 不引入额外信号处理
- ❌ 不修改 DCP 格式
- ❌ 不修改 HF 格式
- ❌ 不修改 resume 逻辑
- ❌ 不修改 optimizer 状态结构
- ❌ 不修改评测加载方式

**本功能只负责一件事**：当前完整 checkpoint 保存完之后，判断还够不够空间存下一份。

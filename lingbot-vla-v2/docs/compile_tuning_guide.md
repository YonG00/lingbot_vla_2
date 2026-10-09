# 编译开销治理（torch.compile）：评测禁编译 + dynamo 重编译上限

> 起因：2026-10-09 50-task GMean200 正式跑（GBS=24，96G）**31 分钟里真正训练只有 10.8 分钟（35%）**，
> 任务切换处出现 **353 秒（5 分 53 秒）无日志空窗**（该窗口 220 行 `torch/_dynamo` 警告、**0 次评测**）。

## 1. 两个已确认的成因

| 成因 | 机制 | 代码位置 |
|---|---|---|
| **A. 评测期翻转 dynamo guard 监视的状态** | 每次评测临时改 `config.use_cache`、`attention_implementation`、`model._use_compile_predict_velocity`、逐模块 `.training`，进出各一次 ⇒ 编译图 guard 失效 ⇒ 重编译。外层是**整模型**图（6B），重编译极贵。 | `lingbotvla/utils/open_loop_validation.py` 的 `safe_eval_context` |
| **B. dynamo 重编译上限太小（默认 8）** | 超过 `cache_size_limit` 后 dynamo **放弃编译、整段退回 eager** ⇒ 警告刷屏 + 之后每步变慢（日志里那 220 行就是这个） | `torch._dynamo.config.cache_size_limit` |

⚠️ `TORCHDYNAMO_CACHE_SIZE_LIMIT` 环境变量在 torch 2.8 上**不生效**（实测仍为 8）⇒ 必须代码设置。

## 2. 已落地的改动（默认安全、可回退）

新增 `lingbotvla/utils/compile_tuning.py`：

| 函数 | 作用 | 回退开关 |
|---|---|---|
| `apply_dynamo_tuning()` | 把 `cache_size_limit` 提到 **64**（幂等；在训练器 `torch.compile(model)` **之前**调用） | `AL_DYNAMO_CACHE_SIZE_LIMIT=0` 跳过；也可给其它整数覆盖 |
| `eval_compile_disabled()` | 评测区整体"禁编译"（**优先 `torch.compiler.set_stance("force_eager")`**） | `AL_EVAL_DISABLE_COMPILE=0` |

**接线点**：
- `tasks/vla/train_lingbotvla.py`：`if args.train.use_compile:` 分支内、`torch.compile(model)` **之前**调用 `apply_dynamo_tuning(logger=logger)`；
- `lingbotvla/utils/open_loop_validation.py`：`safe_eval_context` 在 `try:` 前 `__enter__()`、在 `finally:` 首行 `__exit__()`（**异常路径也保证退出**）。

**为什么评测禁编译是安全的**：评测本来就强制走 eager 热点路径
（`_use_compile_predict_velocity = False` + `torch.inference_mode()`），
关掉的只是"外层编译图在评测模式下的重编译"；**数值不变**（算子实现相同）。

**torch 版本差异（实测）**：
- torch 2.8：`torch.compiler.disable()` **不能**当上下文管理器（`RuntimeError: torch._dynamo.optimize(...) is used with a context manager`）；
  `torch.compiler.set_stance("force_eager")` **可以** ⇒ 优先它；
- 代码按「`set_stance` → `compiler.disable` → `torch._dynamo.config.disable`」**择优探测**并缓存选择，日志会打印实际机制。

## 3. 怎么验证（下次开卡）

### 3.1 先诊断"到底是谁触发重编译"（~10 分钟）
```bash
TORCH_LOGS=recompiles,guards TORCHINDUCTOR_CACHE_DIR=/data/inductor_cache \
  <原启动命令，步数设为跨 1 个 unit 边界 + 1 次任务切换（+150 步）>
```
- trigger 含 `use_cache` / `attention_implementation` / `.training` / `_use_compile_predict_velocity` ⇒ **A**
- trigger 含 token 长度 / 序列维符号 ⇒ **B（形状）** ⇒ 才考虑动态形状或形状分桶

### 3.2 量收益（基线已实测）
| 指标 | 基线（每 40 分钟跑） |
|---|---|
| 稳态步时 | **2.583 s/step** |
| unit 评测墙钟 | **~9 s**（4 轨迹）；scout 2–12 s |
| 任务切换重编译空窗 | **353 s**（一次） |
| 总时长构成 | 纯训练 10.8 min / Bootstrap 7 min / 首次编译预热 ~6 min / 边界评测+重编译 ~7 min |

验收口径：**同一任务**跑 ≥150 步，比较 `StepTime` 与"空窗"是否消失；评测墙钟不应变慢 >10%。

## 4. 尚未做（等实测决定）

| 候选 | 说明 |
|---|---|
| ① 内层 `dynamic=False` → `True` | `modeling_lingbot_vla_v2.py:981`；动态形状免"每任务一套编译"，但**稳态可能变慢** ⇒ 必须 A/B（慢 >5% 就不划算） |
| 形状分桶（备选） | 把指令长度补齐到少数几档固定长度 ⇒ 静态且编译次数有界，通常比纯 dynamic 更划算 |
| ④ 关掉外层整模型编译 | 项目内已有 denoise 热点的小段编译；外层图大、guard 多、重编译贵 ⇒ 需 A/B |

## 5. 测试

`tests/test_compile_tuning.py`（5 项）：默认值/幂等/env 覆盖/非法值、`=0` 回退、
**区间内确实不再编译**（用 dynamo counters 断言）、以及接线检查（AST：`safe_eval_context`
必须在 `finally` 里退出、`apply_dynamo_tuning` 必须在 `torch.compile` 之前）。

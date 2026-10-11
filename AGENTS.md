# AGENTS.md —— LingBot-VLA-v2 项目导航（**只做指针，不放状态与结论**）

> 分层记忆：本文件（导航）→ `.workbuddy/memory/MEMORY.md`（索引 + 当前状态 + 红线）→
> `.workbuddy/memory/knowledge/*.md`（按主题的经验库）→ `.workbuddy/memory/YYYY-MM-DD.md`（当日流水）。
> **重要结论写进知识库与工作日志，不写进本文件**（维护约定见 §3）。

## 0. 工程坐标

- 工作区：`/Users/mac/Documents/lingbot_v2`（本文件所在处）
- 仓库：`lingbot_vla_2/lingbot-vla-v2`（package `lingbot-vla-v2`），分支 `feature/auto-learning-v1`
- 训练机：AutoDL，`bash ssh_srv.sh ctrl '<cmd>'`；代码在 `/data/code`；Python 用  
  `/data/miniconda3/envs/lingbotvla/bin/python`（本机 `/opt/anaconda3/bin/python` 缺 torchdata/新版 transformers ⇒ 重依赖测试只能在机器上跑）
- 机器同步（**唯一**正确方式）：`cd /data/code && git fetch -q origin && git reset --hard origin/feature/auto-learning-v1`
- **推送约定（用户 2026-10-09 定）**：**用本机 push**（训练机虽也能推、有写权限，但实测更慢）。
  本机 commit（身份正确）+ `git push origin feature/auto-learning-v1`；push 单独一条命令、不与大任务串。
  实测：慢的是到 GitHub 的国际链路，两边都 3–13 s；机器默认 git 身份已修为与本机一致（备用）。
- **同步口径（用户 2026-10-09 定）**：默认**只做两方对齐** = **训练机 ↔ GitHub**。
  不再做「本机/机器/远端」三方对比、不为对齐而额外 fetch/reset 本机；
  **本机只在用户明确要求时再更新**（别把时间花在本机同步上）。
  交付验证只需给出：`machine HEAD == origin/<branch>` + 机器测试结果。
- **本机 AGENTS.md/记忆不在 git 仓库内**（仓库是内层目录）⇒ 改这里不需要 commit；改仓库代码才 commit

> **分支约定（2026-10-09 用户定，已固化）**：**自动学习相关的所有工作（含 Eval Batch 诊断/修复）
> 一律在 `feature/auto-learning-v1` 上做，不开子分支、不开临时分支**。
> GitHub 上只保留 3 个分支：`main` / `feature/auto-learning-v1` / `feature/quick-eval-multigpu-v1`。
> （曾经的 `feature/eval-batch-diag-fix` 已 fast-forward 合回 `feature/auto-learning-v1` 并删除，
> 其 3 个提交成为 `8b51f0f` / `27cc8ff` / `aec8472`，内容一字未改。）
> ⚠️ `git reset --hard origin/<branch>` 会移动**当前所在分支**的指针 —— 在机器上同步前先确认
> `git branch --show-current` 是 `feature/auto-learning-v1`，否则会把别的分支指针带跑。

## 1. 铁律（血泪教训）

1. **测试全绿前不得提交**（曾带失败测试提交 `a484009`，后用 `6f9fcff` 修）。
2. `git commit` **一律 `-F <文件>`**：`-m "…反引号…"` 会被 shell 命令替换，消息被破坏。
3. **改代码前先 `grep` 目标行是否存在**；按**行号/谓词**定位，不做整段字符串替换（缩进/空格多次踩坑）。
4. `git apply --check` 用**绝对路径**（相对路径 + cd 变化 ⇒ `can't open patch`）。
5. `pkill -f "<模式>"` 会匹配**执行它的那条命令自身** ⇒ 先杀掉自己会话、后续命令不执行！  
   用方括号：`pkill -f "[r]atio-gpu"`。
6. 沙箱/长任务：机器侧长跑用 `setsid nohup env … > log 2>&1 < /dev/null &`，再轮询。
7. **GPU 开关机、租卡、删历史产物由用户决定**；用户明说"关机"才关机。
8. **长任务一律用后台作业**：发起工具调用时就带 `run_in_background: true`，**同一条调用里绝不 sleep/tail/轮询**
   （2026-10-09 连踩两次：把 push+自测+启动串成前台命令 ⇒ 用户被迫强杀）。发起后立刻回话，跑完由通知接结果。
   🔴 **2026-10-10 用户定为契约（第三次重申）**：远端取证 / 日志轮询 / 扫描 / 训练 / pytest **全部**后台；
   远端用 `setsid nohup … > 日志 2>&1 < /dev/null &`，读日志也另起一条后台调用。前台只负责"发起"与"回话"。
   🔴 **后台作业一律 `wait: false`（或直接取回执）**：`job_output(wait=true)` / `wait_agent(timeout)` 等待
   **等于把后台当前台用**，同一条禁令。只在作业已明确完成、或完成通知已到达时才读结果。
   （2026-10-10 两次违反：`job_output wait:true` 挂 600 s / 900 s，用户打断两次。）
9. **上 GPU 前先跑 CPU 端整链自测**（`tools/eval_batch_stochastic_acceptance.py --selftest` 之类）：
   2026-10-09 两次 GPU 试跑都死在**接口契约**上（`_infer_core` 返回 list[dict]；`ft.unapply()` 无 `"actions"` 键），
   这些本可 1 秒内在 CPU 暴露 ⇒ **每次 GPU 炸出来的 bug，必须补一条 CPU 契约测试**。
10. 机器侧 `pgrep -f "<名>"` 会**自匹配执行它的那条命令** ⇒ 一律用方括号（`[s]tochastic_acceptance`），否则永远判"在跑"。
11. 🔴 **动手前先获批（2026-10-10 用户定，最高优先级；当日两次重申）**：**默认只读 + 提建议**；改代码、推机器、
    跑训练/扫描、停进程、提交、push、建分支、删文件，一律**先说方案 → 等用户点头 → 再执行**。用户原话：
    「我没让你动手，你就先别动可以吗？或者跟我提建议，我同意你再动」；
    「你只是提供建议，具体行动需要我的允许。**特别是遇到问题，先不急着动手，让我明白先，我同意再行动**」。
    ⚠️ **不要把"用户同意的那一步"顺手扩权**（当天：用户说"推机器"，我连带停进程/整树覆盖，被叫停）。
    ⚠️ **遇到报错/异常时的默认动作 = 只读取证 + 讲清楚 + 给选项**，不是"先修一下试试"。
    允许无需请示的动作只有：读文件、只读取证（`/proc`、日志、`git status/log`、md5）、写工作区记忆。
12. 🔴🔴 **禁止对后台作业使用任何"等待"语义（2026-10-10 用户第 4 次纠正后升级为硬约束）**：
    **`job_output(wait=true)` / `job_output(timeout_ms=…)` / `wait_agent(timeout…)` 一律不得使用**，
    这是本会话最高频的违规（`wait:true` 共犯 4 次，前 3 次已写进铁律 8 仍复发 ⇒ 说明"写规则"不够）。
    **唯一允许的读取方式 = 无参数 `job_output(job_id)`**（已完成的作业会返回结果，未完成返回 running）。
    判定一个作业是否完成，**只看完成通知**，或用另一个后台调用去远端取证；**绝不等待**。
    自检（每次要调 `job_output` 前默问一句）：「我这条调用带了 wait/timeout 参数吗？」带了 ⇒ 删掉再发。
    （同日教训：模型加载/扫描期间的"空转等待"没有产出任何信息，用户看到的是卡住。）


## 2. 状态 / 待办（**指针，不放内容**）

> 本文件**只做导航**。状态、结论、经验一律不写在这里（曾涨到 34 KB / 593 行后开始失真）。

| 想知道什么 | 去哪 |
|---|---|
| **当前 HEAD / 未提交状态 / 下一步** | `.workbuddy/memory/MEMORY.md` §当前状态 / 待办 |
| **今天的流水（做了什么、实测多少）** | `.workbuddy/memory/YYYY-MM-DD.md`（最新一篇） |
| 评测链路 / 口径 / 归因 | `.workbuddy/memory/knowledge/eval.md` |
| 改代码、可疑 bug、**已修 bug 台账** | `.workbuddy/memory/knowledge/codebase.md` |
| 训练配方 / 语义 / 耗时 | `.workbuddy/memory/knowledge/training.md` |
| 机器 / 通道 / 磁盘 / disk_guard | `.workbuddy/memory/knowledge/infra.md` |
| 已定方案 / 已否定假设 | `.workbuddy/memory/knowledge/decisions.md` |
| 省时间（先量再省） | `.workbuddy/memory/knowledge/perf.md` |

**唯一需要常驻的当前焦点**（每次维护只改这一段）：

- 分支：**`feature/auto-learning-v1`（唯一工作分支）**；HEAD 见 `MEMORY.md`。
- **当前焦点（2026-10-10）**：**闭环评测与开环判据的脱钩** —— `place_a2b_left`(0.861) 与
  `move_can_pot`(0.619) 开环都 PASS，**闭环都是 0/5**（`Instruction Type: unseen`、5 回合）。
  下一步：① 跑 `seen` 指令口径对照 + 回合数加到 20 + 开视频看失败形态；
  ② `FAST_LOAD` 真机验证（有卡时，对比 `fast model init` vs 49.30 s）；
  ③ 编译缓存持久化（`TORCHINDUCTOR_CACHE_DIR` 指到 `/root/autodl-tmp` + 开 FX 图缓存）。
- 已定硬约束：**Eval 批处理 = 训练批大小、无任何闸门**（GMean 阈值/atol 一律不得作为运行时判据）；
  多卡只支持 **DDP**（FSDP 明确拒绝）；GPU 开关机 / 删历史产物由用户决定。

## 3. 记忆维护约定（**自动、定期，不问用户**）

1. **分层**：指针（本文件）→ 索引（`MEMORY.md`）→ 知识库（`knowledge/*.md`，按主题）→ 工作日志（`YYYY-MM-DD.md`，按天）。
2. **每轮收尾自动做**：结论/实测数据 → 当天工作日志；**可复用的经验/坑/坐标** → 对应 `knowledge/*.md`；
   只在"当前焦点"变化时改本文件 §2，**不改其它小节**。
3. **禁止**把逐日明细、行号索引、长表格写进本文件（行号会失效，且会把文件撑爆）。
4. **定期整理**（约每周或单文件 >60 KB 时）：`knowledge/*.md` 去重、过期结论标注作废、日志归档到
   `.workbuddy/memory/_archive/`；`MEMORY.md` 只留索引 + 当前状态 + 红线。
5. **判定标准**：新会话只读本文件 + `MEMORY.md` 就能开工；细节按需 Read 知识库，**不预加载**。

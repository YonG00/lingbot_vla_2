# AGENTS.md —— LingBot-VLA-v2 项目导航（**只做指针，不放状态与结论**）

> 本文件只回答三件事：**仓库在哪、怎么协作、去哪读细节**。
> 状态、实测数字、经验一律不写在这里（会过期）；它们放在下表指向的文档里。

## 0. 工程坐标（以当前仓库为准）

- 仓库根：本文件所在目录（git 顶层）；Python 包在 `lingbot-vla-v2/`（package `lingbot-vla-v2`）。
- **本地实验分支：`learn_skill_mix_task`**，由 `Task_Auto_Learning` 创建；生产基线仍为 `Task_Auto_Learning`。
  实验使用独立服务器目录，提交/push/同步须获批；下方生产分支的 reset/push 指令不能用于同步本实验。
  其它分支（`main` / `feature/auto-learning-v1` / `feature/quick-eval-multigpu-v1` / `feature/rocm-adapt`）**不作为开发基线**，不要往里同步或合并。
- **训练机：`ctrl`（AutoDL）**；代码在 `/data/code`；Python：`/data/miniconda3/envs/lingbotvla/bin/python`。
  本机 Python 缺 torchdata / 新版 transformers ⇒ 重依赖测试只能在训练机上跑。AMD/ROCm 路线**不再使用**（`docs/rocm/` 仅作历史资料）。
- 训练机同步（唯一正确方式）：先确认 `git branch --show-current` 是 `Task_Auto_Learning`，再
  `cd /data/code && git fetch -q origin && git reset --hard origin/Task_Auto_Learning`。
- 同步口径：默认只做 **训练机 ↔ GitHub** 两方对齐；交付验证给出 `machine HEAD == origin/Task_Auto_Learning` + 机器测试结果即可。
- 推送：本机 commit + `git push origin Task_Auto_Learning`，push 单独一条命令，不与大任务串。
- 本仓库没有 `.workbuddy/` 记忆目录；上一级目录里的旧记忆**不作依据**。需要沉淀经验时，写进本仓库 `docs/`（见 §2）。

## 1. 铁律

1. **动手前先获批**：默认只读 + 提建议。改代码、推机器、跑训练/扫描、停进程、提交、push、建/删分支、删文件，
   一律先说方案 → 等用户点头 → 再执行；不要把"用户同意的那一步"顺手扩权。
   遇到报错/异常的默认动作 = **只读取证 + 讲清楚 + 给选项**，不是"先修修看"。
   无需请示：读文件、只读取证（日志、`git status/log`、md5）。
2. **测试全绿前不得提交**；区分"本次引入的失败"与"既有失败"，后者要给证据。
3. `git commit` 一律 `-F <文件>`（`-m` 里的反引号会被 shell 替换）。
4. 改代码前先 `grep` 目标位置是否存在；按**函数名/谓词**定位，不靠行号、不做整段字符串替换。
5. `git apply --check` 用**绝对路径**。
6. `pkill -f` / `pgrep -f` 会匹配执行它的那条命令自身 ⇒ 一律用方括号写法：`pkill -f "[r]atio-gpu"`。
7. **长任务一律后台**：机器侧用 `setsid nohup … > log 2>&1 < /dev/null &`，再另起调用轮询日志；
   **同一条调用里不 sleep/tail 轮询，也不对后台作业做"等待"**（只看完成通知或另起一次取证）。
8. **上 GPU 前先跑 CPU 端整链自测**；每次 GPU 上炸出的接口契约类 bug，必须补一条 CPU 契约测试。
9. **有训练在跑时，禁做训练/GPU 测试**（含会 import torch 的 pytest 收集）：只允许 `py_compile`、纯 CPU 单测、只读取证。
10. **GPU 开关机、租卡、删历史产物由用户决定**；用户明说"关机"才关机。
11. 远端脚本一律"本地写文件 → scp → 远端执行"，不要用 heredoc/嵌套引号拼命令。

## 2. 去哪读细节（指针）

| 想知道什么 | 去哪 |
|---|---|
| **AL 系统怎么运转**（数据模型 / 评测 split / 判定口径 / 状态机 / 缓存 / 存档四象限 / 成本模型 / 已知冲突） | 技能 **`lingbot-auto-learning-logic`**（改 AL 代码前先读） |
| **怎么改代码才不出事**（讨论 → 清单 → 逐条执行 → 提交；改动本意四问；验证矩阵） | 技能 **`lingbot-code-changes`** |
| **改动台账**（改了什么 / 为什么 / 是否已上机验证 / 怎么回滚；`AL_*` 开关总表） | `lingbot-vla-v2/docs/auto_learning/AL_PENDING_CHANGES.md` |
| 单技能实验 / 指定学习与 Replay 任务 / 无 PASS 与扫描 / HF-only | `lingbot-vla-v2/docs/auto_learning/skill_mix_task_experiment_zh.md` |
| AL 上手 / 概念 / 启动命令 | `lingbot-vla-v2/README_AUTOLEARNING.md`、`docs/auto_learning/auto_learning_overview_zh.md`、`auto_learning_launch_guide_zh.md` |
| 已知坑与自查表 | `lingbot-vla-v2/docs/auto_learning/auto_learning_pitfalls_zh.md` |
| 文档总索引（按"想做什么"查） | `lingbot-vla-v2/docs/README.md` |
| 工具脚本索引 | `lingbot-vla-v2/tools/README.md` |
| 训练监控面板 | `lingbot-vla-v2/tools/auto_learning_pipeline/al_panel.py`（一键：`al_watch.sh`） |
| AL 启动器 | `lingbot-vla-v2/experiment/robotwin/al_launch.py` |
| AL 配置 | `lingbot-vla-v2/configs/auto_learning/` |
| 磁盘 / 检查点保留 | `lingbot-vla-v2/docs/disk_guard_guide.md` |

> 技能已安装在 `~/.agents/skills/`（`lingbot-auto-learning-logic`、`lingbot-code-changes`），
> 源仓库 `github.com/YonG00/skills`；技能内容有更新时重新拷贝覆盖即可。
> ⚠️ 技能里出现的 AMD / `/models/robotwin-persistent` / `/work_ov` 等路径与 `cpu1` 机器信息属于旧环境，
> **以本文件 §0（ctrl / `/data/code`）为准**；技能里的"逻辑与口径"部分仍然有效。

## 3. 本文件维护约定

1. 只在**分支、机器坐标、铁律、指针表**变化时改本文件；不写状态、数字、逐日流水。
2. 可复用的经验：改动类写进 `AL_PENDING_CHANGES.md`；系统逻辑类写进技能或 `docs/auto_learning/` 对应文档。
3. 新增文档后，同步 `lingbot-vla-v2/docs/README.md` 的索引。

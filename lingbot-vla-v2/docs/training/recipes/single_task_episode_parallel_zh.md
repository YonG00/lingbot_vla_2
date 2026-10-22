# 单任务按回合并行闭环评测

仅有一个评测任务时，可在同一个模型服务上同时运行多个独立仿真回合。模型前向仍按B1串行，仿真进程并行；不是复制多份模型，也不是把不同回合拼成训练batch。默认行为不变。

## 配方参数

```json
{
  "evaluation": {
    "tasks": ["stack_bowls_two"],
    "episodes_per_task": 3,
    "checkpoints": ["final"],
    "shared_model_session": true,
    "max_parallel_tasks": 1,
    "single_task_episode_workers": 3,
    "video_episodes": 0,
    "reuse_training_model": true
  }
}
```

此为evaluation片段，需放入完整训练配方。`single_task_episode_workers`取1..8，缺省1；只有单任务时生效，实际并发不超过回合数；多任务仍按max_parallel_tasks调度。示例中的video_episodes=0关闭录像，可改为要录的前N个回合。复用训练模型时，原来的驻留服务和零次权重重载契约保持不变。独立评测启动时仍只加载一次模型。

每个回合分配独立连接、缓存、场景种子和输出目录。默认种子100000起连续编号；可显式填写 `single_task_episode_seeds: [100000,100001,100002]`。如官方专家检查拒绝指定种子，本回合立即失败，不能自行跳种子，否则可能与另一回合重复。此时应传入串行评测或预检已确认有效的种子列表。不是任何任务的默认前三个整数种子都保证有效。

客户端使用运行时AST适配，只改变起始场景种子、episode序号和语言描述生成数量，并增加拒绝种子检查；不修改外部RoboTwin源码文件。脚本结构若不符合契约立即报错。语言描述仍按原总回合数生成，避免拆成1回合改变指令候选数。模型推理仍使用既有每次请求固定噪声种子和独立连接状态机制。

## 结果与录像目录

- `evaluation_<step>/<task>/episodes/episode_000/` 等：每路job、命令、日志、summary与单回合视频。
- `evaluation_<step>/<task>/summary.json`：统一的任务成功率与按全局回合顺序排列的seeds、episode_results。
- `evaluation_<step>/<task>/results/<task>/episode0_global.mp4` 等：按全局回合编号汇总的视频；关闭录像时不生成。
- 顶层summary.json仍是一项任务一条记录，供训练流水和补课决策读取；中间进度可能包含尚未合并的回合记录，只有complete才视为最终结果。

## 验证状态（2026-10-22）

94项相关纯CPU测试通过：种子/输出隔离、录前N回合、三路真正重叠、只开关一次服务、聚合成功率、坏种子或缺失结果拒绝完成、原多任务和驻留评测回归。真实RoboTwin客户端AST适配编译通过。

实测采用18cm目标训练的step200，stack_bowls_two单任务、3回合、3路、录像0。目录位于该run的 `episode_parallel3_novideo_20261022_224804`。确认只加载一次模型、3连接/3路重叠、B1推理，OIDN错误0、录像0；峰值43218MiB（42.21GiB），总耗时174.71秒，仿真重叠窗口116.60秒。

**功能验证通过，串行一致性尚未确认。** 同种子100000/100001/100002，并行1/3；既有驻留串行0/3，差异为100002。两次除并发外还存在驻留/重新加载及录像条件差异，因此不能把差异归因于并发，也不能宣称数值等价或严格加速倍数。已追加相同HF加载路径、同种子、不录像的串行对照，目录 `episode_serial_novideo_control_20261022_225402`，本次提交时仍在运行。

用户明确要求先push：本次以默认关闭的实验功能提交，并保留这一未决问题；不将1/3算作训练效果改善。证据 `refs/data_selection/single_task_episode_parallel3_novideo_verification.json`。

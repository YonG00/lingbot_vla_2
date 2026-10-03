# 评测步数上限 `_eval_step_limit.yml` 使用说明

> 文件：`RoboTwin-lingbot/task_config/_eval_step_limit.yml`（50 项，task_name → 最大仿真步数）
> 消费方：`RoboTwin-lingbot/envs/_base_task.py:141-148`

---

## 1. 它是什么

**评测时每个 episode 的最大仿真步数**。只在 `eval_mode` 下生效，训练完全不读它。

```python
# envs/_base_task.py:141-148
if self.eval_mode:
    with open(os.path.join(CONFIGS_PATH, "_eval_step_limit.yml"), "r") as f:
        try:
            data = yaml.safe_load(f)
            self.step_lim = data[self.task_name]          # 按任务名取
        except:
            print(f"{self.task_name} not in step limit file, set to 1000")
            self.step_lim = 1000                          # 缺项兜底 = 1000
```

超出 `step_lim` 还没完成任务 ⇒ 该 episode 判**失败**。

## 2. 🔴 为什么它是「对比关键参数」

它**直接改变成功率口径**：

- **调低** ⇒ 本来慢一点也能成功的任务被截断成失败 ⇒ **成功率虚低**；
- **调高** ⇒ 慢任务有更多机会 ⇒ 成功率虚高。

所以**跨 checkpoint 比成功率时，这个文件必须锁定不变**；否则测出来的差异可能只是口径差异。
（同理，评测报告里要写明当时的 `md5`。）

副作用（也是当初调低它的动机）：**失败的 episode 会跑满 `step_lim`** ⇒ 失败越多越慢。
调低 `step_lim` 能明显缩短失败跑的墙钟，代价是成功率口径跟着变。

## 3. 当前值（= 官方原版，2026-10-03 恢复）

50 项全表，最松 `put_bottles_dustbin: 1700` / `open_microwave: 1500`，最紧 `400`（多数任务）。

| 步数 | 任务 |
|---|---|
| 1700 | `put_bottles_dustbin` |
| 1500 | `open_microwave` |
| 1200 | `blocks_ranking_rgb` `blocks_ranking_size` `stack_blocks_three` `stack_bowls_three` |
| 900 | `hanging_mug` `stack_bowls_two` |
| 800 | `handover_block` `place_cans_plasticbox` `stack_blocks_two` |
| 700 | `open_laptop` `place_bread_basket` `place_can_basket` `place_object_basket` `put_object_cabinet` `shake_bottle` `shake_bottle_horizontally` |
| 600 | `dump_bin_bigbin` `handover_mic` `place_dual_shoes` |
| 500 | `place_bread_skillet` `place_empty_cup` `place_burger_fries` `place_shoe` `scan_object` |
| 400 | 其余 24 项（`adjust_bottle` `beat_block_hammer` `click_alarmclock` `click_bell` `grab_roller` `lift_pot` `move_can_pot` `move_playingcard_away` `move_stapler_pad` `pick_diverse_bottles` `pick_dual_bottles` `place_a2b_left` `place_a2b_right` `place_container_plate` `place_fan` `place_mouse_pad` `place_object_scale` `place_object_stand` `place_phone_stand` `move_pillbottle_pad` `press_stapler` `rotate_qrcode` `stamp_seal` `turn_switch`） |

> 我们 A/B 用的 L1 sentinel 4 任务（`lift_pot` / `turn_switch` / `click_alarmclock` / `open_microwave`）
> 分别落在 400 / 400 / 400 / 1500。

## 4. 历史与「口径」⚠️（这段最要紧）

| commit | 时间 | 动作 | md5 |
|---|---|---|---|
| `d21ea1c` | 2026-09-26 16:52 | 随 RoboTwin working tree 引入（**官方原版**） | `171fbdafc05a9981ec39e935f7b0f85e` |
| `2fcf9be` | 2026-10-02 **20:52** | 「调低评测步数上限：**全表 −150**」 | `ef49a9d47c32dc4bd2e13b22cca73d22` |
| 本次 | 2026-10-03 23:4x | **恢复原版**（与 `d21ea1c` **字节一致**） | `171fbdafc05a9981ec39e935f7b0f85e` |

- 恢复动作**等价于 `git revert 2fcf9be`**（`2fcf9be` 之后没有别的 commit 碰过这个文件）。

### 🔴 已产出的结果分属两种口径

| 结果 | 时间 | 当时口径 |
|---|---|---|
| base 冒烟 **25.0%** (1/4)、50k 冒烟 **75.0%** (3/4) | 2026-10-02 **20:19–20:27** | **原版**（早于 20:52） |
| 50k 正式 **91.7%** (11/12) | 2026-10-02 23:22–23:29 | **−150** |
| `phase1_L1` 正式 **0/12** | 2026-10-03 | **−150** |
| 对照组 `vit_frozen` 779 / 1558 **双双 0/12** | 2026-10-03 22:58–23:05 | **−150** |

**⇒ 两条必须记住的结论：**

1. **「冒烟 25% vs 正式 0%」本来就混了口径** —— 冒烟是原版（更松）、正式是 −150（更紧）。
   所以「base 可能优于训练后」这个方向性判断，**证据比原先以为的更弱**。
2. **接下来补 base 的正式 12 回合时，必须与对照目标同口径**。
   若用恢复后的**原版**去测 base，再和 `phase1_L1` 的 **−150** 结果并排，就是**换了个变量**。
   二选一：① 评测时临时把 yml 改回 −150（只改这一个文件，测完恢复）；② 或者把 `phase1_L1` / 对照组**也重测一遍**。

> 我们 A/B 的 L1 sentinel 4 任务在两种口径下的差别：
> `lift_pot` 400→250、`turn_switch` 400→250、`click_alarmclock` 400→250、`place_shoe` 500→350。
> 原版**更宽松** ⇒ 理论上「差一点就能成功」的 rollout 在原版下有可能翻成成功，
> 所以 `0/12` 这个结论本身也带口径依赖，别当成绝对事实。

## 5. 怎么改

直接编辑 yml 即可（无代码改动）。注意：

- **必须覆盖用到的任务名**，否则静默回落到 `1000`（只打一行 `not in step limit file`，不报错）；
- 改完**记下 `md5`** 写进评测报告；
- 想临时调低只跑得快一点：全表统一减一个常数最省事，但**记得这只是「快」，不是「准」**。

## 6. 验证

```bash
cd /data/code
md5sum RoboTwin-lingbot/task_config/_eval_step_limit.yml      # 期望 171fbdafc05a9981ec39e935f7b0f85e
git show d21ea1c:RoboTwin-lingbot/task_config/_eval_step_limit.yml | md5sum   # 应一致 ⇒ 确为原版
grep -c ':' RoboTwin-lingbot/task_config/_eval_step_limit.yml # 50
```

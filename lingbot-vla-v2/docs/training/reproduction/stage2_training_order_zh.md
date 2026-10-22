# 第二阶段训练顺序与更新数

本阶段按下表依次训练，不做闭环评测。共12批、61项训练安排、9150步。所有批次统一10%复习：从第一阶段学过的50个任务中均匀取样，一次复习一个旧任务，每5步切换，排除当前主任务。

同一批内连续训练；一批完成后保存模型，再开始下一批。每批重新建立优化器和学习率计划。

[返回复现说明](three_stage_baseline_zh.md#第二阶段技能验证与任务筛选)

| 批次 | 总顺序 | 任务 | 更新步数 | 复习样本比例 |
|---:|---:|---|---:|---:|
| 1 | 1 | `adjust_bottle` | 200 | 10% |
| 2 | 2 | `place_container_plate` | 200 | 10% |
| 3 | 3 | `place_container_plate` | 100 | 10% |
| 3 | 4 | `place_empty_cup` | 200 | 10% |
| 3 | 5 | `move_can_pot` | 200 | 10% |
| 4 | 6 | `turn_switch` | 200 | 10% |
| 4 | 7 | `move_pillbottle_pad` | 200 | 10% |
| 4 | 8 | `move_stapler_pad` | 200 | 10% |
| 5 | 9 | `move_stapler_pad` | 200 | 10% |
| 5 | 10 | `move_pillbottle_pad` | 200 | 10% |
| 5 | 11 | `turn_switch` | 100 | 10% |
| 6 | 12 | `move_playingcard_away` | 200 | 10% |
| 6 | 13 | `place_phone_stand` | 200 | 10% |
| 6 | 14 | `place_object_scale` | 200 | 10% |
| 6 | 15 | `rotate_qrcode` | 200 | 10% |
| 7 | 16 | `rotate_qrcode` | 100 | 10% |
| 7 | 17 | `move_playingcard_away` | 100 | 10% |
| 7 | 18 | `pick_dual_bottles` | 200 | 10% |
| 7 | 19 | `pick_diverse_bottles` | 200 | 10% |
| 7 | 20 | `grab_roller` | 200 | 10% |
| 7 | 21 | `lift_pot` | 200 | 10% |
| 7 | 22 | `stamp_seal` | 200 | 10% |
| 8 | 23 | `shake_bottle_horizontally` | 150 | 10% |
| 8 | 24 | `pick_dual_bottles` | 100 | 10% |
| 8 | 25 | `pick_diverse_bottles` | 100 | 10% |
| 8 | 26 | `put_bottles_dustbin` | 150 | 10% |
| 8 | 27 | `dump_bin_bigbin` | 150 | 10% |
| 8 | 28 | `place_container_plate` | 100 | 10% |
| 8 | 29 | `stack_bowls_two` | 150 | 10% |
| 8 | 30 | `stack_bowls_three` | 150 | 10% |
| 8 | 31 | `blocks_ranking_size` | 150 | 10% |
| 8 | 32 | `stack_blocks_two` | 150 | 10% |
| 9 | 33 | `blocks_ranking_size` | 100 | 10% |
| 9 | 34 | `move_stapler_pad` | 100 | 10% |
| 9 | 35 | `pick_dual_bottles` | 100 | 10% |
| 9 | 36 | `place_container_plate` | 100 | 10% |
| 9 | 37 | `stack_bowls_two` | 100 | 10% |
| 9 | 38 | `blocks_ranking_rgb` | 150 | 10% |
| 9 | 39 | `place_bread_skillet` | 150 | 10% |
| 9 | 40 | `place_can_basket` | 150 | 10% |
| 9 | 41 | `place_cans_plasticbox` | 150 | 10% |
| 9 | 42 | `place_bread_basket` | 150 | 10% |
| 10 | 43 | `place_can_basket` | 100 | 10% |
| 10 | 44 | `place_mouse_pad` | 200 | 10% |
| 10 | 45 | `pick_dual_bottles` | 300 | 10% |
| 10 | 46 | `stack_bowls_two` | 300 | 10% |
| 10 | 47 | `handover_block` | 150 | 10% |
| 10 | 48 | `place_shoe` | 150 | 10% |
| 10 | 49 | `place_fan` | 150 | 10% |
| 11 | 50 | `handover_block` | 200 | 10% |
| 11 | 51 | `stack_bowls_two` | 200 | 10% |
| 12 | 52 | `beat_block_hammer` | 80 | 10% |
| 12 | 53 | `handover_mic` | 80 | 10% |
| 12 | 54 | `hanging_mug` | 80 | 10% |
| 12 | 55 | `place_a2b_left` | 80 | 10% |
| 12 | 56 | `place_burger_fries` | 80 | 10% |
| 12 | 57 | `place_dual_shoes` | 80 | 10% |
| 12 | 58 | `place_object_stand` | 80 | 10% |
| 12 | 59 | `put_object_cabinet` | 80 | 10% |
| 12 | 60 | `scan_object` | 80 | 10% |
| 12 | 61 | `stack_blocks_three` | 80 | 10% |

默认每2000步保存一次；各批不足2000步，因此均在批末保存。第二阶段结束后，继续用该模型进入第三阶段：先混合训练1000步，再按项目最高成绩表进行五批任务抽检并补训。

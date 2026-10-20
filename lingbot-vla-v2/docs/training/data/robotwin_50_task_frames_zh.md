# RoboTwin 50 类任务轨迹帧数统计

**口径**：`RoboTwin_lerobot_v30` 数据集，50 个任务各 50 个完整回合（共 2,500 回合 / 548,893 帧）；
`总帧数` = 该任务 50 个回合 `length` 之和，`平均帧/条` = 总帧数 ÷ 50。
帧率 15 fps，1 帧 ≈ 0.067 秒（例如 219.6 帧 ≈ 14.6 秒）。
数据只读自 `refs/al_artifacts/al/task_splits_50/manifest.json` 的 `train_frames + val_frames`，与本仓库文档记录的
548,893 / 2,500 完全一致。

## 汇总

| 指标 | 数值 |
|---|---|
| 总帧数 / 总回合 | 548,893 / 2,500 |
| **平均轨迹帧数** | **219.6 帧/条** |
| 中位数（按任务平均后） | 167.4 帧/条 |
| 四分位 | p25 142.1 / p50 167.4 / p75 271.4 |
| 每任务总帧数 | 均值 10,978，中位 8,370，范围 3,855–31,231 |
| 最长 / 最短 | `put_bottles_dustbin` 624.6 vs `click_bell` 77.1（**8.1 倍**） |

分布右偏：均值被长的堆叠/倒空类任务拉高，半数任务其实在 167 帧以下。
训练侧换算：`steps/epoch = train_frames / global_batch_size`，因此同样 50 回合，
`put_bottles_dustbin` 的每 epoch 步数约为 `click_bell` 的 8 倍。

## 明细（按平均帧/条降序）

| # | 任务 | 回合数 | 总帧数 | 平均帧/条 |
|---:|---|---:|---:|---:|
| 1 | `put_bottles_dustbin` | 50 | 31,231 | 624.6 |
| 2 | `open_microwave` | 50 | 24,333 | 486.7 |
| 3 | `stack_blocks_three` | 50 | 23,619 | 472.4 |
| 4 | `stack_bowls_three` | 50 | 23,526 | 470.5 |
| 5 | `blocks_ranking_size` | 50 | 23,098 | 462.0 |
| 6 | `blocks_ranking_rgb` | 50 | 23,041 | 460.8 |
| 7 | `hanging_mug` | 50 | 16,847 | 336.9 |
| 8 | `stack_blocks_two` | 50 | 15,647 | 312.9 |
| 9 | `stack_bowls_two` | 50 | 15,609 | 312.2 |
| 10 | `place_cans_plasticbox` | 50 | 14,375 | 287.5 |
| 11 | `handover_block` | 50 | 14,057 | 281.1 |
| 12 | `shake_bottle_horizontally` | 50 | 13,910 | 278.2 |
| 13 | `put_object_cabinet` | 50 | 13,460 | 269.2 |
| 14 | `place_can_basket` | 50 | 12,595 | 251.9 |
| 15 | `shake_bottle` | 50 | 12,436 | 248.7 |
| 16 | `place_object_basket` | 50 | 12,298 | 246.0 |
| 17 | `dump_bin_bigbin` | 50 | 12,122 | 242.4 |
| 18 | `place_burger_fries` | 50 | 12,046 | 240.9 |
| 19 | `place_bread_basket` | 50 | 11,956 | 239.1 |
| 20 | `place_dual_shoes` | 50 | 11,194 | 223.9 |
| 21 | `handover_mic` | 50 | 11,092 | 221.8 |
| 22 | `open_laptop` | 50 | 10,412 | 208.2 |
| 23 | `place_shoe` | 50 | 8,874 | 177.5 |
| 24 | `place_empty_cup` | 50 | 8,617 | 172.3 |
| 25 | `scan_object` | 50 | 8,463 | 169.3 |
| 26 | `place_bread_skillet` | 50 | 8,277 | 165.5 |
| 27 | `place_container_plate` | 50 | 7,934 | 158.7 |
| 28 | `move_stapler_pad` | 50 | 7,731 | 154.6 |
| 29 | `rotate_qrcode` | 50 | 7,718 | 154.4 |
| 30 | `move_can_pot` | 50 | 7,568 | 151.4 |
| 31 | `place_mouse_pad` | 50 | 7,466 | 149.3 |
| 32 | `place_a2b_left` | 50 | 7,451 | 149.0 |
| 33 | `place_fan` | 50 | 7,358 | 147.2 |
| 34 | `place_a2b_right` | 50 | 7,349 | 147.0 |
| 35 | `move_pillbottle_pad` | 50 | 7,345 | 146.9 |
| 36 | `place_object_scale` | 50 | 7,266 | 145.3 |
| 37 | `stamp_seal` | 50 | 7,263 | 145.3 |
| 38 | `adjust_bottle` | 50 | 7,156 | 143.1 |
| 39 | `place_object_stand` | 50 | 6,952 | 139.0 |
| 40 | `place_phone_stand` | 50 | 6,357 | 127.1 |
| 41 | `pick_dual_bottles` | 50 | 6,129 | 122.6 |
| 42 | `pick_diverse_bottles` | 50 | 6,060 | 121.2 |
| 43 | `press_stapler` | 50 | 5,946 | 118.9 |
| 44 | `move_playingcard_away` | 50 | 5,884 | 117.7 |
| 45 | `beat_block_hammer` | 50 | 5,682 | 113.6 |
| 46 | `lift_pot` | 50 | 5,554 | 111.1 |
| 47 | `turn_switch` | 50 | 4,791 | 95.8 |
| 48 | `grab_roller` | 50 | 4,691 | 93.8 |
| 49 | `click_alarmclock` | 50 | 4,252 | 85.0 |
| 50 | `click_bell` | 50 | 3,855 | 77.1 |

## 复现

```bash
cd lingbot-vla-v2
python - <<'PY'
import json
d = json.load(open('refs/al_artifacts/al/task_splits_50/manifest.json'))
for k, v in sorted(d['tasks'].items(),
                   key=lambda kv: -(kv[1]['train_frames'] + kv[1]['val_frames']) / kv[1]['n_total']):
    total = v['train_frames'] + v['val_frames']
    print(f"{k:<28}{v['n_total']:>4}{total:>8}{total / v['n_total']:>9.1f}")
PY
```

`tools/task_split.py` 生成 manifest 时也会打印同样的逐任务表（`n_train / n_val / train_frames / val_frames`），本文件只是把 50 行按“平均帧/条”重排并补了汇总统计。

> 口径提醒：这里是**数据集的帧数**，不是训练时的样本数；训练按 `action_chunk=50` 划窗后，
> 单条轨迹的可采样起点约为 `max(1, 帧数 - 50 + 1)`，且实际步数由 `train_frames / global_batch_size` 决定。

# A. Unit Loss 对账（独立计算，不依赖曲线外观）

## `training/loss` 原始值（TB events）
- step 501: 0.161427
- step 502: 0.171758
- step 503: 0.174543
- step 504: 0.200772
- step 505: 0.193066
- step 506: 0.266990
- step 507: 0.240388
- step 508: 0.298582
- step 509: 0.275514
- step 510: 0.256375

## `auto_learning/unit_loss`（TB events）
- step 505: 0.180313
- step 510: 0.267570

## 独立计算的两个五步均值
- 501–505: mean = **0.180313**  (n=5)
- 506–510: mean = **0.267570**  (n=5)

## 与 TB 的 unit_loss 差异
- @505: unit_loss=0.180313 vs 独立均值=0.180313 ⇒ 差 -0.000000 (-0.000%)  ✅ 一致
- @510: unit_loss=0.267570 vs 独立均值=0.267570 ⇒ 差 +0.000000 (+0.000%)  ✅ 一致

> 说明：`unit_loss` 由 `TrainResult.loss = mean(per_step_losses)`（`real/hook.py`）给出；若与上面的独立均值不一致，通常意味着 unit 内的 step 与 `training/loss` 的 step 口径不同（如首步/边界步）。

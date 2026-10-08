# C. Hardness 扫描对账 & Checkpoint 清单

## Hardness 扫描（事件流，权威）
- 第 1 次: task=`place_container_plate` | seconds=**400.634** | is_first=True | n_scanned=None | n_total=6346 | trajs=0 | GPU/CPU/IO 未分别埋点（仅 wall-time）
- 第 2 次: task=`turn_switch` | seconds=**246.99** | is_first=False | n_scanned=None | n_total=3834 | trajs=0 | GPU/CPU/IO 未分别埋点（仅 wall-time）

## TensorBoard 是否记录（含第二次）
- `auto_learning/hardness_scan_seconds`: [(500, 400.634), (505, 246.99)]
- `auto_learning/hardness_scan_seconds_warmup`: [(500, 400.634)]
- `auto_learning/hardness_scan_seconds_steady`: [(505, 246.99)]
- `auto_learning/hardness_scan_samples`: [(500, 2204.0), (505, 1357.0)]
- `auto_learning/hardness_scan_trajs`: [(500, 14.0), (505, 14.0)]

## 日志时间戳（扫描起止）
-     "async_hf_max_pending": 1,
- 10/08/2026 16:44:48 - INFO - __main__ - [auto_learning] eval place_container_plate/train n_ids=4: mse=0.047930 nmse=0.1605 (4轨迹/16chunk, 10.1s)
- 10/08/2026 16:44:58 - INFO - __main__ - [auto_learning] eval place_container_plate/val n_ids=4: mse=0.047138 nmse=0.1578 (4轨迹/16chunk, 10.8s)
- 10/08/2026 16:51:39 - INFO - __main__ - [auto_learning] 第一个 learning unit 的 TrainRequest 已发布
- Step: 501/510 [07:24<1:06:41, 444.57s/it]10/08/2026 16:51:48 - INFO - __main__ - Step 501/109823, Epoch 1, Loss 0.1614, VLA_Loss 0.1514, Depth_Loss 0.9000, Future_Depth_Loss 1.1991, FutureVi
- Step: 502/510 [07:31<24:57, 187.24s/it]  10/08/2026 16:51:56 - INFO - __main__ - Step 502/109823, Epoch 1, Loss 0.1718, VLA_Loss 0.1637, Depth_Loss 0.4651, Future_Depth_Loss 1.1416, FutureVi
- Step: 503/510 [07:38<12:14, 104.98s/it]10/08/2026 16:52:03 - INFO - __main__ - Step 503/109823, Epoch 1, Loss 0.1745, VLA_Loss 0.1666, Depth_Loss 0.5988, Future_Depth_Loss 0.9584, FutureVide
- Step: 504/510 [07:45<06:38, 66.36s/it] 10/08/2026 16:52:10 - INFO - __main__ - Step 504/109823, Epoch 1, Loss 0.2008, VLA_Loss 0.1936, Depth_Loss 0.5134, Future_Depth_Loss 0.8620, FutureVide
- 10/08/2026 16:52:28 - INFO - __main__ - [auto_learning] eval place_container_plate/train n_ids=4: mse=0.046643 nmse=0.1562 (4轨迹/16chunk, 11.6s)
- 10/08/2026 16:52:38 - INFO - __main__ - [auto_learning] eval place_container_plate/val n_ids=4: mse=0.046240 nmse=0.1548 (4轨迹/16chunk, 10.0s)
- Step: 505/510 [08:14<04:23, 52.75s/it]10/08/2026 16:52:38 - INFO - __main__ - Step 505/109823, Epoch 1, Loss 0.1931, VLA_Loss 0.1865, Depth_Loss 0.5693, Future_Depth_Loss 0.6429, FutureVideo
- 10/08/2026 16:52:43 - INFO - __main__ - [auto_learning] eval turn_switch/train n_ids=4: mse=0.488107 nmse=1.6657 (4轨迹/8chunk, 5.1s)
- 10/08/2026 16:52:49 - INFO - __main__ - [auto_learning] eval turn_switch/val n_ids=4: mse=0.539926 nmse=1.8425 (4轨迹/8chunk, 5.2s)
- Step: 506/510 [12:40<08:21, 125.34s/it]10/08/2026 16:57:05 - INFO - __main__ - Step 506/109823, Epoch 1, Loss 0.2670, VLA_Loss 0.2568, Depth_Loss 0.9989, Future_Depth_Loss 1.1399, FutureVide
- Step: 507/510 [12:47<04:20, 86.71s/it] 10/08/2026 16:57:12 - INFO - __main__ - Step 507/109823, Epoch 1, Loss 0.2404, VLA_Loss 0.2311, Depth_Loss 0.9643, Future_Depth_Loss 0.9533, FutureVide
- Step: 508/510 [12:55<02:02, 61.42s/it]10/08/2026 16:57:19 - INFO - __main__ - Step 508/109823, Epoch 1, Loss 0.2986, VLA_Loss 0.2897, Depth_Loss 0.5807, Future_Depth_Loss 1.2328, FutureVideo
- Step: 509/510 [13:02<00:44, 44.53s/it]10/08/2026 16:57:26 - INFO - __main__ - Step 509/109823, Epoch 1, Loss 0.2755, VLA_Loss 0.2668, Depth_Loss 0.7114, Future_Depth_Loss 1.0468, FutureVideo
- 10/08/2026 16:57:40 - INFO - __main__ - [auto_learning] eval turn_switch/train n_ids=4: mse=0.487017 nmse=1.6620 (4轨迹/8chunk, 5.9s)
- 10/08/2026 16:57:45 - INFO - __main__ - [auto_learning] eval turn_switch/val n_ids=4: mse=0.538658 nmse=1.8382 (4轨迹/8chunk, 5.7s)
- [INFO][lingbotvla.utils.checkpoint_guard:55] 10/08/2026 16:57:45 >> [DiskCheck] (收尾存档前 step 510) 尚无历史占用参考; 放行 (disk_avail_now=203.2GB)
- 10/08/2026 16:57:46 - INFO - __main__ - [ckpt] dcp_save_mode=final_only：收尾写完整 DCP（估 ≈72G，可用 203G）
- 10/08/2026 16:59:50 - INFO - __main__ - Distributed checkpoint saved at /data/outputs/al_smoke_gbs4_4task_tb10/checkpoints/global_step_510 successfully!
- 10/08/2026 16:59:50 - INFO - __main__ - [async_hf] saving HF checkpoint for /data/outputs/al_smoke_gbs4_4task_tb10/checkpoints/global_step_510

## global_step_510 文件清单（不含权重内容）
- `model/.metadata` — 754.9 KiB
- `model/__0_0.distcp` — 760.2 MiB **(仅清单，未打包)**
- `model/__0_1.distcp` — 760.2 MiB **(仅清单，未打包)**
- `model/__0_10.distcp` — 760.3 MiB **(仅清单，未打包)**
- `model/__0_11.distcp` — 760.3 MiB **(仅清单，未打包)**
- `model/__0_12.distcp` — 760.3 MiB **(仅清单，未打包)**
- `model/__0_13.distcp` — 760.3 MiB **(仅清单，未打包)**
- `model/__0_14.distcp` — 760.2 MiB **(仅清单，未打包)**
- `model/__0_15.distcp` — 760.2 MiB **(仅清单，未打包)**
- `model/__0_2.distcp` — 760.3 MiB **(仅清单，未打包)**
- `model/__0_3.distcp` — 760.3 MiB **(仅清单，未打包)**
- `model/__0_4.distcp` — 760.3 MiB **(仅清单，未打包)**
- `model/__0_5.distcp` — 760.3 MiB **(仅清单，未打包)**
- `model/__0_6.distcp` — 760.2 MiB **(仅清单，未打包)**
- `model/__0_7.distcp` — 760.2 MiB **(仅清单，未打包)**
- `model/__0_8.distcp` — 760.2 MiB **(仅清单，未打包)**
- `model/__0_9.distcp` — 760.2 MiB **(仅清单，未打包)**
- `.hf_ckpt.tmp.510.7061/model-00001-of-00006.safetensors` — 4718.3 MiB **(仅清单，未打包)**
- `.hf_ckpt.tmp.510.7061/model-00002-of-00006.safetensors` — 4715.3 MiB **(仅清单，未打包)**
- `.hf_ckpt.tmp.510.7061/model-00003-of-00006.safetensors` — 4715.3 MiB **(仅清单，未打包)**
- `.hf_ckpt.tmp.510.7061/model-00004-of-00006.safetensors` — 1633.4 MiB **(仅清单，未打包)**
- `extra_state/extra_state_rank_0.pt` — 0.3 MiB **(仅清单，未打包)**
- `optimizer/.metadata` — 1091.5 KiB
- `optimizer/__0_0.distcp` — 757.1 MiB **(仅清单，未打包)**
- `optimizer/__0_1.distcp` — 757.2 MiB **(仅清单，未打包)**
- `optimizer/__0_10.distcp` — 757.3 MiB **(仅清单，未打包)**
- `optimizer/__0_11.distcp` — 757.3 MiB **(仅清单，未打包)**
- `optimizer/__0_12.distcp` — 757.2 MiB **(仅清单，未打包)**
- `optimizer/__0_13.distcp` — 757.2 MiB **(仅清单，未打包)**
- `optimizer/__0_14.distcp` — 757.3 MiB **(仅清单，未打包)**
- `optimizer/__0_15.distcp` — 757.3 MiB **(仅清单，未打包)**
- `optimizer/__0_2.distcp` — 757.3 MiB **(仅清单，未打包)**
- `optimizer/__0_3.distcp` — 757.2 MiB **(仅清单，未打包)**
- `optimizer/__0_4.distcp` — 757.2 MiB **(仅清单，未打包)**
- `optimizer/__0_5.distcp` — 757.2 MiB **(仅清单，未打包)**
- `optimizer/__0_6.distcp` — 757.3 MiB **(仅清单，未打包)**
- `optimizer/__0_7.distcp` — 757.3 MiB **(仅清单，未打包)**
- `optimizer/__0_8.distcp` — 757.3 MiB **(仅清单，未打包)**
- `optimizer/__0_9.distcp` — 757.3 MiB **(仅清单，未打包)**
- 合计 39.12 GiB
- `hf_ckpt/` 存在: False | index.json: False | 分片数: 0
- ⚠️ 未完成的 HF 临时目录: ['/data/outputs/al_smoke_gbs4_4task_tb10/checkpoints/global_step_510/.hf_ckpt.tmp.510.7061']

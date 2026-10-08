# 远端代码状态（对账用，务必看这条）

- 远端 `/data/code` 的 **git 指针 = `1f58d3a`**（按既定规矩**未对齐**到本地/远端 push 的最新提交）
- **实际文件内容等价于本地与 origin 的 `968789f`**（逐文件 md5 已核对一致）：
- `lingbotvla/auto_learning/orchestration/scheduler.py` md5=$(md5sum /data/code/lingbot-vla-v2/lingbotvla/auto_learning/orchestration/scheduler.py | cut -d  -f1)
- `lingbotvla/auto_learning/real/build.py` md5=$(md5sum /data/code/lingbot-vla-v2/lingbotvla/auto_learning/real/build.py | cut -d  -f1)
- `tasks/vla/train_lingbotvla.py` md5=$(md5sum /data/code/lingbot-vla-v2/tasks/vla/train_lingbotvla.py | cut -d  -f1)
- `lingbotvla/utils/tb_task_loss.py` md5=$(md5sum /data/code/lingbot-vla-v2/lingbotvla/utils/tb_task_loss.py | cut -d  -f1)
- `tests/test_al_tb_alignment_extra.py` md5=$(md5sum /data/code/lingbot-vla-v2/tests/test_al_tb_alignment_extra.py | cut -d  -f1)
- `tests/test_detailed_loss_aggregation.py` md5=$(md5sum /data/code/lingbot-vla-v2/tests/test_detailed_loss_aggregation.py | cut -d  -f1)
- `tests/test_hardness_scan_timing.py` md5=$(md5sum /data/code/lingbot-vla-v2/tests/test_hardness_scan_timing.py | cut -d  -f1)
- `configs/auto_learning/smoke_gbs4_4task_tb5.yaml` md5=$(md5sum /data/code/lingbot-vla-v2/configs/auto_learning/smoke_gbs4_4task_tb5.yaml | cut -d  -f1)

> 因此审计时请以 **968789f** 作为代码基准；`1f58d3a` 只是远端指针。
> TB 对齐补丁与另两处修复分别对应本地提交：4b26f91（TB 对齐）/ 5721886（审查方测试）/ 8acaacb（smoke 配置）/ b3b0b44（detailed_loss 跨 GBS 聚合）/ a91d1d5（Hardness 计时）/ 968789f（文档）。

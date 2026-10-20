# 顺序续训：plate100 → cup200 → pot200

2026-10-20用户授权从`place_container_plate_from_adjust_bf16_200_m56_add200_20261020/checkpoints/global_step_200/hf_ckpt`
开始，先place_container_plate补100步，再place_empty_cup200步，再move_can_pot200步，
本轮累计500步、250/500各存BF16 HF，只评最终500的三个任务各3回合。

- 配方：`configs/training_recipes/sequential_plate100_cup200_pot200.json`。
- 输出：`/data/outputs/sequential_plate100_cup200_pot200_from_plate200_bf16_m56_20261020`。
- 来源为最新plate单任务200，不是原adjust_bottle200，也不继承已删除的三任务400。
- 新优化器与连续500步cosine调度；阶段之间不重载权重、不重置优化器/LR，不额外保存100/300步。
- `sequential`模式显式task_order：1–100 plate、101–300 cup、301–500 pot；没有交错训练/复习。
- 每任务50条，全150条24119帧；BF16参数/计算/HF，micro=GBS56/GAS1，GC/compile开启，workers8。
- 本轮用户明确将保存节奏设为每250步；保存250处于cup阶段，500为最终pot阶段之后。

用户随后要求先按旧流程启动，不让内存模型交接开发阻塞训练：
本轮训练完成并验证两份HF后，重新加载最终500到FP32，使用同一服务完成三个任务各3回合，结束服务。
**HF仍BF16约11.88GiB/份；FP32仅指推理时内存dtype，不另存24GiB FP32 HF。**
BF16闭环也可作为后续可选验证，但本轮未切换；训练模型保留并就地评测也尚未启用。

状态：`<run>_pipeline/status.json`；训练日志与6007事件在run下；结果`<run>/evaluation_500/<task>/summary.json`。
没有自动关机，不删除其他存档。

## 启动与契约

```bash
python experiment/task_recipes/run.py --recipe configs/training_recipes/sequential_plate100_cup200_pot200.json
python experiment/task_recipes/run.py --recipe configs/training_recipes/sequential_plate100_cup200_pot200.json --execute
```

先CPU整链→独立三步GPU探针→正式500步。真实框架CPU检查原先写死GAS1，导致此前GAS2测试失败，
已将expected_gas显式参数贯通且旧默认1不变；修复后的GAS2 CPU装配与本次56/GAS1真实150回合检查均通过。
阶段边界及保存位置纳入`experiment.task_recipes.test_contracts`，本地13项通过。

不变量：来源精确、100/200/200顺序、500全局预算、250/500两份BF16 HF、只评500；
症状：冻结计划/数据索引/虚拟epoch/梯度精度/HF断言失败；
回滚：不运行新配方，旧来源模型独立保留；不可通过改正在运行的recipe或step_plan热修复。

实际启动验收：CPU检查、三步GPU探针通过，正式已更新7/500步；6007可选本轮run。最终250/500与9回合闭环尚未完成。

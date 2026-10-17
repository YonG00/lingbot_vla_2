# AL固定步数与数据epoch边界

故障：stand训练数据5561帧、GBS24，一轮231步。旧内层循环在231结束，随后 `flush_partial_unit()` 会把200后仅31步的unit提前结算和测评；不是100步周期，也不是HF存档触发。

修复：AL开启、明确max_steps且save_epochs=0时，内层循环以剩余optimizer步数为预算，持续消费无限AL采样流。迭代器只在unit切换时重建；不在数据长度边界flush。普通训练、没有max_steps或显式按epoch存档仍走原路径。真正结束/max_steps/STOP时保留最终flush，不取消安全存档记账。

测试意图：AST提取训练器实际预算与range，覆盖普通训练/固定AL/恢复偏移/显式epoch保存。真实CPU hook+sampler从0跑到255，证明231时仍为第三个unit的第31步，只有100/200正常结算；255最终flush后样本数510，安全边界恢复。通过替身阻断torch导入，不使用GPU或模型。

```bash
python3 -m unittest discover -s tests/al_epoch_schedule -p test_loop_budget.py -v
```

当前已经在跑的训练进程继续旧代码；不热改、不重启。后续run使用修复，真实GPU跨数据epoch验收尚待下一次训练。不能把本次CPU通过表述为当前run不再出现额外测评。

本地与服务器隔离目录的2项循环测试通过，连同面板及HF策略共40项纯CPU通过；汇总日志见 [实机只读与CPU结果](../ee_fk/evidence/panel-singleline-2026-10-17/live_check.log.txt)。测试在整个hook/scheduler执行期阻断torch，包括延迟导入，而不只是模块初始化时阻断。

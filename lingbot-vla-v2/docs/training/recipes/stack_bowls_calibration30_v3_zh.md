# 夹碗阶段均匀校准V3（约30%）

用户确认：保留全部抓碗的基础校准，而不是仅20次远抓；通过片段内均匀选点减少加权帧，同时补远定位和第二放置高度带。

每次抓碗阶段配额：aim13、align6、descend最多10、settle最多6、close8；包含各阶段首尾，内部均匀取点，短段不复制补帧。100次抓碗全部覆盖。保留V2中的172帧远定位、251帧第二放置3–8厘米高度带。合并去重4635/15609帧=29.69%，允许目标30%±2个百分点，越界则拒绝继续。

只稀疏标记loss权重；不删原始轨迹，不改连续50步动作块，不重排未来视频/深度标签；命中3倍、未命中1倍，按原样本数（FM原有效元素数）归约。新策略显式替换同任务旧70%策略，不与旧清单再次取并集。

高度/距离仍为示范TCP参考点的运动学代理，不是真实物体净空或碰撞证明。完整阈值见 configs/data_selection/stack_bowls_two_calibration30_v3.json；选择器 tools/select_bowl_calibration.py。规则、事件和完整清单见 refs/data_selection/stack_bowls_two_calibration30_v3/。

本轮用户明确来源为20261022_1209_round_3_recovery的step160，重新训练200步；并非继续上一轮加权200。GBS56/GAS1/BF16，10%单任务复习每5步轮换；100/200保存。最终200驻留评测夹碗3回合，全部录像；无补课或自动关机。配方 configs/training_recipes/stack_bowls200_from_recovery3_s160_calibration30_weight3.json。

2026-10-22启动：run /data/outputs/20261022_2006_bowls200_from_recovery3_s160_bf16_m56_replay10pct_calibration30_weight3。已授权仅删除上一轮19:09 run的step100，保留该run最终200和本轮来源160；释放11.89GiB，空闲30.89GiB，安全预算29.13GiB。

验证：本地及服务器37项筛选/重要性/损失CPU测试、63项配方测试通过；真实1250轨迹250626帧装配通过，首批15/56样本权重3。GPU探针与正式进度见后续流水，尚不记200步完成。

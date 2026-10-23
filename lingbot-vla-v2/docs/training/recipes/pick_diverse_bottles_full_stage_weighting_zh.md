# pick_diverse_bottles 完整轨迹分阶段与加权

2026-10-23：数据与配置准备完成，未启动本任务训练。处理独立离线副本，没有覆盖运行中的1940训练代码。

## 数据与阶段

50条示范，原始episode1650–1699，对应原始录像/HDF5 episode0–49，共6060帧；逐条核对全部50条HDF5关节状态匹配。完整双臂覆盖、不抽帧、100次抓取均通过判定。

专家任务也是双手抓取→抬高10cm→持瓶到目标位置，末尾不松爪。沿用固定双瓶的运动边界判据，但每条示范的瓶子型号取自scene_info.json，左右手分别绑定真实对象，不能写死base13/base16。17个子阶段每手最大占比约9.39%，无超过10%的子阶段；这些是FK/夹爪命令代理标签，不是接触传感器真值。

## 配置位置与倍率

`refs/data_selection/pick_diverse_bottles_full_stage20_v1/`：

- `segments.json`：完整轨迹阶段与逐片段对象标签。
- `stage_loss_weights.json`：可调倍率，`overrides.fm.left/right`、`overrides.visual.left/right`均展开17子阶段。
- `weight_config.json`：带SHA校验的训练入口。
- `summary.json`：各阶段/各倍率帧数与未来端点有效性统计。
- `object_identities.json`：逐示范真实瓶子型号。
- `validation.json`、`visual_qc.jpg`：验证范围与三示范24画面抽检。

|阶段|动作/视觉倍率|左帧数|右帧数|
|---|---:|---:|---:|
|最后前伸|20|476|483|
|开爪稳定|20|309|310|
|闭爪|20|550|550|
|精细对准|2|300|302|
|提起（含闭爪保持、初始提离和继续抬起）|2|1116|1093|
|其余准备、伸手、搬运、终点保持|1|3309|3322|

20倍区域左右手去重1353/6060帧，22.33%。动作按自身未来时刻/手臂取权重；视觉按自身目标时间取左右手最大值；按权重和归一化，动作padding不监督。H50时2450个锚点未来图像端点越界，对应未来分支权重0。

## 复用入口

```bash
python tools/label_dual_bottle_stages.py \
  --task pick_diverse_bottles \
  --dataset-root DATASET \
  --object-map refs/data_selection/pick_diverse_bottles_full_stage20_v1/object_identities.json \
  --critical-weight 20 \
  --output-dir NEW_OUTPUT
```

旧命令不传task仍默认pick_dual_bottles，关键倍率默认5，避免改变原有产物语义。对象映射须与当前数据episode坐标一致，数据换版后应重新核对。

新倍率需复制权重文件后通过snapshot_stage_policy.py --weights生成新快照，不能热改运行中的冻结配置。未来训练可将LINGBOT_SAMPLE_WEIGHT_CONFIG指向本任务weight_config.json，既有冻结注册表机制可保留其他任务重要性。尚未执行真实训练DataLoader/model/GPU验证，不把离线标签通过当成训练成功。

## 验证

4项新增CPU测试覆盖随机对象、缺失对象拒绝、双手全帧覆盖、旧任务兼容；加13项既有完整阶段测试均通过。三条不同瓶子示范各8张画面抽检，与伸手、对准、闭爪、提起搬运和终点持瓶顺序一致；未逐帧验证真实接触。当前只生成标注和权重，不改变采样概率，也不启动训练。

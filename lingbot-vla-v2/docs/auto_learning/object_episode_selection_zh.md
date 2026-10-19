# 指定目标物的episode筛选与审核

`tools/select_object_episodes.py` 仅做CPU数据审计，不改原始数据、不建立训练dataset、不调用GPU或接入采样器。当前默认词表支持 `--target mouse`（mouse/mice/鼠标及047_mouse）；其他目标以目标词召回，并通过 `--keyword`、`--object-id` 扩展。不同型号不排除。

## 使用

```bash
python tools/select_object_episodes.py \
  --dataset-root <LeRobot数据根目录> \
  --target mouse --primary-task place_mouse_pad \
  --embodiment-config <实际机器人config.yml> \
  --output-dir artifacts/data_audits/mouse/<新审计批次> \
  --extract-keyframes
```

长扫描须后台运行，独立调用读取日志。`--extract-keyframes` 可省略；启用时需CPU ffmpeg。每条关键词候选提取cam_high的8个均匀帧合成总览图，不复制整段视频。图旁记录精确episode帧号，不能以总览图替代完整动作审核。

## 召回与索引

全库核对 `info.total_episodes/total_frames`、episode唯一性与manifest归属。只读解析环境源码AST中的物体字符串，补充可能包含目标物的任务；源码仅表示场景可能包含，不证明该episode操作了目标物。`--candidate-task` 可显式补充可能来源。关键词命中与可能来源均进入候选，主任务进入独立清单，不混入辅助。

只读取主任务和候选的状态列，核对完整帧、episode归属、绝对index、时间戳/FPS及task_index与原指令。复用既有 `scan_episode` 输出闭爪/腕部运动候选，独立记录；不改变原窗口算法。夹爪打开事件也仅作控制信号提示，柜子任务的事件可能属于开抽屉而不是抓鼠标。接近/搬运/放置的目标物阶段需要审核者确认，首版不自动推断。

审核坐标是episode内frame_index；绝对range右端开区间；事件window左右端均包含。不存在可持久化的local_idx；将来混训必须从联合白名单重新建立resolver。当前不改变动作chunk或padding。

视频来自v3元数据的camera/chunk/file/from_timestamp/to_timestamp，绝不把dataset绝对index当视频时间。HTML初始跳到该episode起点，到片段终点暂停；未提供偏移时不猜。HTML引用服务器视频绝对路径，在本机打开时图片可查看，播放原视频须有对应挂载/访问路径。

## 输出与审核

- `inventory.jsonl`：全部episode召回情况，便于追查漏检。
- `primary_episodes.json`：主任务单独清单。
- `candidates.jsonl`：辅助候选、指令、控制事件、视频定位及质量错误。
- `review.html`、`episode_*.jpg`：审核页面与关键词候选总览图。
- `source_identity.json`、`summary.json`、`report.md`：来源、统计、指纹与边界。
- `review_template.json`：全部候选初始为pending，审核者填写证据与决定。
- `approved.json`：无明确审核时为空，且标记 `training_ready=false`；本阶段尚无混合训练接口。

审核模板填入 `reviewer/evidence/manipulated_target_verified/demonstration_success_verified`；只有数据校验通过、对象操作与示范成功均明确核验为true时才能设为approved。不确定保留pending，非目标操作或失败示范设为rejected。阶段窗口可按 `approach/grasp_lift/transport_place` 填入成对episode帧号；留空表示未标阶段，不能擅自拿控制事件作为已核验窗口。

复用同一参数，追加 `--review <填写后的模板>` 并改用新的输出目录，生成批准清单。审核指纹不一致、重复或未知episode、窗口越界、质量不合格均拒绝。运行不会自动读取旧输出，也不覆盖已有审计目录。

指纹包含元数据/任务表/manifest/相关环境源码/机器人配置的内容SHA、工具源码SHA与控制事件设置；大Parquet和候选引用视频记录相对路径、大小、mtime_ns，**不是全量视频内容SHA**。文件系统保留mtime/size的内容篡改不在检测范围；数据移动或规则变动需重新核验。

## 验证与限制

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/test_select_object_episodes.py tests/test_target_change_window.py
```

当前工具做保守召回，不提供视觉分类器；未召回不能解释为确认不含目标物。关键词中目标可能只是参照物，不能自动批准。没有保存的物体ID/示范成功标签时须看视频；不会把专家数据来源当作逐episode成功证据。CPU质量校验不等于语义和示范质量校验。采样配额、原指令保留、混合训练与闭环验证属于后续另行确认的工作。

## 目录与Git边界

- `tools/select_object_episodes.py`、`tests/test_select_object_episodes.py`：实现与CPU契约。
- `docs/auto_learning/`：设计、用法、审核结论与索引，只链接生成产物。
- `artifacts/data_audits/<目标物>/<批次>/`：候选、指纹、HTML、图片及运行输出；被Git忽略，不当成版本化配置。
- `refs/data_manifests/<目标物>/<批次>/`：审核清单、审核决定和必要的来源记录；人工核对后版本化，JSON内容与审计输出保持一致。

脚本允许显式指定独立输出目录；不会自动把生成产物提升为refs清单。复制审核结果到refs时须核对字节/哈希、来源指纹和审核边界。本轮没有修改全仓历史目录，也没有改变训练入口。

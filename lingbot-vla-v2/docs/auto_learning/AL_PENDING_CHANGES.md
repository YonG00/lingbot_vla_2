# 复现main改动台账

## 部分验证

2026-10-25：基于 official_adjust_bottle_control 整理 main，迁入 episode_parallel/model_benchmark 及其评测依赖，删除历史专项实验文档、ROCm目录和临时关机脚本。未迁入样本加权、TCP loss或训练代码修改。

- 不变量：lingbotvla、tasks、三阶段配方保持原基线字节；模型身份锁保持不变。评测默认3回合3路不录像，完整任务/config才允许续跑跳过。
- 出错症状：源码锁校验失败、续跑manifest不匹配、种子重复或汇总state非complete。
- 快速回滚：切回 official_adjust_bottle_control；原main保存在 archive/main-before-reproduction-20261025。
- 验证入口：experiment/reproduction/test_contracts.py、experiment/official_control/test_model_benchmark.py、test_episode_parallel.py、test_inline_seed_search.py；GPU全链未验证。
- 依赖边界：保留训练核心可选历史模块防止导入断裂；不宣称已完成精确历史重放。完整历史差异见 docs/training/records/recovery160_origin_zh.md。

## 本次CPU验证

123项测试通过：三阶段与迁入评测55项、任务配方框架68项。冻结源码锁383文件通过；核心训练文件与 origin/official_adjust_bottle_control 无差异。CLI预览确认100项300回合默认3路不录像，以及单任务4回合4路参数。未连接GPU或启动实际训练/评测。

## 2026-10-25 复现文档整体评测流程

将第三阶段抽检任务的录像检查与最终模型benchmark分开说明；在复现正文新增配置生成、预览启动、结果位置及续跑命令。仅文档变更，训练和评测实现不变。

- 不变量：整体评测读取final_model.json，50任务×2设置×3回合，3路、不录像，不改变模型。
- 出错症状：预览不是100项300回合，或配置中的模型不是最终模型路径。
- 回滚：撤回本次文档提交，不影响已运行的训练和评测。
- 验证入口：提取文档配置生成代码，用临时训练记录生成配置，调用model_benchmark.py预览；检查Markdown本地链接与shell语法。

## 2026-10-25 提交包上机验证发现的录像缺失

- 原因：Git快照中的RoboTwin客户端未携带服务器对照目录已有的全局录像参数传递和编码器生命周期，仿真虽完成却被missing or unexpected video拦截。补入该客户端衔接；训练实现和提交权重不变。
- 不变量：无录像时不创建编码器；录像在策略reset前启动，正常或异常结束均关闭；仅全局观测相机用于录像，不加入模型输入。
- 症状：回合执行成功但无episode*_global.mp4，汇总state=incomplete。
- 回滚：撤回本条客户端改动；不录像路径仍可运行，但不得宣称录像通过。
- 验证：新增test_recording_client_contract的4项真实代码块CPU契约，相关总127项通过；服务器完成micro56、GAS1、BF16真实2步及1708张量HF导出，新权重通过eval.sh加载完成3回合并行带视频，三个视频经ffprobe检查，OIDN错误为0。
- 同时修正test_template_has_no_server_paths：检查原始模板硬编码而非运行时合法路径，避免服务器/data和/root目录触发误报；更新对应测试文件源码锁。
- 边界：完整三阶段未执行；当前容量约148GiB，小于完整流程290.40GiB存档预算，正式preflight按预期拒绝。不能把短测视为完整训练或成绩复现。

补充验证：2任务×clean/randomized各1回合最小benchmark完整结束，4项error/pending均为0，共用一个服务；完成项续跑时通过拒绝再次加载模型的回调检查。三个视频完整解码通过。

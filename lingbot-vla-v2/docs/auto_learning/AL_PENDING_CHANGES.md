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

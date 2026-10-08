# 测试产物清理记录（2026-10-08 / 09 跨零点执行）

## 一、删除的三个一次性测试目录（用户批准）

| 目录（绝对路径） | 大小 | 内容核对 |
|---|---:|---|
| `/data/outputs/al_dcp_retention_test_1008_2242` | **48G** | `checkpoints/global_step_510` + `global_step_515`（各 24G，**测试用 DCP**）；`auto_learning_events.jsonl`、`lingbotvla_cli.yaml`、TB events |
| `/data/outputs/al_hf_direct_acceptance_1008_2255` | **48G** | `hf_milestones/global_step_501/hf_ckpt`（**测试导出** 6 分片）+ `checkpoints/global_step_502`（24G 测试 DCP）+ 事件/TB/配置 |
| `/data/outputs/al_smoke_gbs4_4task_tb10` | **40G** | `checkpoints/global_step_510`（40G 测试 DCP）+ `auto_learning_events.jsonl`、`runs/`（TB）、`tb.log`、`vram.csv`、`lingbotvla_cli.yaml` |

## 二、删除前必须完成的核对（全部通过）

1. **轻量证据已提取并保存**：`stage3/cleanup_evidence_2026-10-08/`（1.6 MB，随本报告一并入库 ⇒ 实例释放也不丢）：
   - `00_inventory.md`：三目录 `ls -la` 与顶层结构
   - 各目录：`auto_learning_events.jsonl`（AL 决策事件）、`lingbotvla_cli.yaml`（启动配置）、TB 事件文件、`tb_files.txt`、`checkpoints_inventory.txt`、`unique_scan.txt`
   - `retention_and_ckpt_records.txt`：A 运行的 **`[ckpt-retention]` DCP 保留记录**与存档事件
   - `hf_acceptance_report.log`：**HF 直出验收完整报告**（6 分片 / 146.8 s / RSS 29.4 GiB / 1708 张量 / 最大绝对差 0.000e+00 / 无 tmp 残留）
   - `dcp_test.sh`、`hf_accept.sh`、`hf_accept_cmd.sh`、`dcp_test_final.txt`：启动脚本与最终证据文本
2. **完整路径与内容核对**：三目录均在 `/data/outputs/` 下，均为 2026-10-08 一次性测试产物；
   **符号链接 0 个**；扫描 `norm_stats*` / `*.py` / `*.patch` **均无命中**；
   大文件仅为测试期 `.distcp` 分片与测试导出 `.safetensors` ⇒ **不含唯一原始模型、正式训练成果或源码**
3. **删除方式**：仅用**精确绝对路径** `rm -rf <dir>`，**未使用任何通配符**；删除后逐一核验「不存在」✅
4. **未触碰**：`/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt`（step500 初始 HF，24G）✅、
   `/data/code`（代码）✅、`/data/code/lingbot-vla-v2/assets/norm_stats/`（`robotwin.json` + `robotwin_competition_clean.json`）✅、
   正式训练数据与其他未授权目录 ✅（`/data/outputs` 目录数 18 → 15）

## 三、结果

| 指标 | 清理前 | 清理后 |
|---|---:|---:|
| `df -h /data` 已用 | 284G | **150G** |
| `df -h /data` 可用 | **60G** | **194G** |
| 使用率 | 83% | **44%** |

**实测释放：134 GB**（`df` 口径；`du` 名义合计 48+48+40=136G，差额为 `du`/`df` 记账差异）。
⇒ 正式训练的磁盘预算（2×DCP 47.4 GiB + 1×HF ≈ 71 GB）**已重新变得宽裕**（194G 可用）✅

## 四、保留与后续

- 未删除：对齐前的 1.49 GB 备份 `backup_code_before_git_align_20261008_234444.tar.gz`（如需再腾空间可删）
- 仍待做（下次开卡）：bf16 HF 真实导出/回读 GPU 复验、DCP Resume 验收（等正式训练产生新 DCP）、多 rank FSDP2/96G 大 GBS

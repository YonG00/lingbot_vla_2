# 审计数据包（本体不入 git）

| 项 | 值 |
|---|---|
| 包路径（远端） | `/data/tmp/al_smoke_tb10_audit.tar.gz` |
| 大小 | 150,023 B（26 个文件，解包 736 KB） |
| **SHA256** | `f23f32eb4e10e82db339f5b7aa570c92fc97330ec2161a3ba7c09aabac679c75` |
| 重新生成 | `tools/make_audit_pkg.sh`（在目标机上运行，只读原始产物、不打包权重/DCP） |
| 复核方式 | 解包后 `sha256sum -c MANIFEST.sha256` |

> 本目录只放**文本对账与清单**（可 review、可 diff）；原始 events/日志/binary 包不入 git。

# al_smoke_tb10 审计数据包

来源：`/data/outputs/al_smoke_gbs4_4task_tb10/`（48G 单卡 BF16，MICRO=1/GAS=4/GBS=4，5-step unit，STEP_OFFSET=500，MAX_STEPS=510，torch.compile **已禁用**）

## 目录
```
run_ok/            成功 run 的原始日志 / 事件 / 显存 / runner 输出
tb/success_run/    成功 run 的 TensorBoard events（原始）
tb/failed_attempt1/ 之前的失败 run（被看门狗杀掉）events，**已分离标记**
tb/failed_p1/      更早一轮失败的 events，**已分离标记**
config/            YAML / 启动器 / runner / VERSION_AND_HASHES.md / effective_args.json
reconcile/         A_unit_loss.md / B_replay.md / C_hardness_and_ckpt.md
MISSING_FILES.md   本次请求但不存在的文件（明确标注）
MANIFEST.sha256    所有打包文件的 SHA256
```

## 未打包（按规则排除）
- DCP（`*.distcp`）、HF 权重（`hf_ckpt/*.safetensors`）、数据集、大型缓存、任何凭据
- Checkpoint 只提供**文件清单与大小**（见 `reconcile/C_hardness_and_ckpt.md`）

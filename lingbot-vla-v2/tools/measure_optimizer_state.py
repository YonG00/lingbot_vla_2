#!/usr/bin/env python3
"""精确测算 LingBot-VLA-v2 训练存档（DCP + hf_ckpt）的体积。

依据源码事实（已 grep 验证）：
  - 权重 dtype = F32（config `enable_fp32: true`）  ⇒ Muon 动量 4 B/参数
  - `DistributedMuon.step` 里 `state["momentum_buffer"] = torch.zeros_like(p)`
    ⇒ 动量缓冲与参数同 dtype = fp32 = 4 B/参数
  - `torch.optim.AdamW` 的 exp_avg / exp_avg_sq 亦为 fp32 ⇒ 8 B/参数
  - 分流规则见 `lingbotvla/optim/dist_muon_params.py`：
      2D 或 3D(shape[0]>1) 且名字不含默认 AdamW 模式 ⇒ Muon
      否则 ⇒ AdamW（含全部 nn.Embedding 与 1D norm/bias）

用法：
  python measure_optimizer_state.py <hf_ckpt_or_model_dir> [--json]

输出：每个可训子集的参数量、optimizer 状态体积、单份存档预估体积。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from collections import defaultdict

# 与 dist_muon_params._DEFAULT_ADAMW_NAME_PATTERNS 保持一致
DEFAULT_ADAMW_PATTERNS = (
    "embed_tokens",
    "embedding",
    "lm_head",
    "output_layer",
    "pos_embed",
    "cls_token",
    "mask_token",
    "storage_tokens",
    "register_tokens",
)

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "U8": 1, "BOOL": 1}


def read_headers(root: str) -> dict:
    """扫描目录下所有 safetensors，返回 {key: (shape, dtype)}。"""
    shards = sorted(glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True))
    if not shards:
        raise SystemExit(f"未找到 safetensors：{root}")
    out = {}
    for f in shards:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            out[k] = (tuple(v["shape"]), v["dtype"])
    return out, shards


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def classify(key: str) -> str:
    """把 FQN 归到 vit / llm / expert / head 四类。"""
    if ".qwenvl.model.visual." in key:
        return "vit"
    if ".qwen_expert." in key:
        return "expert"
    if ".qwenvl." in key:
        return "llm"
    return "head"


def is_muon(name: str, shape) -> bool:
    lname = name.lower()
    if any(p in lname for p in DEFAULT_ADAMW_PATTERNS):
        return False
    nd = len(shape)
    if nd == 2:
        return True
    if nd == 3 and shape[0] > 1:
        return True
    return False


# 两种训练设置的可训集合（与实验/对照组脚本一致）
RUNS = {
    "exp  train_expert_only=true (expert+head)": lambda k: classify(k) in ("expert", "head"),
    "ctrl freeze_vision_encoder=true (all-vit)": lambda k: classify(k) != "vit",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--fp32-model-gib", type=float, default=23.8,
                    help="hf_ckpt / DCP model/ 实测体积（GiB）")
    args = ap.parse_args()

    H, shards = read_headers(args.model_dir)
    total_disk = sum(os.path.getsize(s) for s in shards) / 2 ** 30

    per_cls = defaultdict(lambda: [0, 0, 0])  # keys, numel, bytes
    for k, (s, dt) in H.items():
        c = classify(k)
        per_cls[c][0] += 1
        per_cls[c][1] += numel(s)
        per_cls[c][2] += numel(s) * DTYPE_BYTES.get(dt, 4)

    report = {
        "model_dir": args.model_dir,
        "n_keys": len(H),
        "n_shards": len(shards),
        "disk_gib": round(total_disk, 2),
        "classes": {
            c: {"keys": v[0], "params_b": round(v[1] / 1e9, 4), "gib": round(v[2] / 2 ** 30, 2)}
            for c, v in per_cls.items()
        },
        "runs": {},
    }

    for name, sel in RUNS.items():
        mu = aw = 0
        n_mu = n_aw = 0
        tp = 0
        for k, (s, dt) in H.items():
            if not sel(k):
                continue
            ne = numel(s)
            tp += ne
            if is_muon(k, s):
                mu += ne * 4
                n_mu += 1
            else:
                aw += ne * 8
                n_aw += 1
        opt_gib = (mu + aw) / 2 ** 30
        report["runs"][name] = {
            "trainable_params_b": round(tp / 1e9, 4),
            "trainable_pct": round(tp / sum(v[1] for v in per_cls.values()) * 100, 1),
            "muon_keys": n_mu, "adamw_keys": n_aw,
            "muon_gib": round(mu / 2 ** 30, 2), "adamw_gib": round(aw / 2 ** 30, 2),
            "optimizer_gib": round(opt_gib, 2),
            "save_gib": round(args.fp32_model_gib * 2 + opt_gib, 1),
        }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"模型目录      : {report['model_dir']}")
    print(f"分片/键数     : {report['n_shards']} 片 / {report['n_keys']} 键，磁盘 {report['disk_gib']} GiB")
    print()
    print(f"{'类别':<10}{'键数':>7}{'参数量(B)':>14}{'fp32 GiB':>11}")
    for c in ("vit", "llm", "expert", "head"):
        v = report["classes"].get(c)
        if not v:
            continue
        print(f"{c:<10}{v['keys']:>7}{v['params_b']:>14.4f}{v['gib']:>11.2f}")
    tot_p = sum(v["params_b"] for v in report["classes"].values())
    tot_g = sum(v["gib"] for v in report["classes"].values())
    print(f"{'合计':<10}{report['n_keys']:>7}{tot_p:>14.4f}{tot_g:>11.2f}")
    print()
    print(f"{'训练设置':<44}{'可训(B)':>10}{'占比':>7}{'Muon':>8}{'AdamW':>8}{'opt':>8}{'单份存档':>10}")
    for name, r in report["runs"].items():
        print(f"{name:<44}{r['trainable_params_b']:>10.4f}{r['trainable_pct']:>6.1f}%"
              f"{r['muon_gib']:>8.1f}{r['adamw_gib']:>8.1f}{r['optimizer_gib']:>8.1f}"
              f"{r['save_gib']:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

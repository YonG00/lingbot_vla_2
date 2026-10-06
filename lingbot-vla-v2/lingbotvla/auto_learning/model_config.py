"""按 **deploy / 训练的真实方式** 构造 LingBot-VLA-v2 的 config 与 processor。

为什么需要这个模块（review v0.1 #3）
-----------------------------------
`build_vla_dataset()` 要的不是「基座 Qwen 的 config」，而是 **`LingbotVLAV2Config`**
（含 `max_state_dim` / `max_action_dim` / `chunk_size` / `tokenizer_path`）。
真实 deploy（`deploy/lingbot_vla_v2_policy.py::load_vla`）的做法是：

    training_config = yaml.safe_load(lingbotvla_cli.yaml)
    cfg = dict(training_config['model']);  cfg.update(training_config['train'])
    config = LingbotVLAV2Config(**cfg)

⇒ 这里**照抄同一条路径**，只是**不加载权重**（baseline 只需要 config + processor）。

⚠️ 本模块**不做** deploy 里的 `merge_qwen_config` —— 那一步是为建模/推理准备的
（`hidden_size` / `vision_config` 等）；GT 收集路径上 `FeatureTransform` 只读
`model_config.max_state_dim` / `max_action_dim`，`build_vla_dataset` 只读 `tokenizer_path`。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

__all__ = ["build_lingbot_config", "build_lingbot_processor", "resolve_base_model_path"]


def resolve_base_model_path(tokenizer_path: str) -> str:
    """基座路径：环境变量 `QWEN3VL_PATH` 优先（与 deploy 一致）。"""
    return os.environ.get("QWEN3VL_PATH") or tokenizer_path


def build_lingbot_config(training_config: Dict[str, Any]):
    """从 `lingbotvla_cli.yaml` 的内容构造 `LingbotVLAV2Config`（**不加载权重**）。"""
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )

    if "model" not in training_config:
        raise ValueError("训练配置里没有 `model` 段，无法构造 LingbotVLAV2Config")
    merged = dict(training_config["model"])
    merged.update(training_config.get("train", {}) or {})

    config = LingbotVLAV2Config(**merged)
    # 与 deploy 一致：训练配置里多出来的键也挂上去（deploy 就是这么兜底的）
    for key, value in merged.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    base = resolve_base_model_path(str(merged.get("tokenizer_path", "")))
    if not base:
        raise ValueError("训练配置的 model.tokenizer_path 为空，且环境无 QWEN3VL_PATH")
    config.tokenizer_path = base
    return config


def build_lingbot_processor(config_or_path: Any):
    """构造 processor —— 与训练 `build_processor(args.model.tokenizer_path)` 同一入口。

    ⚠️ **processor 是必需的**：`FeatureTransform.apply()` 会读
    `self.processor.image_processor`（传 None 会在第一个 item 上直接 AttributeError）。
    """
    from lingbotvla.models import build_processor

    path = config_or_path if isinstance(config_or_path, str) else config_or_path.tokenizer_path
    return build_processor(path)


def load_config_and_processor(
    cli_yaml_path: str,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """便捷入口：读 yaml → (LingbotVLAV2Config, processor, raw_yaml)。

    ⚠️ **不改** `use_cache` / `attention_implementation` —— 那是**推理**才需要的
    （deploy 会设 `use_cache=True` + eager）；GT 收集不走模型，保持训练配置原样即可。
    """
    import yaml

    with open(cli_yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = build_lingbot_config(raw)
    processor = build_lingbot_processor(config)
    return config, processor, raw


def effective_chunk_size(raw: Dict[str, Any], config: Any) -> Optional[int]:
    """**有效** chunk_size —— review v0.1 #3 指出的坑。

    `data.chunk_size` 通常**不存在**（训练时才由 `model.config.chunk_size` 补上），
    所以指纹不能只看 `data.chunk_size`，否则会记成 `None` ⇒ 配置变了缓存也不失效。
    """
    for src in (
        (raw.get("data") or {}).get("chunk_size"),
        (raw.get("train") or {}).get("chunk_size"),
        (raw.get("model") or {}).get("chunk_size"),
        getattr(config, "chunk_size", None),
    ):
        if src:
            return int(src)
    return None

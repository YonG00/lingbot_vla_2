"""HardnessScorer —— 固定 noise + 固定 flow time + no_grad → per-sample L1_fm。

复用（**不新增模型逻辑**）
-------------------------
* `LingbotVlaV2Policy.forward` 已经返回 `loss_dict["batch_mean_losses"]`（shape (B,)）
* `FlowMatchingV2.forward` 已经支持外部传入 `noise` / `time`
  ⇒ 固定这两个量即可得到**确定性**的逐样本难度

⚠️ 两个坑
---------
1. **必须传 `joint_mask`**：否则 `LingbotVlaV2Policy.forward` 会落到
   ``losses.mean(dim=(1,2))`` 分支，与训练时的 masked 口径**不同**。
2. `loss_type` 是**从 `model.config.loss_type` 读的**（不是 forward 的入参）
   ⇒ 需要 `L1_fm` 时临时改 config，`finally` 还原。

B0 只做接口 + deterministic test，**不改训练 sampler**。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .eval_context import resolve_logger

DEFAULT_SEED = 1234
DEFAULT_FLOW_TIME = 0.5


def find_loss_dict(outputs: Any) -> Optional[Dict[str, Any]]:
    """从 forward 的返回里找出含 `batch_mean_losses` 的那个 dict。

    真实返回是一个 11 元组（第 7 个是 `loss_dict`），但**按下标硬编码很脆** ——
    这里改成「扫描第一个含目标键的 dict」。
    """
    if isinstance(outputs, dict):
        return outputs if "batch_mean_losses" in outputs else None
    if isinstance(outputs, (tuple, list)):
        for x in outputs:
            if isinstance(x, dict) and "batch_mean_losses" in x:
                return x
    return None


class HardnessScorer:
    """`score(items) -> np.ndarray(B,)`，值为 per-sample `L1_fm`（越大越难）。"""

    def __init__(
        self,
        model: Any,
        *,
        seed: int = DEFAULT_SEED,
        flow_time: float = DEFAULT_FLOW_TIME,
        loss_type: str = "L1_fm",
        device: Optional[str] = None,
        logger: Any = None,
        require_joint_mask: bool = True,
    ):
        self.model = model
        self.seed = int(seed)
        self.flow_time = float(flow_time)
        self.loss_type = str(loss_type)
        # review v0.1 #4：device 默认**从模型推断**，不写死 "cuda"
        self.device = device or self._infer_device()
        self.logger = resolve_logger(logger)
        self.require_joint_mask = bool(require_joint_mask)

    # -- dtype / device（review v0.1 #4）-------------------------------------
    def _infer_device(self) -> str:
        try:
            return str(next(self.model.parameters()).device)
        except Exception:  # noqa: BLE001 —— 桩模型 / 空参数
            return "cuda"

    def forward_dtype(self):
        """**真实 forward 会用的 dtype**。

        `LingbotVlaV2Policy.forward()`：
          * `config.action_fp32=True`  ⇒ `state/actions` 转 **float32**
          * 否则                        ⇒ 转 `next(self.parameters()).dtype`（单卡常见 BF16）

        而 `FlowMatchingV2.forward` 里 `dtype = state.dtype` —— **外部传入的 noise/time
        不会被转换**。所以 noise/time 必须按上面这个 dtype 构造，否则：
        轻则口径与训练不一致，重则直接 dtype mismatch。
        """
        import torch

        if getattr(getattr(self.model, "config", None), "action_fp32", False):
            return torch.float32
        try:
            return next(self.model.parameters()).dtype
        except Exception:  # noqa: BLE001
            return torch.float32

    # -- 确定性构造 ---------------------------------------------------------
    def fixed_noise(self, shape: Sequence[int], dtype):
        """**专用 generator** ⇒ 同一 seed 永远得到同一份噪声。

        ⚠️ 不能靠 `torch.manual_seed` + `noise=None`：`torch.randn` 取的是随机流上
        「第几次」的值，模型加载本身会消耗随机数 ⇒「同一个 seed」≠「同一份噪声」。
        """
        import torch

        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        g = torch.Generator(device=dev)
        g.manual_seed(self.seed)
        return torch.randn(tuple(int(x) for x in shape), generator=g, device=self.device, dtype=dtype)

    def fixed_time(self, batch: int, dtype):
        """固定 flow time（整批同一个 t）⇒ 不同 step / 不同次调用可比。"""
        import torch

        return torch.full((int(batch),), self.flow_time, device=self.device, dtype=dtype)

    # -- 主入口 -------------------------------------------------------------
    def score(self, items: Sequence[Dict[str, Any]]) -> Any:
        """对一批 dataset item 算逐样本难度。

        ``items`` 是**已 transform** 的 dataset item（含 `actions` / `joint_mask`）。
        返回 `np.ndarray`，shape `(B,)`。
        """
        import numpy as np
        import torch
        from torch.utils.data._utils.collate import default_collate

        if not items:
            raise ValueError("items 为空，无法打分")
        batch = default_collate([dict(it) for it in items])
        if self.require_joint_mask and "joint_mask" not in batch:
            raise ValueError(
                "缺少 `joint_mask` ⇒ 会落到未掩码分支，口径与训练不一致。"
                "（确实要用未掩码口径请设 require_joint_mask=False）")

        cfg = getattr(self.model, "config", None)
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        grid = batch.pop("image_grid_thw", None)
        actions = batch.get("actions")
        if actions is None:
            raise ValueError("batch 里没有 `actions`，无法算 flow-matching 损失")
        # 🔴 review v0.1 #4：noise/time 必须用**真实 forward 的 dtype**，
        #    而不是 dataset 里 actions 的 dtype（后者通常是 FP32）。
        dtype = self.forward_dtype()

        n_action_steps = int(getattr(cfg, "n_action_steps", actions.shape[1]))
        max_action_dim = int(getattr(cfg, "max_action_dim", actions.shape[2]))
        noise = self.fixed_noise((actions.shape[0], n_action_steps, max_action_dim), dtype)
        time = self.fixed_time(actions.shape[0], dtype)

        train_flags = [(m, m.training) for m in self.model.modules()]
        old_loss_type = getattr(cfg, "loss_type", None)
        try:
            self.model.eval()
            if old_loss_type != self.loss_type:
                # `loss_type` 是从 config 读的 ⇒ 临时改、finally 还原
                cfg.loss_type = self.loss_type
            with torch.no_grad():
                out = self.model(
                    **batch,
                    noise=noise,
                    time=time,
                    image_grid_thw=grid,
                )
            loss_dict = find_loss_dict(out)
            if loss_dict is None:
                raise RuntimeError(
                    "forward 返回里找不到 `batch_mean_losses`；"
                    f"返回类型 = {type(out).__name__}")
            bml = loss_dict["batch_mean_losses"]
            return bml.detach().float().cpu().numpy().reshape(-1)
        finally:
            if old_loss_type is not None:
                cfg.loss_type = old_loss_type
            for m, t in train_flags:
                m.training = t

    # -- 自检 ---------------------------------------------------------------
    def score_deterministic(self, items: Sequence[Dict[str, Any]]) -> bool:
        """跑两遍，断言逐位一致（real-model 集成测试用）。"""
        import numpy as np

        a = self.score(items)
        b = self.score(items)
        return bool(np.array_equal(a, b))


__all__ = ["HardnessScorer", "find_loss_dict", "DEFAULT_SEED", "DEFAULT_FLOW_TIME"]

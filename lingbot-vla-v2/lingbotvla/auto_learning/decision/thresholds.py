"""按任务的**通过阈值表**（pass thresholds）—— 判定口径的唯一入口。

为什么要有这个模块
------------------
原来全框架只有**一个全局标量** `cfg.pass_nmse`，在 **5 个语义位置 / 7 个比较点**被比较：

| # | 位置 | 语义 |
|---|------|------|
| 1 | ``decision/state_machine.py`` ``decide()`` | 主 PASS 判定（每个 unit 后） |
| 2 | ``decision/state_machine.py`` ``forget_code()`` | 遗忘**原因码**分类 |
| 3 | ``orchestration/scheduler.py`` bootstrap | 免费 PASS（scout→confirm） |
| 4 | ``orchestration/scheduler.py`` rescan | 免费 PASS（状态迁移后重扫） |
| 5 | ``orchestration/baselines.py`` | **uniform 对照组统计**（漏改 ⇒ 对比静默失效） |

另外 ``decision/review.py`` 的 ``is_forgotten`` / ``_suspect`` 也读它。

这些点各写各的比较 ⇒ 一旦改口径就必然漏改，而**漏改不会报错**（尤其第 5 处：只会让
uniform / AL 两组的"通过"口径不一致，实验结论静默错掉）。所以本模块把口径收敛到
**两个函数**：:func:`pass_line` 与 :func:`check_pass` / :func:`is_forgotten_ex`。

三种口径
--------
* ``pass_metric="nmse"``（**默认，行为与改造前逐字一致**）：阈值 = ``cfg.pass_nmse``。
* ``pass_metric="mse"``：阈值 = **该任务**在阈值表里的值（算术平均动作 MSE）。
* ``pass_metric="gmean_mse"``：阈值来自 ``stat="geomean"`` 的参考表；
  候选也对逐轨迹 MSE 做等权几何聚合。

⚠️ 前两种口径**数学等价**（``nmse = mse / baseline_mse``，baseline 是该任务的固定常数），
差别只在"拿哪个数去比 / 报告里显示哪个数"。切到 ``mse`` 的好处是量纲直观：
直接和**成品模型**在同一任务上的绝对误差对比。

🔴 为什么不再"全用 nmse"
------------------------
``nmse`` 的分母是**该任务自身的动作方差** ⇒ 分母尺度因任务而异，
"跨任务比较"必须用它（否则基线大的任务天然吃亏）；而"判定"要的是绝对精度 ⇒ 用 MSE。
⇒ **判定用 MSE，跨任务排序/统计仍用 NMSE**，各司其职（见 ``scheduler.py`` 的 worst/median 统计）。

``None`` 阈值的语义（重要）
---------------------------
阈值表里某个任务可以是 ``None``，含义是"**该任务没有可用的通过线**"。
正常标定流程里每个任务都会有开环分数 ⇒ 都有数值线；``None`` 只作为**异常兜底**
（该任务评测结果缺失、或阈值表被显式标成 null 表示"暂不考核"）。
此时 :func:`check_pass` 返回 :attr:`PassCheck.NO_THRESHOLD`：

* **不判 PASS**（绝不回退成宽松阈值）
* 上层应把它当成"待标定 / needs_calibration"，且**不消耗 attempt 预算**

配置指纹
--------
阈值是从某个 ``task_baseline.json``（同一份数据/归一化/相机配置）标定出来的。
若 baseline 重算而阈值表没重算，两个数会错位且**不报错** ⇒ 本模块在 ``load()`` 时
**强校验 ``config_fingerprint``**，不一致直接 fail-fast（除非显式 ``allow_fingerprint_mismatch``）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .metrics import is_finite_metric

#: 判定口径
PASS_METRIC_NMSE = "nmse"
PASS_METRIC_MSE = "mse"
PASS_METRIC_GMEAN = "gmean_mse"
PASS_METRICS = (PASS_METRIC_NMSE, PASS_METRIC_MSE, PASS_METRIC_GMEAN)

#: 阈值表文件版本
THRESHOLDS_VERSION = 1

# --------------------------------------------------------------------------- #
# 候选池 ratio 分档（``cfg.pool_filter_by_gmean_ratio``，**默认关闭**）
#
# 这是**候选池筛选**的尺子，与 PASS 判定（:func:`check_pass`）是两件事：
# PASS 判定永远比较 ``实测值 <= 该任务参考线``；ratio 分档只决定"要不要花算力去练"。
# 两者共用同一把尺子（同一个 ``gmean_mse`` 与同一条参考线）⇒ 不会分叉。
# --------------------------------------------------------------------------- #
POOL_RATIO_DISABLED = "disabled"      # 开关关闭 ⇒ 不用 ratio 判据（旧 NMSE 口径）
POOL_RATIO_PASSED = "pool_passed"     # ratio < pool_ratio_pass ⇒ 视为已过，不进候选池
POOL_RATIO_TRAINABLE = "trainable"    # pass <= ratio <= skip ⇒ 可练，进候选池
POOL_RATIO_TOO_HARD = "too_hard"      # ratio > pool_ratio_skip ⇒ 太难，暂不选
POOL_RATIO_NO_LINE = "no_line"        # 该任务没有可用参考线 ⇒ 算不出 ratio
POOL_RATIO_INVALID = "invalid"        # 实测 gmean 非有限（NaN/Inf/缺失）
POOL_RATIO_BUCKETS = (
    POOL_RATIO_DISABLED, POOL_RATIO_PASSED, POOL_RATIO_TRAINABLE,
    POOL_RATIO_TOO_HARD, POOL_RATIO_NO_LINE, POOL_RATIO_INVALID,
)


class ThresholdsError(ValueError):
    """阈值表加载/校验失败（配置或文件问题，必须让人看到）。"""


# --------------------------------------------------------------------------- #
# 阈值表
# --------------------------------------------------------------------------- #
@dataclass
class PassThresholds:
    """一份按任务的通过阈值表（判定口径 = ``metric`` 指定的那个）。"""

    config_fingerprint: str = ""
    tasks: Dict[str, Optional[float]] = field(default_factory=dict)
    metric: str = PASS_METRIC_MSE
    margin: float = 0.0
    stat: str = ""
    reference: str = ""
    version: int = THRESHOLDS_VERSION

    # ---------------------------------------------------------------- #
    def get(self, task: str) -> Optional[float]:
        """取该任务的阈值；任务不在表里 ⇒ ``None``（= 无可用线）。"""
        return self.tasks.get(task)

    def __contains__(self, task: str) -> bool:
        return task in self.tasks

    @property
    def n_usable(self) -> int:
        return sum(1 for v in self.tasks.values() if v is not None)

    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "metric": self.metric,
            "config_fingerprint": self.config_fingerprint,
            "margin": self.margin,
            "stat": self.stat,
            "reference": self.reference,
            "tasks": dict(self.tasks),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PassThresholds":
        if not isinstance(raw, dict):
            raise ThresholdsError(f"阈值表必须是 dict，实际 {type(raw).__name__}")
        tasks_raw = raw.get("tasks")
        if not isinstance(tasks_raw, dict):
            raise ThresholdsError("阈值表缺少 'tasks'（必须是 dict: 任务名 → 阈值或 null）")
        tasks: Dict[str, Optional[float]] = {}
        for name, value in tasks_raw.items():
            if value is None:
                tasks[str(name)] = None
                continue
            # 允许写成 {"pass_threshold": x} 的详细形式
            if isinstance(value, dict):
                value = value.get("pass_threshold", value.get("threshold"))
            if value is None:
                tasks[str(name)] = None
                continue
            try:
                tasks[str(name)] = float(value)
            except (TypeError, ValueError):
                raise ThresholdsError(
                    f"任务 {name!r} 的阈值不是数字也不是 null：{value!r}"
                ) from None
        metric = str(raw.get("metric") or PASS_METRIC_MSE)
        if metric not in PASS_METRICS:
            raise ThresholdsError(f"阈值表 metric 非法：{metric!r}（只能是 {PASS_METRICS}）")
        return cls(
            config_fingerprint=str(raw.get("config_fingerprint") or ""),
            tasks=tasks,
            metric=metric,
            margin=float(raw.get("margin") or 0.0),
            stat=str(raw.get("stat") or ""),
            reference=str(raw.get("reference") or ""),
            version=int(raw.get("version") or THRESHOLDS_VERSION),
        )

    # ---------------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        path: str,
        *,
        expect_fingerprint: Optional[str] = None,
        allow_fingerprint_mismatch: bool = False,
        require_metric: Optional[str] = None,
    ) -> "PassThresholds":
        """从 JSON 读阈值表并做**强校验**。

        参数
        ----
        expect_fingerprint
            调用方算出的 config 指纹（应与 ``task_baseline.json`` 的同一个）。
            与文件里的不一致 ⇒ fail-fast（这是防"baseline 重算了、阈值没重算"的静默错判）。
        allow_fingerprint_mismatch
            显式放行（仅测试/临时用途），放行时会打印警告。
        require_metric
            要求文件的 ``metric`` 必须是它（防止把 nmse 单位的表当 mse 用）。
        """
        if not os.path.isfile(path):
            raise ThresholdsError(f"阈值表文件不存在：{path}")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        obj = cls.from_dict(raw)

        if require_metric is not None and obj.metric != require_metric:
            raise ThresholdsError(
                f"阈值表 metric={obj.metric!r}，但当前口径要求 {require_metric!r}；"
                "两者单位不同，混用会直接错判。"
            )
        if expect_fingerprint is not None:
            if not obj.config_fingerprint:
                raise ThresholdsError(
                    f"阈值表 {path} 没有 config_fingerprint ⇒ 无法确认它与当前数据/归一化配置"
                    "匹配（这类错位不会报错，只会静默错判），拒绝加载。"
                )
            if obj.config_fingerprint != expect_fingerprint and not allow_fingerprint_mismatch:
                raise ThresholdsError(
                    "阈值表与当前配置的 config_fingerprint 不一致：\n"
                    f"  阈值表 : {obj.config_fingerprint}\n"
                    f"  当前   : {expect_fingerprint}\n"
                    "⇒ 多半是数据/归一化/相机配置变了但阈值表没重算。"
                    "请重跑 tools/compute_pass_thresholds.py；"
                    "确要用旧表请显式 allow_fingerprint_mismatch=True。"
                )
            if obj.config_fingerprint != expect_fingerprint:
                import warnings

                warnings.warn(
                    "PassThresholds: 显式放行了 config_fingerprint 不一致 "
                    f"({obj.config_fingerprint} != {expect_fingerprint})",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return obj

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")


# --------------------------------------------------------------------------- #
# 口径解析（**全框架唯一的判定入口**）
# --------------------------------------------------------------------------- #
def verify_threshold_stat_compatible(table: PassThresholds, *, pass_metric: str = PASS_METRIC_MSE) -> None:
    """Fail closed unless reference statistic matches the selected candidate metric."""
    if pass_metric == PASS_METRIC_GMEAN:
        if table.metric != PASS_METRIC_MSE or table.stat != "geomean":
            raise ThresholdsError("pass_metric='gmean_mse' 要求阈值表 metric='mse' 且 stat='geomean'")
        import math
        for task, value in table.tasks.items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ThresholdsError(f"gmean_mse: {task!r} 的阈值必须是有限正数或 null")
        return
    if table.stat == "geomean":
        raise ThresholdsError(
            "阈值表 stat='geomean'，但运行时 pass_metric='mse' 仍比较候选轨迹的算术平均 MSE；"
            "两者不是同一个统计量，拒绝静默错判 PASS。"
            "请先实现并验收候选 GMean-MSE 完整判定链路；不能只更换阈值表。")


def active_metric(cfg: Any) -> str:
    """当前生效的判定口径（``nmse`` / ``mse`` / ``gmean_mse``）。"""
    metric = str(getattr(cfg, "pass_metric", PASS_METRIC_NMSE) or PASS_METRIC_NMSE)
    if metric not in PASS_METRICS:
        raise ThresholdsError(f"cfg.pass_metric 非法：{metric!r}")
    return metric


def attached_thresholds(cfg: Any) -> Optional[PassThresholds]:
    """取挂在 cfg 上的阈值表（由 ``real/build.py`` 加载后挂上；没挂 ⇒ None）。"""
    obj = getattr(cfg, "pass_thresholds", None)
    return obj if isinstance(obj, PassThresholds) else None


def pass_line(cfg: Any, task: str) -> Tuple[str, Optional[float]]:
    """返回 ``(metric, 阈值)`` —— **本模块之外不应再出现裸的 ``pass_nmse`` 比较**。

    * ``metric="nmse"`` ⇒ 阈值 = ``cfg.pass_nmse``（永远有值，沿用旧行为）
    * ``metric="mse"``  ⇒ 阈值 = 阈值表里该任务的值；
      表没挂 / 任务不在表里 / 表里是 ``None`` ⇒ 阈值 ``None``（= 无可用线，不判 PASS）
    """
    metric = active_metric(cfg)
    if metric == PASS_METRIC_NMSE:
        line = getattr(cfg, "pass_nmse", None)
        return metric, (float(line) if line is not None else None)
    table = attached_thresholds(cfg)
    if table is None:
        return metric, None
    if metric == PASS_METRIC_GMEAN:
        verify_threshold_stat_compatible(table, pass_metric=metric)
    return metric, table.get(task)


def metric_value(metric: str, *, nmse: Optional[float], mse: Optional[float], gmean_mse: Optional[float] = None) -> Optional[float]:
    """按口径挑出参与比较的那个数值。"""
    return gmean_mse if metric == PASS_METRIC_GMEAN else (mse if metric == PASS_METRIC_MSE else nmse)


# --------------------------------------------------------------------------- #
# 候选池 ratio（``pool_filter_by_gmean_ratio``）—— **只筛池子，不改 PASS 判定**
# --------------------------------------------------------------------------- #
def pool_filter_enabled(cfg: Any) -> bool:
    """候选池是否改用 GMean ratio 筛选（**默认关闭**）。

    ``cfg.pool_filter_by_gmean_ratio`` 是总开关；口径不是 ``gmean_mse`` 时**视为未启用**
    （``AutoLearningConfig.validate()`` 已在配置层 fail-fast，这里再兜一层：
    绝不静默拿别的尺子算 ratio）。
    """
    if not bool(getattr(cfg, "pool_filter_by_gmean_ratio", False)):
        return False
    return active_metric(cfg) == PASS_METRIC_GMEAN


def gmean_pool_ratio(cfg: Any, task: str, *, gmean_mse: Optional[float]) -> Optional[float]:
    """``ratio = 该任务实测 gmean_mse / 该任务的参考线``。

    参考线 = ``pass_thresholds_file`` 里该任务的值（与 PASS 判定**同一条线**）。
    没有可用线 / 参考线非正 / 实测值非有限 ⇒ ``None``（算不出 ratio，不做任何猜测）。
    """
    metric, line = pass_line(cfg, task)
    if metric != PASS_METRIC_GMEAN:
        return None
    if not (line is not None and is_finite_metric(line) and float(line) > 0.0):
        return None
    if not is_finite_metric(gmean_mse):
        return None
    return float(gmean_mse) / float(line)


def gmean_pool_bucket(cfg: Any, task: str, *, gmean_mse: Optional[float]) -> str:
    """把一条 scout/confirm 的 gmean 实测值分到 :data:`POOL_RATIO_BUCKETS` 里的某一档。

    开关关闭 ⇒ :data:`POOL_RATIO_DISABLED`（调用方必须走旧行为，本函数绝不自行降级）。
    """
    if not pool_filter_enabled(cfg):
        return POOL_RATIO_DISABLED
    if not is_finite_metric(gmean_mse):
        return POOL_RATIO_INVALID
    ratio = gmean_pool_ratio(cfg, task, gmean_mse=gmean_mse)
    if ratio is None:
        return POOL_RATIO_NO_LINE
    if ratio < float(getattr(cfg, "pool_ratio_pass", 0.2)):
        return POOL_RATIO_PASSED
    if ratio > float(getattr(cfg, "pool_ratio_skip", 5.0)):
        return POOL_RATIO_TOO_HARD
    return POOL_RATIO_TRAINABLE


# --------------------------------------------------------------------------- #
# 判定结果
# --------------------------------------------------------------------------- #
class PassCheck(str):
    """判定结果（用 str 子类，日志/事件里可直接序列化）。"""

    PASS = "pass"                    # 达标
    BELOW = "below"                  # 有阈值但没到
    NO_THRESHOLD = "no_threshold"    # 该任务没有可用阈值 ⇒ 不判 PASS（needs_calibration）
    INVALID = "invalid"              # 指标非有限（NaN/Inf）


def check_pass(cfg: Any, task: str, *, nmse: Optional[float], mse: Optional[float],
               gmean_mse: Optional[float] = None) -> str:
    """统一的"是否达标"判定。**所有 PASS 判定点都必须走这里。**

    返回 :class:`PassCheck` 里的一个字符串常量。
    """
    metric, line = pass_line(cfg, task)
    if line is None:
        return PassCheck.NO_THRESHOLD
    val = metric_value(metric, nmse=nmse, mse=mse, gmean_mse=gmean_mse)
    if not is_finite_metric(val):
        return PassCheck.INVALID
    return PassCheck.PASS if float(val) <= line else PassCheck.BELOW


def is_pass(cfg: Any, task: str, *, nmse: Optional[float], mse: Optional[float],
            gmean_mse: Optional[float] = None) -> bool:
    """:func:`check_pass` 的布尔快捷方式（``NO_THRESHOLD`` / ``INVALID`` 都算 False）。"""
    return check_pass(cfg, task, nmse=nmse, mse=mse, gmean_mse=gmean_mse) == PassCheck.PASS


def is_forgotten_ex(
    cfg: Any,
    task: str,
    *,
    cur_nmse: Optional[float],
    cur_mse: Optional[float],
    best_nmse: Optional[float],
    cur_gmean_mse: Optional[float] = None,
    best_gmean_mse: Optional[float] = None,
) -> bool:
    """遗忘判定：**掉出及格线 或 相对退化超阈值**（文档 §34）。

    * "掉出及格线"那半用 :func:`pass_line`（⇒ 口径跟着 ``pass_metric`` 走）。
      该任务**没有可用阈值**时，这半**不参与判定**（只剩相对退化）。
    * "相对退化"那半是 ``(cur − best) / best`` —— **尺度无关的比值**，
      换成 mse 后数值几乎不变（分子分母同乘 baseline）⇒ 固定用 nmse 计算。
    """
    from .metrics import forget_ratio  # 局部导入，避免循环依赖

    metric, line = pass_line(cfg, task)
    if line is not None:
        val = metric_value(metric, nmse=cur_nmse, mse=cur_mse, gmean_mse=cur_gmean_mse)
        if is_finite_metric(val) and float(val) > line:
            return True
    # GMean 的相对退化也应使用 GMean 自己的历史最佳值。
    ratio = (forget_ratio(best_gmean_mse, cur_gmean_mse)
             if metric == PASS_METRIC_GMEAN else forget_ratio(best_nmse, cur_nmse))
    thr = getattr(cfg, "forget_relative_threshold", None)
    return ratio is not None and thr is not None and ratio > thr


def forget_code_ex(
    cfg: Any,
    task: str,
    *,
    cur_nmse: Optional[float],
    cur_mse: Optional[float],
    cur_gmean_mse: Optional[float] = None,
) -> str:
    """遗忘**原因码**（掉出及格线 vs 相对退化）—— 与判定口径保持一致。

    ⚠️ 这里从 `..types` 取 ReasonCode（**不能**从 `.state_machine` 取：本模块被
    state_machine 反向依赖，会形成循环导入）。
    """
    from ..types import ReasonCode

    metric, line = pass_line(cfg, task)
    if line is not None:
        val = metric_value(metric, nmse=cur_nmse, mse=cur_mse, gmean_mse=cur_gmean_mse)
        if val is not None and is_finite_metric(val) and float(val) > line:
            return ReasonCode.FORGOTTEN_BELOW_PASS_LINE.value
    return ReasonCode.FORGOTTEN_RELATIVE_DEGRADATION.value


__all__ = [
    "PASS_METRIC_NMSE", "PASS_METRIC_MSE", "PASS_METRIC_GMEAN", "PASS_METRICS",
    "THRESHOLDS_VERSION", "ThresholdsError",
    "POOL_RATIO_DISABLED", "POOL_RATIO_PASSED", "POOL_RATIO_TRAINABLE",
    "POOL_RATIO_TOO_HARD", "POOL_RATIO_NO_LINE", "POOL_RATIO_INVALID",
    "POOL_RATIO_BUCKETS",
    "PassThresholds", "PassCheck",
    "active_metric", "attached_thresholds", "pass_line", "metric_value",
    "verify_threshold_stat_compatible",
    "pool_filter_enabled", "gmean_pool_ratio", "gmean_pool_bucket",
    "check_pass", "is_pass", "is_forgotten_ex", "forget_code_ex",
]

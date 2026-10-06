"""观测层 —— 旁挂只读，不参与任何决策。

    logger.py   events.jsonl + history.csv + TB 同名 tag（文档 §40 的 5 个 panel）
    report.py   控制台表格 / summary.json / heatmap（文档 §39）

因为只读，所以「关掉观测层」不会改变任何结果 —— 这也是它能被放心塞满埋点的原因。
"""

from .logger import EventLogger
from .report import build_heatmap_matrix, format_event, format_table, plot_all

__all__ = ["EventLogger", "plot_all", "build_heatmap_matrix", "format_table", "format_event"]

"""纯CPU汇总clear/reuse对照日志；不启动训练，不导入torch。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
from statistics import median


def summarize(path):
    groups = defaultdict(list)
    steps = {}
    eval_seconds = []
    first_time = last_time = None
    oom_warnings = 0
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            stamp = re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", line)
            if stamp:
                last_time = datetime.strptime(stamp[0], "%m/%d/%Y %H:%M:%S")
                if first_time is None:
                    first_time = last_time
            if "[cuda_cache] " in line:
                try:
                    report = json.loads(line.split("[cuda_cache] ", 1)[1])
                    if not isinstance(report, dict) or not isinstance(report.get("context"), str):
                        continue
                    if (not isinstance(report.get("cleared"), bool)
                            or not all(isinstance(report.get(k), dict) for k in ("before", "after"))
                            or not all(isinstance(report.get(k), (int, float))
                                       and math.isfinite(report[k]) and report[k] >= 0
                                       for k in ("gc_seconds", "cuda_seconds"))):
                        continue
                    groups[report["context"]].append(report)
                except (ValueError, KeyError):
                    pass
            match = re.search(r"\bStep (\d+)/\d+.*?\bLoss ([0-9.eE+-]+),.*?\bStepTime ([0-9.]+)s", line)
            if match:
                steps[int(match[1])] = (float(match[2]), float(match[3]))
            match = re.search(r"\(\d+轨迹/\d+chunk, ([0-9.]+)s\)", line)
            if match:
                eval_seconds.append(float(match[1]))
            if "批处理 OOM" in line or "CUDA out of memory" in line or "OutOfMemoryError" in line:
                oom_warnings += 1
    step_times = [value[1] for _, value in sorted(steps.items())]
    summary = {
        "steps": len(step_times), "median_step_seconds": median(step_times) if step_times else None,
        "step_time_sum_seconds": sum(step_times), "oom_log_lines": oom_warnings, "cache": {},
        "step_range": [min(steps), max(steps)] if steps else None,
        "last_loss": steps[max(steps)][0] if steps else None,
        "eval_calls": len(eval_seconds), "eval_time_sum_seconds": sum(eval_seconds),
        "elapsed_log_seconds": (last_time - first_time).total_seconds() if first_time else None,
    }
    for context, reports in groups.items():
        summary["cache"][context] = {
            "calls": len(reports), "cleared": sum(bool(r["cleared"]) for r in reports),
            "gc_seconds": sum(r["gc_seconds"] for r in reports),
            "cuda_call_seconds": sum(r["cuda_seconds"] for r in reports),
            "max_observed_reserved_bytes": max(
                (snapshot.get("reserved", 0) for r in reports
                 for snapshot in (r["before"], r["after"])), default=0),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-log", required=True)
    parser.add_argument("--reuse-log", required=True)
    args = parser.parse_args()
    print(json.dumps({"clear": summarize(args.clear_log), "reuse": summarize(args.reuse_log)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Auto Learning 的**安全 DCP 保留策略**（用户 2026-10-08 批准的规则）。

规则（safe-by-default，宁可少删也不误删）：

1. 只保留**最近 `keep_last` 份「完整」DCP**（默认 2）；
2. **先确认新 DCP 完整、再清理最旧**：调用方必须把刚保存成功并校验过的 step
   作为 `newest_verified_step` 传进来；
3. **新 DCP 保存失败（或存在更新的未完成目录）时，一个字节都不删** —— 旧的有效
   恢复点必须完好；
4. 只认 DCP 根目录下的**直接子目录** `global_step_<N>`；`hf_milestones/`（HF 里程碑树）
   与任何其它目录永不参与清理，且根目录本身若指向 HF 树则直接拒绝执行；
5. 未完成的目录（崩溃残留）**既不删也不计入保留份数**，只记日志提示；
6. 每一步的保留/删除都写日志（含 step 列表）。

本模块只有纯决策 + 一层薄执行，便于 CPU 临时目录测试。
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

#: 只认这种直接子目录名
_DCP_DIR_RE = re.compile(r"^global_step_(\d+)$")

#: 完整性要求：路径必须是**非空**的目录/文件
_REQUIRED_DIRS = (("model",), ("optimizer",), ("extra_state",))

#: 禁止把清理根指向这些名字（HF 里程碑树）
_FORBIDDEN_ROOT_NAMES = frozenset({"hf_milestones"})


def _nonempty_dir(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))


def is_complete_dcp(path: str) -> bool:
    """结构性完整性判断：model/optimizer/extra_state 三个非空目录 + 各自 .metadata。

    这是**保守**判断：任何异常/缺失都返回 False（False 只意味着"不参与清理"，
    绝不意味着"可以删"）。
    """
    if not os.path.isdir(path):
        return False
    try:
        for rel in _REQUIRED_DIRS:
            if not _nonempty_dir(os.path.join(path, *rel)):
                return False
        for rel in (("model", ".metadata"), ("optimizer", ".metadata")):
            if not os.path.isfile(os.path.join(path, *rel)):
                return False
        if not any(name.endswith(".distcp") for name in os.listdir(os.path.join(path, "model"))):
            return False
        if not any(name.endswith(".distcp") for name in os.listdir(os.path.join(path, "optimizer"))):
            return False
    except OSError:
        return False
    return True


def list_dcps(root: str) -> List[Tuple[int, str]]:
    """返回 `[(step, path)]`，按 step 升序；只认 `global_step_<N>` 直接子目录。"""
    if not os.path.isdir(root):
        return []
    found: List[Tuple[int, str]] = []
    for name in os.listdir(root):
        m = _DCP_DIR_RE.match(name)
        if not m:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            found.append((int(m.group(1)), path))
    found.sort(key=lambda item: item[0])
    return found


def root_is_forbidden(root: str) -> bool:
    """清理根不得是（或位于）HF 里程碑树 —— 防止误删里程碑。"""
    parts = [p for p in os.path.normpath(root).split(os.sep) if p]
    return any(p in _FORBIDDEN_ROOT_NAMES for p in parts)


def should_prune(*, smoke_no_checkpoint: bool, keep_last: Optional[int], is_rank0: bool) -> bool:
    """执行前的总开关（纯函数，便于测试）。"""
    if smoke_no_checkpoint or not is_rank0:
        return False
    return bool(keep_last) and int(keep_last) > 0


def plan_retention(
    root: str,
    keep_last: int,
    *,
    newest_verified_step: Optional[int] = None,
) -> Dict[str, Any]:
    """纯决策，不触碰磁盘。返回 keep / delete / incomplete / guarded 及原因。"""
    plan: Dict[str, Any] = {
        "root": root,
        "keep_last": int(keep_last),
        "keep": [], "delete": [], "incomplete": [],
        "guarded": False, "reason": "",
    }
    if root_is_forbidden(root):
        plan["guarded"] = True
        plan["reason"] = f"拒绝在 HF 里程碑树内执行清理: {root}"
        return plan
    if plan["keep_last"] < 1:
        plan["guarded"] = True
        plan["reason"] = "keep_last<1 ⇒ 保留策略关闭（不删任何东西）"
        return plan

    entries = list_dcps(root)
    if not entries:
        plan["guarded"] = True
        plan["reason"] = "根目录下没有 global_step_* 目录"
        return plan

    complete: List[Tuple[int, str]] = []
    for step, path in entries:
        (complete if is_complete_dcp(path) else plan["incomplete"]).append((step, path))

    if not complete:
        plan["guarded"] = True
        plan["reason"] = "没有任何「完整」DCP ⇒ 不清理"
        return plan

    newest_complete_step = complete[-1][0]
    # ---- 关键保护：本次保存必须"就是最新的完整 DCP"才允许清理 ----
    if newest_verified_step is None:
        plan["guarded"] = True
        plan["reason"] = "没有已验证的新 DCP（保存失败/未校验）⇒ 不清理"
        return plan
    if int(newest_verified_step) != newest_complete_step:
        plan["guarded"] = True
        plan["reason"] = (
            f"已验证 step={int(newest_verified_step)} 但最新完整 DCP 是 {newest_complete_step}"
            "（可能保存失败或存在更新的残留）⇒ 不清理"
        )
        return plan
    if plan["incomplete"] and max(s for s, _ in plan["incomplete"]) > newest_complete_step:
        plan["guarded"] = True
        plan["reason"] = (
            f"存在比 {newest_complete_step} 更新的未完成目录"
            f"（step={max(s for s, _ in plan['incomplete'])}）⇒ 保守不清理"
        )
        return plan

    plan["keep"] = complete[-plan["keep_last"]:]
    plan["delete"] = complete[:-plan["keep_last"]]
    return plan


def prune_dcps(
    root: str,
    keep_last: int,
    *,
    newest_verified_step: Optional[int] = None,
    logger: Any = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """按 `plan_retention` 的结论执行删除，并逐步写日志。返回执行结果。"""
    plan = plan_retention(root, keep_last, newest_verified_step=newest_verified_step)
    kept = [s for s, _ in plan["keep"]]
    victims = [(s, p) for s, p in plan["delete"]]
    incomplete = [s for s, _ in plan["incomplete"]]

    def _log(msg: str, warn: bool = False) -> None:
        if logger is None:
            return
        (logger.warning if warn else logger.info_rank0)(msg)

    if plan["guarded"]:
        _log(f"[ckpt-retention] 跳过清理：{plan['reason']}；保留 {kept} ｜ 未完成 {incomplete}")
        plan["deleted"] = []
        return plan

    deleted: List[int] = []
    failures: List[Tuple[int, str]] = []
    for step, path in victims:
        if dry_run:
            deleted.append(step)
            continue
        try:
            shutil.rmtree(path)
            deleted.append(step)
        except OSError as exc:  # 单个目录失败不影响其它，且绝不掩盖已保留的恢复点
            failures.append((step, repr(exc)))
    _log(
        f"[ckpt-retention] 保留最近 {plan['keep_last']} 份完整 DCP：保留 step={kept}"
        f"｜删除 step={deleted}｜未完成（不动）step={incomplete}"
        + (f"｜删除失败 {failures}" if failures else "")
    )
    plan["deleted"] = deleted
    plan["failures"] = failures
    return plan

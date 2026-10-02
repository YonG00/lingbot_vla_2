"""checkpoint 保存兜底 + 磁盘容量保护。

从 tasks/vla/train_lingbotvla.py 抽出来, 便于单元测试 (该训练脚本导入需要
CUDA, 而本模块只依赖 torch.distributed / shutil)。

包含两部分:

1. DCP 保存兜底 (`dcp_save_or_abort`)
   DCP 保存失败 -> 打印完整 traceback -> 直接 re-raise, 让 torchrun 正常
   teardown 整个多卡作业。异常路径里**绝不**做任何 distributed collective,
   因为此时某些 rank 可能已经卡在 checkpoint collective 上, 再插入集合操作
   会形成二次死锁。

2. 磁盘容量保护 (`disk_guard_check`)
   测量: 后台 HF 线程在 HF 全部写盘完成的那一刻记录 disk_avail_after 并算出
         checkpoint_used —— 不阻塞训练主循环。
   判断: 主循环周期性调用 disk_guard_check, 用【当前实时可用空间】与
         max_checkpoint_used * margin 比较。
   一致性: 决策由 rank0 计算并 broadcast 给所有 rank。实测占用只在 rank0 上
         可得, 若各 rank 各自判断会出现分歧, 导致 rank0 退出而其余 rank 仍
         进入 Checkpointer.save() 的 distributed 死锁。
"""
import logging
import os
import shutil

import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def disk_avail_gb(path):
    """返回 path 所在文件系统的可用空间 (GB); 读取失败返回 None。"""
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return None


def dist_ready():
    return dist.is_available() and dist.is_initialized()


def is_rank0():
    return (not dist_ready()) or dist.get_rank() == 0


def _log0(msg):
    """只在 rank0 打日志, 避免多卡重复刷屏。"""
    if is_rank0():
        logger.info(msg)


# ---------------------------------------------------------------------------
# 1. DCP 保存兜底
# ---------------------------------------------------------------------------

def dcp_save_or_abort(checkpointer, save_path, state, global_step):
    """保存 DCP; 失败则记录完整 traceback 并 re-raise。

    不 catch 后继续训练, 也不在异常路径里做任何 distributed collective ——
    直接抛出, 交给 torchrun teardown 整个作业。
    """
    try:
        checkpointer.save(save_path, state, global_steps=global_step)
    except Exception:
        logger.exception(
            "[checkpoint] DCP checkpoint save failed at global_step=%s; "
            "aborting training.", global_step)
        raise


# ---------------------------------------------------------------------------
# 2. 磁盘容量保护
# ---------------------------------------------------------------------------

def eval_rank0(save_root, hf_saver, max_used_gb, margin, label=""):
    """rank0 侧的容量判断。返回 (allow, max_used_gb, avail_gb_or_None, reason)。

    allow=False 的两种情况:
      * reason="disk"      : 剩余空间不足以再存一份 checkpoint
      * reason="hf_failed" : 上一份的异步 HF 保存失败 (DCP 仍有效, 但停止后续训练)

    fail-open (allow=True, reason=""): 读盘失败 / 占用实测无效 / 收集异常。
    """
    # 1) 收集上一份 checkpoint 的实测占用 (正常情况下任务早已完成, 不阻塞)
    try:
        drained = hf_saver.drain_pending_best_effort()
    except Exception as exc:
        _log0(f"[DiskCheck] ({label}) 收集上一份占用失败 ({exc!r}); 跳过本轮容量判断")
        return True, max_used_gb, None, ""

    # 2) 读当前实时可用空间 (即使跳过判断也要读, 供本次 disk_avail_before 采样)
    avail_gb = disk_avail_gb(save_root)

    # 3) 逐份处理上一轮的实测结果
    skip_round = False
    for r in drained:
        # 3a) 异步 HF 保存失败 -> 不 crash (DCP 已成功), 但停止后续训练
        if getattr(r, "hf_save_failed", False) or getattr(r, "error", ""):
            _log0(
                f"[checkpoint] step={getattr(r, 'global_step', '?')} 异步 HF 保存失败 "
                f"({str(getattr(r, 'error', ''))[:160]}); DCP 仍有效, 停止后续训练。"
            )
            return False, max_used_gb, avail_gb, "hf_failed"

        used = getattr(r, "checkpoint_used_gb", -1.0)
        if not used or used <= 0:
            _log0(
                f"[DiskCheck] step={getattr(r, 'global_step', '?')} 占用实测无效 "
                f"(checkpoint_used={used}); 跳过本轮容量判断"
            )
            skip_round = True
            continue
        max_used_gb = max(max_used_gb, used)
        _log0(
            f"[DiskCheck] step={getattr(r, 'global_step', '?')} "
            f"disk_avail_before={getattr(r, 'disk_avail_before_gb', -1.0):.1f}GB "
            f"disk_avail_after={getattr(r, 'disk_avail_after_gb', -1.0):.1f}GB "
            f"checkpoint_used={used:.1f}GB "
            f"max_checkpoint_used={max_used_gb:.1f}GB"
        )

    if skip_round:
        return True, max_used_gb, avail_gb, ""

    if avail_gb is None:
        _log0(f"[DiskCheck] ({label}) 读取磁盘空间失败; 跳过本轮容量判断")
        return True, max_used_gb, None, ""

    # 4) 还没有可参考的占用 (第一份 checkpoint) -> 放行
    if max_used_gb <= 0:
        _log0(
            f"[DiskCheck] ({label}) 尚无历史占用参考; 放行 "
            f"(disk_avail_now={avail_gb:.1f}GB)"
        )
        return True, max_used_gb, avail_gb, ""

    required_gb = max_used_gb * margin
    allow = avail_gb >= required_gb
    _log0(
        f"[DiskCheck] ({label}) "
        f"max_checkpoint_used={max_used_gb:.1f}GB "
        f"next_checkpoint_required={required_gb:.1f}GB "
        f"disk_avail_now={avail_gb:.1f}GB "
        f"continue_training={'true' if allow else 'false'}"
    )
    return allow, max_used_gb, avail_gb, ("disk" if not allow else "")


def disk_guard_check(save_root, hf_saver, max_used_gb, margin, label=""):
    """**全体 rank 共同参与**的容量检查。

    rank0 计算 -> broadcast_object_list 同步 (stop, max_used, avail, reason)
    给所有 rank, 保证所有 rank 得到完全一致的结论。

    返回 (allow_save, max_used_gb, avail_before_gb, reason)。
    reason: "" | "disk" | "hf_failed"。
    """
    rank0 = is_rank0()
    if rank0:
        allow, max_used_gb, avail_gb, reason = eval_rank0(
            save_root, hf_saver, max_used_gb, margin, label)
    else:
        allow, avail_gb, reason = True, None, ""

    payload = [
        bool(allow),
        float(max_used_gb),
        float(avail_gb) if avail_gb is not None else -1.0,
        str(reason),
    ]
    if dist_ready():
        dist.broadcast_object_list(payload, src=0)

    allow = payload[0]
    max_used_gb = payload[1]
    avail_before = payload[2] if payload[2] >= 0 else None
    reason = payload[3]
    if (not allow) and is_rank0():
        logger.info(f"[DiskCheck] 决策: 停止训练 (reason={reason})")
    return allow, max_used_gb, avail_before, reason

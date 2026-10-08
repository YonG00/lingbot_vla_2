"""Pure, optimizer-step-safe checkpoint decisions for Auto Learning.

The milestone counter is *ever newly passed non-Bootstrap task IDs*, not
current PASS cardinality.  REOPEN/REPASS does not create another milestone.
"""
from __future__ import annotations


def learned_nonbootstrap_count(state) -> int:
    """Count distinct first PASS tasks after Bootstrap (training or Rescan)."""
    bootstrap = set(getattr(state, "bootstrap_passed", ()) or ())
    training = getattr(state, "newly_passed", ()) or ()
    # Old DCP cannot distinguish Bootstrap from Rescan in auto_passed.
    # Only count explicitly trained first passes in that case.
    automatic = ((getattr(state, "auto_passed", ()) or ())
                 if getattr(state, "bootstrap_passed_available", True) else ())
    return len((set(training) | set(automatic)) - bootstrap)


def pass_milestone_index(state, interval: int) -> int:
    if interval < 1:
        raise ValueError("HF PASS milestone interval must be a positive integer")
    return learned_nonbootstrap_count(state) // interval


def milestone_action(
    *, observed_index: int, committed_index: int, final_due: bool,
    dcp_due: bool,
) -> str:
    """Return 'none', 'hf' or 'covered_by_dcp' (final DCP also covers it).

    Caller advances committed_index after *successful* HF export or DCP save.
    """
    if observed_index <= committed_index:
        return "none"
    return "covered_by_dcp" if final_due or dcp_due else "hf"

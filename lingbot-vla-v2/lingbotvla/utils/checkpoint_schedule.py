"""Pure CPU checkpoint predicate shared by trainer and schedule contracts."""


def checkpoint_due(step, interval, explicit_steps=None, disabled=False):
    if disabled:return False
    if explicit_steps is not None:return step in explicit_steps
    return bool(interval and step % interval==0)

"""Names for newly created runs; historical paths and source checkpoint paths are untouched."""
from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo


def dated_run_path(value, now=None):
    path=Path(value)
    if re.match(r'^20\d{6}_\d{4}(?:_|$)',path.name):return str(path)
    stamp=(now or datetime.now(ZoneInfo('Asia/Shanghai'))).strftime('%Y%m%d_%H%M')
    name=re.sub(r'_20\d{6}(?:_\d{4,6})?$','',path.name)
    return str(path.with_name(stamp+'_'+name))

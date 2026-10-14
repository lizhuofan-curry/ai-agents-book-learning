"""本地JSON快照：写临时文件后替换，短暂文件错误只重试写盘，不重试API。"""

import errno
import json
import os
from pathlib import Path
import time
import uuid


def atomic_json_dump(path, value, *, replace=None, pause=None):
    path = Path(path)
    replace = replace or os.replace
    pause = pause or time.sleep
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')
    retryable = {errno.EINVAL, errno.EACCES, errno.EBUSY, errno.EPERM}
    for attempt in range(3):
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        written = False
        try:
            # 与目标同目录，先完整关闭临时文件，再替换旧快照。
            with temporary.open('xb') as file:
                file.write(payload)
            written = True
            replace(temporary, path)
            return
        except OSError as error:
            can_retry = (error.errno in retryable or getattr(error, 'winerror', None) in (5, 32, 33))
            if can_retry and attempt < 2:
                temporary.unlink(missing_ok=True)
                pause(0.1 * (attempt + 1))
                continue
            # 替换仍失败时保留已写好的待提交快照，避免丢掉刚收到的API响应。
            if written and temporary.exists():
                error.pending_record_path = str(temporary)
                error.add_note(f'旧记录保留；未提交的完整快照：{temporary}')
            else:
                temporary.unlink(missing_ok=True)
            raise

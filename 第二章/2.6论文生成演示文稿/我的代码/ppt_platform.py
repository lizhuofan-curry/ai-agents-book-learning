"""PPT工具链的跨平台程序发现；Mac记录与历史Windows配置分别保存。"""
import os
from pathlib import Path
import shutil
import sys


def venv_python(project):
    return project / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def config_file(root):
    if sys.platform == 'darwin':
        return root.parents[1] / '.uv-cache/platform/macos/ppt-toolchain/config.json'
    return root / '工具链配置.json'


def check_file(root):
    return config_file(root).with_name('check.json') if sys.platform == 'darwin' else root / '工具链检查.json'


def program(name, candidates=()):
    found = shutil.which(name)
    if found:
        return str(Path(found).resolve())
    if sys.platform == 'darwin':
        candidates = [Path('/opt/homebrew/bin') / name, Path('/usr/local/bin') / name, *candidates]
        if name == 'soffice':
            candidates = [Path('/Applications/LibreOffice.app/Contents/MacOS/soffice'), *candidates]
    return next((str(p.resolve()) for p in candidates
                 if p.is_file() and os.access(p, os.X_OK)
                 and (os.name == 'nt' or p.suffix.lower() != '.exe')), None)

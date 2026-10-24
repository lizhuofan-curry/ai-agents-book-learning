"""给当前项目虚拟环境设置可识别名称；保留环境路径和已安装依赖。"""
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    environment = ROOT / '.venv'
    if Path(sys.prefix).resolve() != environment.resolve():
        raise SystemExit('请使用本项目 .venv 中的 Python 运行此脚本。')
    platform = 'Mac' if sys.platform == 'darwin' else 'Windows' if sys.platform == 'win32' else 'Linux'
    label = f'{ROOT.name}-{platform}-Python{sys.version_info.major}.{sys.version_info.minor}'
    config = environment / 'pyvenv.cfg'
    content = config.read_text(encoding='utf-8')
    match = re.search(r'^prompt\s*=\s*(.*?)\s*$', content, flags=re.MULTILINE)
    if match:
        previous = match.group(1)
        content = content[:match.start()] + 'prompt = ' + label + content[match.end():]
    else:
        previous = None
        content = content.rstrip() + '\nprompt = ' + label + '\n'
    config.write_text(content, encoding='utf-8')
    if previous and previous != label:
        scripts = environment / ('Scripts' if sys.platform == 'win32' else 'bin')
        for name in ('activate', 'activate.fish', 'activate.csh', 'activate.nu', 'activate.bat', 'Activate.ps1'):
            path = scripts / name
            if path.is_file():
                path.write_text(path.read_text(encoding='utf-8').replace(previous, label), encoding='utf-8')
    print('环境名称：' + label)
    print('解释器：' + sys.executable)


if __name__ == '__main__':
    main()

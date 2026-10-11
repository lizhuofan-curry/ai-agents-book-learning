"""由用户运行：补齐共用Python环境、本实验Node包及浏览器，不调用模型。

LibreOffice使用官网的可视化安装流程，明确选择E盘；本脚本不静默安装系统软件。
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PYTHON = PROJECT / '.venv' / 'Scripts' / 'python.exe'
TOOLS = ROOT / '工具'
CONFIG_PATH = ROOT / '工具链配置.json'


def program(name, candidates=()):
    found = shutil.which(name)
    if found:
        return str(Path(found).resolve())
    return next((str(path.resolve()) for path in candidates if path.is_file()), None)


def make_config():
    user_dir = Path.home()
    programs = {
        'uv': program('uv', [Path(r'E:\Hermes\bin\uv.exe')]),
        'node': program('node', [Path(r'E:\ai\node.exe')]),
        'pdftoppm': program('pdftoppm', [
            user_dir / '.cache' / 'codex-runtimes' / 'codex-primary-runtime' /
            'dependencies' / 'native' / 'poppler' / 'Library' / 'bin' / 'pdftoppm.exe',
        ]),
        'soffice': program('soffice', [
            Path(r'E:\LibreOffice\program\soffice.exe'),
            TOOLS / 'LibreOffice' / 'program' / 'soffice.exe',
            Path(r'C:\Program Files\LibreOffice\program\soffice.exe'),
            Path(r'C:\Program Files (x86)\LibreOffice\program\soffice.exe'),
        ]),
    }
    cache = PROJECT / '.uv-cache'
    environment = {
        'UV_CACHE_DIR': str(cache), 'NPM_CONFIG_CACHE': str(cache / 'npm'),
        'PLAYWRIGHT_BROWSERS_PATH': str(TOOLS / 'ms-playwright'),
        'TEMP': str(TOOLS / 'temp'), 'TMP': str(TOOLS / 'temp'),
        'TMPDIR': str(TOOLS / 'temp'),
        'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
    }
    for key in ('UV_CACHE_DIR', 'NPM_CONFIG_CACHE', 'PLAYWRIGHT_BROWSERS_PATH', 'TEMP'):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    return {'prepared_at': datetime.now().isoformat(timespec='seconds'),
            'python': str(PYTHON), 'programs': programs, 'environment': environment,
            'steps_completed': [], 'failure': None, 'toolchain_ready': False}


def child_environment(config):
    environment = os.environ.copy()
    environment.update(config['environment'])
    directories = [str(PYTHON.parent)]
    directories.extend(str(Path(path).parent) for path in config['programs'].values() if path)
    environment['PATH'] = os.pathsep.join(directories + [environment.get('PATH', '')])
    return environment


def npm_cli_path(node):
    candidates = [Path(node).parent / 'node_modules' / 'npm' / 'bin' / 'npm-cli.js']
    npm = shutil.which('npm')
    if npm:
        candidates.append(Path(npm).parent / 'node_modules' / 'npm' / 'bin' / 'npm-cli.js')
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise RuntimeError('未找到npm-cli.js，请反馈Node/npm安装位置。')
    return str(found)


def main():
    if os.name != 'nt':
        raise SystemExit('此安装入口仅用于当前Windows项目。')
    config = make_config()

    def save():
        CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    def run_step(label, command, cwd):
        print('\n' + label, flush=True)
        # 直接传递参数，不拼接shell命令；日志中只记录本脚本的安装参数。
        subprocess.run(command, cwd=cwd, env=child_environment(config), check=True)
        config['steps_completed'].append(label)
        save()

    save()
    try:
        if not PYTHON.is_file() or not config['programs']['uv'] or not config['programs']['node']:
            raise RuntimeError('未找到共用Python、uv或Node，请反馈输出。')
        if not (ROOT / '准备记录.json').is_file():
            raise RuntimeError('请先完成01材料准备步骤。')
        prepared = json.loads((ROOT / '准备记录.json').read_text(encoding='utf-8'))
        if not prepared.get('ready'):
            raise RuntimeError('材料准备记录尚未就绪。')

        run_step('1/4：补齐根目录Python环境，并更新pyproject.toml与uv.lock。', [
            config['programs']['uv'], 'add', '--group', 'chapter2',
            'python-pptx>=1.0,<2', 'defusedxml>=0.7,<1',
        ], PROJECT)
        run_step('2/4：安装本实验Node依赖，保存精确版本与package-lock.json。', [
            config['programs']['node'], npm_cli_path(config['programs']['node']),
            'install', '--save-exact', '--no-audit', '--no-fund',
            'pptxgenjs@4', 'playwright@1', 'sharp@0',
            'react@19', 'react-dom@19', 'react-icons@5',
        ], ROOT)
        run_step('3/4：下载用于预览渲染的Chromium无界面浏览器到E盘。', [
            config['programs']['node'], str(ROOT / 'node_modules' / 'playwright' / 'cli.js'),
            'install', 'chromium', '--only-shell',
        ], ROOT)
        print('\n4/4：检查导入、官方JS加载、浏览器启动及预览工具路径。', flush=True)
        if not config['programs']['soffice']:
            print('尚未找到LibreOffice。请从官网下载安装，选择“自定义”，', flush=True)
            print(r'把安装位置设为 E:\LibreOffice，然后运行03_检查PPT工具链.py。', flush=True)
            print('官网：https://www.libreoffice.org/download/download-libreoffice/', flush=True)
        if not config['programs']['pdftoppm']:
            print('未找到可复用的pdftoppm，请反馈输出后处理。', flush=True)
        process = subprocess.run([
            str(PYTHON), str(Path(__file__).with_name('03_检查PPT工具链.py')),
        ], cwd=ROOT, env=child_environment(config))
        # 检查入口可能新发现LibreOffice路径，保留它写回的程序位置。
        checked_config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        config['programs'] = checked_config['programs']
        config['toolchain_ready'] = process.returncode == 0
        if not config['toolchain_ready']:
            config['failure'] = '工具链检查未全部通过，详见工具链检查.json；已完成安装步骤可复用。'
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
        config['failure'] = f'{type(exc).__name__}: {exc}'
        print('安装未全部完成：', config['failure'], flush=True)
    finally:
        save()
        print('工具链配置：', CONFIG_PATH, flush=True)
    if not config['toolchain_ready']:
        print('PPT_TOOLCHAIN_INCOMPLETE：保留已完成步骤；依赖未齐，不代表材料下载失败。')
        raise SystemExit(1)
    print('PPT_TOOLCHAIN_READY：工具链检查通过；尚未运行Agent或生成PPT。')


if __name__ == '__main__':
    main()

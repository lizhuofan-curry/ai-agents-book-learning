"""由用户运行：依赖检查；可在手动安装LibreOffice后单独重跑，不重复安装。"""

import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from ppt_platform import check_file, config_file, program


def main():
    record = {'checked_at': datetime.now().isoformat(timespec='seconds'),
              'python': sys.executable, 'python_imports': {}, 'node': None,
              'programs': {}, 'issues': [], 'ready': False,
              'scope': '检查导入、JS加载、浏览器启动与工具路径；未验证论文PPT或Agent登录。'}
    config_path = config_file(ROOT)
    config = None
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
        expected_prefix = ROOT.parents[1] / '.venv'
        if Path(sys.prefix).resolve() != expected_prefix.resolve():
            record['issues'].append('当前不是项目共用虚拟环境，请使用根目录.venv解释器运行。')
        environment = os.environ.copy()
        environment.update(config['environment'])
        for name, candidates in {
            'soffice': [Path(r'E:\LibreOffice\program\soffice.exe'),
                        ROOT / '工具' / 'LibreOffice' / 'program' / 'soffice.exe',
                        Path(r'C:\Program Files\LibreOffice\program\soffice.exe'),
                        Path(r'C:\Program Files (x86)\LibreOffice\program\soffice.exe')],
            'pdftoppm': [Path.home() / '.cache' / 'codex-runtimes' / 'codex-primary-runtime' /
                         'dependencies' / 'native' / 'poppler' / 'Library' / 'bin' / 'pdftoppm.exe'],
        }.items():
            saved = config['programs'].get(name)
            found = saved if saved and Path(saved).is_file() and (os.name == 'nt' or Path(saved).suffix.lower() != '.exe') else program(name, candidates)
            record['programs'][name] = found
            config['programs'][name] = found
            if not found:
                record['issues'].append('未找到程序：' + name)
            else:
                try:
                    flag = '--version' if name == 'soffice' else '-v'
                    probe = subprocess.run([found, flag], env=environment, capture_output=True,
                                           text=True, encoding='utf-8', errors='replace', timeout=30)
                    record.setdefault('program_versions', {})[name] = (probe.stdout + probe.stderr).strip()[:500]
                    if probe.returncode != 0:
                        record['issues'].append('程序版本检查失败：' + name)
                except (OSError, subprocess.SubprocessError) as exc:
                    record['issues'].append('程序不能执行：' + name + '；' + type(exc).__name__)
        for name in ('pptx', 'defusedxml', 'PIL', 'pypdf'):
            try:
                module = importlib.import_module(name)
                record['python_imports'][name] = {'ok': True, 'version': getattr(module, '__version__', None)}
            except Exception as exc:
                record['python_imports'][name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
                record['issues'].append('Python导入失败：' + name)
        node = config['programs'].get('node')
        if not node or not Path(node).is_file():
            record['issues'].append('未找到Node')
        else:
            try:
                process = subprocess.run([
                    node, str(Path(__file__).with_name('check_pptx_toolchain.cjs')),
                ], cwd=ROOT, env=environment, capture_output=True, text=True,
                   encoding='utf-8', errors='replace', timeout=90)
                if process.returncode != 0:
                    record['node'] = {'ok': False, 'error': process.stderr[-4000:]}
                    record['issues'].append('Node/浏览器检查未通过')
                else:
                    record['node'] = {'ok': True, 'details': json.loads(process.stdout)}
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                record['node'] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
                record['issues'].append('Node/浏览器检查异常')
        record['ready'] = not record['issues']
    except (OSError, ValueError, KeyError) as exc:
        record['issues'].append(f'配置读取失败：{type(exc).__name__}: {exc}')
    if config is not None:
        config['toolchain_ready'] = record['ready']
        if record['ready']:
            config['failure'] = None
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    check_path = check_file(ROOT)
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if not record['ready']:
        print('PPT_TOOLCHAIN_INCOMPLETE：工具链未齐，具体问题见issues；不需要重下论文和Skill。')
        raise SystemExit(1)
    print('PPT_TOOLCHAIN_READY：依赖检查通过；未运行Agent、未生成或验收PPT。')


if __name__ == '__main__':
    main()

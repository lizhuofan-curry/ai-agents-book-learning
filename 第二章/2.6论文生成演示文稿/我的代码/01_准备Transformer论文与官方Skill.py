"""实验2-6准备阶段：由用户运行，下载并核验论文与固定版本官方PPTX Skill。

所有文件存到本实验目录；不调用模型、不安装依赖、不生成PPT。
"""

import hashlib
import importlib.util
import json
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = '69c0b1a0674149f27b61b2635f935524b6add202'
PAPER_URL = 'https://arxiv.org/pdf/1706.03762'
PAPER_SHA256 = 'bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697'
ARCHIVE_URL = f'https://codeload.github.com/anthropics/skills/zip/{REVISION}'
REQUIRED_FILES = ('SKILL.md', 'html2pptx.md', 'scripts/html2pptx.js', 'scripts/thumbnail.py')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside_experiment(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f'目标不在本实验目录内：{path}')
    return resolved


def download(url, target):
    target = inside_experiment(target)
    if target.is_file():
        print('复用已有文件：', target, flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = inside_experiment(target.with_suffix(target.suffix + '.download'))
    print('正在下载：', url, flush=True)
    size = 0
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with partial.open('wb') as destination:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    destination.write(chunk)
                    size += len(chunk)
                    print(f'  已接收 {size / 1024 / 1024:.1f} MB', flush=True)
    partial.replace(target)
    print('下载完成：', target, flush=True)


def skill_relative_path(member_name):
    """只接收固定版本存档里的pptx文件；拒绝路径穿越，不使用extractall。"""
    prefix = f'skills-{REVISION}/skills/pptx/'
    if not member_name.startswith(prefix):
        return None
    relative = PurePosixPath(member_name[len(prefix):])
    if relative.is_absolute() or any(part == '..' or '\\' in part or ':' in part for part in relative.parts):
        raise ValueError('存档含不安全的文件路径')
    return relative if relative.parts else None


def unpack_official_skill(archive, skill_dir):
    file_hashes = {}
    with ZipFile(archive) as source:
        for info in source.infolist():
            relative = skill_relative_path(info.filename)
            if relative is None or info.is_dir():
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('不接受存档中的符号链接')
            target = inside_experiment(skill_dir.joinpath(*relative.parts))
            data = source.read(info)
            digest = hashlib.sha256(data).hexdigest()
            if target.exists():
                if not target.is_file() or sha256(target) != digest:
                    raise RuntimeError(f'已有Skill文件与固定版本不一致，请先保留修改再处理：{target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            file_hashes[str(relative)] = digest
    for required in REQUIRED_FILES:
        if required not in file_hashes:
            raise RuntimeError(f'官方Skill缺少文件：{required}')
    return file_hashes


def find_program(name, extra_paths=()):
    found = shutil.which(name)
    if found:
        return str(Path(found).resolve())
    for candidate in extra_paths:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def check_environment():
    user_dir = Path.home()
    programs = {
        'kimi': find_program('kimi', [user_dir / '.kimi-code' / 'bin' / 'kimi.exe']),
        'claude': find_program('claude', [user_dir / '.local' / 'bin' / 'claude.exe']),
        'node': find_program('node'),
        'npm': find_program('npm'),
        'pdftoppm': find_program('pdftoppm'),
        'soffice': find_program('soffice', [
            Path(r'C:\Program Files\LibreOffice\program\soffice.exe'),
            Path(r'C:\Program Files (x86)\LibreOffice\program\soffice.exe'),
            Path(r'E:\LibreOffice\program\soffice.exe'),
        ]),
    }
    python_packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ('requests', 'pypdf', 'PIL', 'pptx', 'defusedxml')
    }
    node_packages = None
    node_error = None
    if programs['node']:
        # 只定位包，不安装包，也不执行包代码。查询位置为实验目录。
        code = (
            'const r={}; for (const n of ["pptxgenjs","playwright","sharp"]) '
            '{try {r[n]=require.resolve(n)} catch {r[n]=null}} '
            'process.stdout.write(JSON.stringify(r))'
        )
        try:
            process = subprocess.run([programs['node'], '-e', code], cwd=ROOT,
                                     capture_output=True, text=True, encoding='utf-8',
                                     errors='replace', timeout=20, check=True)
            node_packages = json.loads(process.stdout)
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            node_error = f'{type(exc).__name__}: {exc}'
    return {'python': sys.executable, 'programs': programs,
            'python_packages': python_packages, 'node_packages_from_experiment': node_packages,
            'node_check_error': node_error,
            'notes': '仅检查程序路径与包可发现性，未验证登录、浏览器安装、模型调用或生成/渲染。'}


def main():
    materials = inside_experiment(ROOT / '材料')
    paper = materials / 'attention-is-all-you-need.pdf'
    archive = materials / f'anthropics-skills-{REVISION}.zip'
    skill_dir = inside_experiment(ROOT / 'skills' / 'pptx')
    receipt_path = ROOT / '准备记录.json'
    receipt = {'prepared_at': datetime.now().isoformat(timespec='seconds'),
               'stage': '准备论文和Skill，尚未生成PPT', 'ready': False,
               'paper': None, 'official_skill': None, 'environment': None, 'failure': None}
    try:
        print('第一步：准备Transformer原论文（所有下载保存在E盘本实验目录）。', flush=True)
        download(PAPER_URL, paper)
        digest = sha256(paper)
        if digest != PAPER_SHA256:
            raise RuntimeError(f'论文SHA256与仓库固定版本不一致：实际{digest}；预期{PAPER_SHA256}。'
                               '已保留下载文件，请反馈此信息，不自动放宽校验。')
        receipt['paper'] = {'title': 'Attention Is All You Need', 'url': PAPER_URL,
                            'path': str(paper), 'sha256': digest, 'hash_verified': True}
        print('论文哈希核验通过。', flush=True)

        print('第二步：准备固定版本的Anthropic官方PPTX Skill。', flush=True)
        download(ARCHIVE_URL, archive)
        file_hashes = unpack_official_skill(archive, skill_dir)
        receipt['official_skill'] = {
            'repository': 'https://github.com/anthropics/skills', 'revision': REVISION,
            'archive_url': ARCHIVE_URL, 'archive_sha256': sha256(archive),
            'skill_path': str(skill_dir), 'skills_dir': str(skill_dir.parent),
            'file_sha256': file_hashes,
            'verification': '按固定revision的GitHub存档提取，文件内容未改写；不是git工作树。',
        }
        print(f'官方Skill准备完成，共{len(file_hashes)}个文件。', flush=True)
        print('第三步：检查现有环境（不安装依赖、不读取密钥）。', flush=True)
        receipt['environment'] = check_environment()
        for name, path in receipt['environment']['programs'].items():
            print(f'  程序 {name}: {path or "未找到"}', flush=True)
        for name, present in receipt['environment']['python_packages'].items():
            print(f'  Python包 {name}: {"可发现" if present else "未找到"}', flush=True)
        print('  Node包：', receipt['environment']['node_packages_from_experiment'], flush=True)
        receipt['ready'] = True  # 仅指论文+Skill材料就绪，运行/渲染依赖可能仍需补齐。
    except Exception as exc:
        receipt['failure'] = f'{type(exc).__name__}: {exc}'
        print('准备未完成：', receipt['failure'], flush=True)
    finally:
        inside_experiment(receipt_path).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        print('准备记录：', receipt_path, flush=True)
    if not receipt['ready']:
        print('PAPER_AND_SKILL_INCOMPLETE：已保留准备记录，请反馈输出。')
        raise SystemExit(1)
    print('论文位置：', paper)
    print('Skill位置：', skill_dir)
    print('PAPER_AND_SKILL_READY：输入材料就绪；不代表依赖齐全或PPT实验完成。')


if __name__ == '__main__':
    main()

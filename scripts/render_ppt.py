"""用当前系统工具渲染已有PPT，输出到新目录，保留原Windows渲染记录。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / '第二章/2.6论文生成演示文稿'
sys.path.insert(0, str(LAB / '我的代码'))
from ppt_platform import config_file, program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='新的输出目录；不得覆盖旧预览')
    parser.add_argument('--dpi', type=int, default=100)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_file() or source.suffix.lower() != '.pptx':
        raise ValueError('请选择已有PPTX文件')
    if not 50 <= args.dpi <= 300:
        raise ValueError('dpi须在50至300之间')
    if output.exists():
        raise ValueError('输出目录已经存在，请选择新目录以保留既有证据')
    environment = os.environ.copy()
    if config_file(LAB).is_file():
        environment.update(json.loads(config_file(LAB).read_text())['environment'])
    tools = {name: program(name) for name in ('soffice', 'pdftoppm')}
    if not all(tools.values()):
        raise RuntimeError('需要LibreOffice和Poppler；先运行PPT工具链检查')
    from pptx import Presentation
    from pypdf import PdfReader
    pages = len(Presentation(source).slides)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    profile = output / 'libreoffice-profile'
    commands = [
        [tools['soffice'], '-env:UserInstallation=' + profile.as_uri(), '--headless',
         '--convert-to', 'pdf', '--outdir', str(output), str(source)],
        [tools['pdftoppm'], '-png', '-r', str(args.dpi),
         str(output / (source.stem + '.pdf')), str(output / 'slide')],
    ]
    receipts = []
    for number, command in enumerate(commands, 1):
        process = subprocess.run(command, env=environment, capture_output=True, text=True,
                                 encoding='utf-8', errors='replace', timeout=240)
        receipts.append({'command': command, 'exit_code': process.returncode,
                         'output': process.stdout + process.stderr})
        (output / 'execution.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n')
        if process.returncode != 0:
            raise RuntimeError(f'渲染第{number}步失败，详见{output / "execution.json"}')
        print(f'渲染步骤{number}/2完成', flush=True)
    pdf_pages = len(PdfReader(output / (source.stem + '.pdf')).pages)
    image_count = len(list(output.glob('slide-*.png')))
    if pages != pdf_pages or pages != image_count or hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError('页数或原PPT哈希检查失败')
    result = {'status': 'ppt_rendered', 'source': str(source), 'source_sha256': source_hash,
              'slide_count': pages, 'pdf_pages': pdf_pages, 'png_count': image_count,
              'source_unchanged': True, 'visual_review': 'not_performed'}
    (output / 'render_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

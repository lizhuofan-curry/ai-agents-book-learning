"""由用户运行：复用07已读正文，补两份官方章节；最多2次网页GET，0次搜索/模型请求。"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import requests
from safe_records import atomic_json_dump

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DEFAULT_RUN = ROOT / '运行结果' / 'deepseek_online_20261014_183831_601088'
EXTRA_PAGES = [
    ('S2', 'https://docs.python.org/3.11/tutorial/errors.html',
     'raising-and-handling-multiple-unrelated-exceptions'),
    ('S3', 'https://docs.python.org/3.11/reference/compound_stmts.html', 'except-star'),
]
spec = importlib.util.spec_from_file_location('official_section_reader', CODE_DIR / '03_读取官方网页正文.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reuse_first_source(run_dir, directory):
    """验证已有响应与快照，不重新搜索、重答或改写旧结果。"""
    original_path = run_dir / 'record.json'
    recovery_path = run_dir / '恢复记录.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    recovery = json.loads(recovery_path.read_text(encoding='utf-8'))
    if not recovery.get('observed_chain_completed') or recovery.get('original_record_sha256') != sha(original_path.read_bytes()):
        raise ValueError('已有在线链路恢复记录未通过，或原记录发生变化。')
    source = deepcopy(next(s for s in original['sources'] if s['id'] == 'S1'))
    if source['url'] != 'https://docs.python.org/3.11/library/exceptions.html':
        raise ValueError('S1不是本实验约定的Python官方异常说明页。')
    # 按固定文件名在指定运行目录内读取，不跟随记录中的任意文件路径。
    raw = (run_dir / 'S1.html').read_bytes()
    text = (run_dir / 'S1.txt').read_text(encoding='utf-8')
    if sha(raw) != source['raw_html_sha256'] or sha(text.encode('utf-8')) != source['text_sha256']:
        raise ValueError('S1原始HTML或正文哈希不符。')
    (directory / 'S1.html').write_bytes(raw)
    (directory / 'S1.txt').write_bytes(text.encode('utf-8'))
    source.update({'text': text, 'section_id': 'exception-groups',
                   'raw_html_path': str(directory / 'S1.html'), 'text_path': str(directory / 'S1.txt'),
                   'reused_from': str(original_path), 'original_record_sha256': sha(original_path.read_bytes()),
                   'origin': 'reused_model_selected_and_read_page'})
    return source


def fetch_section(session, identifier, url, section_id, directory, record):
    event = {'id': identifier, 'page_url': url, 'section_id': section_id, 'status': 'requesting'}
    record['web_events'].append(event)
    record['web_get_attempts'] += 1
    atomic_json_dump(directory / 'record.json', record)
    with session.get(url, timeout=(10, 45), stream=True, allow_redirects=False) as response:
        event['http_status'] = response.status_code
        if response.status_code != 200 or 'html' not in response.headers.get('Content-Type', '').lower():
            raise ValueError(f'{identifier}没有返回HTTP200 HTML；不自动重试或替换材料。')
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > 4 * 1024 * 1024:
                raise ValueError('网页超过4MiB上限；不静默截断。')
            chunks.append(chunk)
    raw = b''.join(chunks)
    (directory / f'{identifier}.html').write_bytes(raw)
    extractor = reader.SectionText(section_id)
    extractor.feed(raw.decode('utf-8'))
    extractor.close()
    text = extractor.text()
    if len(text) < 200 or 'except*' not in text:
        raise ValueError(f'{identifier}提取章节不完整或缺少主题。')
    text_path = directory / f'{identifier}.txt'
    text_path.write_bytes(text.encode('utf-8'))
    event.update({'status': 'received', 'bytes': len(raw), 'text_chars': len(text)})
    return {'id': identifier, 'url': url + '#' + section_id, 'page_url': url,
            'section_id': section_id, 'origin': 'assistant_selected_supplementary_official_section',
            'retrieved_at': datetime.now().astimezone().isoformat(), 'text': text,
            'raw_html_path': str(directory / f'{identifier}.html'), 'text_path': str(text_path),
            'raw_html_sha256': sha(raw), 'text_sha256': sha(text.encode('utf-8'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    directory = ROOT / '在线准备记录' / ('multi_page_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True)
    record = {'stage': 'three_real_page_corpus', 'status': 'preparing',
              'created_at': datetime.now().astimezone().isoformat(), 'web_get_attempts': 0,
              'model_requests': 0, 'search_requests': 0, 'web_events': [], 'sources': [],
              'scope': 'S1复用模型实读页；S2/S3助手选定完整章节；固定材料准备，不是三页自主搜索成绩'}
    print('复用已有S1；补下载S2教程、S3语法参考。最多2次网页GET，0次搜索、0次模型调用。', flush=True)
    try:
        record['sources'].append(reuse_first_source(args.source_run.resolve(), directory))
        atomic_json_dump(directory / 'record.json', record)
        with requests.Session() as session:
            for identifier, url, section_id in EXTRA_PAGES:
                print('正在读取：', url + '#' + section_id, flush=True)
                record['sources'].append(fetch_section(session, identifier, url, section_id, directory, record))
                atomic_json_dump(directory / 'record.json', record)
        record.update({'status': 'corpus_ready', 'total_text_chars': sum(len(s['text']) for s in record['sources'])})
        for source in record['sources']:
            print(source['id'], '正文字符：', len(source['text']), '来源：', source['url'])
        print('REAL_MULTI_PAGE_CORPUS_READY：三份真实正文就绪；压缩比较尚未运行。')
    except Exception as error:
        record.update({'status': 'interrupted', 'error': {'type': type(error).__name__, 'message': str(error)}})
        print('准备中断：', record['error'])
    finally:
        atomic_json_dump(directory / 'record.json', record)
        print('材料记录：', directory / 'record.json')
    if record['status'] != 'corpus_ready':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""用户运行：读取Python官方说明页并保存真实网页；不调用搜索接口或模型。

02返回更新日志，本步骤由助手指定更相关的官方说明页，明确记录来源选择。
这不是模型自主选择网页，也不是六策略实验完成。
"""

import argparse
import hashlib
import json
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
PAGE_URL = 'https://docs.python.org/3.11/library/exceptions.html'
SOURCE_URL = PAGE_URL + '#exception-groups'
SECTION_ID = 'exception-groups'
MAX_BYTES = 4 * 1024 * 1024


class SectionText(HTMLParser):
    """只提取指定section正文，不把导航栏或脚本当作证据。"""

    def __init__(self, section_id):
        super().__init__(convert_charrefs=True)
        self.section_id = section_id
        self.section_depth = 0
        self.found = False
        self.finished = False
        self.ignored_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'section':
            if self.section_depth:
                self.section_depth += 1
            elif not self.finished and dict(attrs).get('id') == self.section_id:
                self.section_depth = 1
                self.found = True
        if not self.section_depth:
            return
        if tag in ('script', 'style'):
            self.ignored_depth += 1
        if tag in ('p', 'div', 'dt', 'dd', 'li', 'pre', 'h1', 'h2', 'h3', 'br'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if not self.section_depth:
            return
        if tag in ('script', 'style') and self.ignored_depth:
            self.ignored_depth -= 1
        if tag in ('p', 'div', 'dt', 'dd', 'li', 'pre', 'h1', 'h2', 'h3'):
            self.parts.append('\n')
        if tag == 'section':
            self.section_depth -= 1
            if self.section_depth == 0:
                self.finished = True

    def handle_data(self, data):
        if self.section_depth and not self.ignored_depth:
            self.parts.append(data)

    def text(self):
        if not self.found or not self.finished:
            raise ValueError('未找到完整的exception-groups正文区域，停止而不是猜测正文。')
        text = re.sub(r'[ \t\r\f\v]+', ' ', ''.join(self.parts))
        text = '\n'.join(line.strip() for line in text.splitlines())
        return re.sub(r'\n{3,}', '\n\n', text).strip()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def find_search_record(explicit_path=None):
    candidates = ([Path(explicit_path)] if explicit_path else
                  sorted((ROOT / '在线准备记录').glob('*/record.json'), reverse=True))
    for path in candidates:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('status') == 'results_received' and data.get('results'):
            return path.resolve(), data
    raise ValueError('未找到02的成功搜索记录，请先运行02。')


def fetch_page(session, record):
    record['web_request_attempts'] = 1
    # URL固定为公开官方页面，不对搜索结果自动执行任何指令。
    with session.get(PAGE_URL, timeout=(10, 30), stream=True, allow_redirects=False) as response:
        record['http_status'] = response.status_code
        record['content_type'] = response.headers.get('Content-Type', '')
        if response.status_code != 200:
            raise RuntimeError(f'官方网页返回HTTP {response.status_code}；不自动重试或替换材料。')
        if 'html' not in record['content_type'].lower():
            raise ValueError('官方网页未返回HTML，停止正文提取。')
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError('网页超过4MiB下载上限，停止而不是静默截断。')
            chunks.append(chunk)
        return b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-record', help='可选：指定02保存的record.json完整路径')
    args = parser.parse_args()
    directory = ROOT / '在线准备记录' / ('webpage_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True)
    record = {'stage': 'official_webpage_snapshot', 'status': 'prepared',
              'created_at': datetime.now().astimezone().isoformat(),
              'search_requests': 0, 'model_requests': 0, 'web_request_attempts': 0,
              'source_url': SOURCE_URL, 'page_url': PAGE_URL,
              'source_origin': 'assistant_selected_official_documentation',
              'selection_reason': '02只返回更新日志，助手指定官方定义页作为后续证据；不是搜索返回或模型自主选择。',
              'scope': '网页快照准备；不是自主研究Agent或六策略成绩'}
    path = directory / 'record.json'
    dump(path, record)
    print('读取官方ExceptionGroup说明页；最多1次网页请求，0次搜索、0次模型请求。', flush=True)
    print('该URL由助手指定；02返回的更新日志仍保留在原记录中。', flush=True)
    try:
        search_path, search = find_search_record(args.search_record)
        record['search_record'] = str(search_path)
        record['search_record_sha256'] = hashlib.sha256(search_path.read_bytes()).hexdigest()
        record['search_provider'] = search['provider']
        record['search_query'] = search['query']
        record['previous_search_results'] = search['results']
        dump(path, record)
        with requests.Session() as session:
            raw = fetch_page(session, record)
        raw_path = directory / 'python_exceptions.html'
        raw_path.write_bytes(raw)
        record['raw_html_path'] = str(raw_path)
        record['raw_html_sha256'] = hashlib.sha256(raw).hexdigest()
        record['downloaded_bytes'] = len(raw)
        extractor = SectionText(SECTION_ID)
        extractor.feed(raw.decode('utf-8'))
        extractor.close()
        text = extractor.text()
        if len(text) < 200 or 'ExceptionGroup' not in text or 'except*' not in text:
            raise ValueError('提取正文缺少预期主题；保留HTML，等待检查，不伪造材料。')
        text_path = directory / 'ExceptionGroup_官方正文.txt'
        # 固定LF换行，保证未来快照的文件字节与文本哈希一致。
        text_path.write_bytes(text.encode('utf-8'))
        source = {'id': 'S1', 'title': 'Python 3.11 — Exception groups',
                  'url': SOURCE_URL, 'retrieved_at': datetime.now().astimezone().isoformat(),
                  'origin': record['source_origin'], 'selected_section': SECTION_ID,
                  'text': text, 'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}
        evidence_path = directory / 'evidence.json'
        dump(evidence_path, {'sources': [source], 'scope': record['scope']})
        record.update({'status': 'webpage_saved', 'selected_text_chars': len(text),
                       'text_path': str(text_path), 'evidence_path': str(evidence_path),
                       'text_sha256': source['text_sha256']})
        print('网页来源：', SOURCE_URL)
        print('下载字节：', len(raw), '；提取正文字符：', len(text))
        print('正文预览（仅显示前300字符，文件保存完整目标章节）：\n', text[:300])
        print('正文文件：', text_path)
    except (Exception, KeyboardInterrupt) as error:
        record.update({'status': 'failed',
                       'error': {'type': type(error).__name__, 'message': str(error)}})
        print('读取未完成：', record['error'])
    finally:
        record['finished_at'] = datetime.now().astimezone().isoformat()
        dump(path, record)
        print('本次记录：', path)
    if record['status'] != 'webpage_saved':
        print('WEBPAGE_PREPARATION_INCOMPLETE：保留已有记录，不填入虚构材料。')
        raise SystemExit(1)
    print('REAL_WEBPAGE_READY：真实章节正文已保存；模型回答和压缩对照尚未执行。')


if __name__ == '__main__':
    main()

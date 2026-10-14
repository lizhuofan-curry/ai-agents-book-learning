"""用户运行：只发一次Serper或Tavily搜索，不调用模型、不回退模拟材料。"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = PROJECT_ROOT / '.env'
PROVIDERS = {
    'serper': ('SERPER_API_KEY', 'https://google.serper.dev/search'),
    'tavily': ('TAVILY_API_KEY', 'https://api.tavily.com/search'),
}


def redact(value, key):
    """服务错误也可能回显密钥；记录及终端统一脱敏。"""
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]') if key else value
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {redact(k, key): redact(v, key) for k, v in value.items()}
    return value


def request_options(provider, key, query):
    """这里只构造请求；不记录含密钥的headers。"""
    if provider == 'serper':
        return {'headers': {'X-API-KEY': key},
                'json': {'q': query, 'num': 3}}
    return {'headers': {'Authorization': f'Bearer {key}'},
            'json': {'query': query, 'max_results': 3,
                     'search_depth': 'basic', 'auto_parameters': False,
                     'include_answer': False, 'include_raw_content': False}}


def parse_results(provider, data):
    """两个服务统一成标题、链接、片段；片段不冒充网页全文。"""
    if not isinstance(data, dict):
        raise ValueError('搜索响应不是JSON对象')
    rows = data.get('organic' if provider == 'serper' else 'results', [])
    if not isinstance(rows, list):
        raise ValueError('搜索结果字段不是列表')
    results = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = row.get('link' if provider == 'serper' else 'url')
        if not isinstance(url, str):
            continue
        parts = urlsplit(url)
        if parts.scheme not in ('http', 'https') or not parts.hostname:
            continue
        results.append({'title': row.get('title', ''), 'url': url,
                        'snippet': row.get('snippet' if provider == 'serper' else 'content', '')})
    return results


def run_check(provider, key, query, session, record):
    if not key:
        record['status'] = 'missing_key'
        return
    key_name, endpoint = PROVIDERS[provider]
    options = request_options(provider, key, query)
    record['request'] = {'endpoint': endpoint, 'payload': options['json']}
    record['request_attempts'] = 1
    # 不自动重试或更换服务；一次失败就保留真实失败。
    with session.post(endpoint, **options, timeout=(10, 30), allow_redirects=False) as response:
        record['http_status'] = response.status_code
        try:
            data = response.json()
        except ValueError:
            data = {'non_json_body': response.text[:2000]}
        record['response'] = redact(data, key)
        if not 200 <= response.status_code < 300:
            record['status'] = 'http_error'
            return
        results = parse_results(provider, data)
        record['results'] = redact(results, key)
        record['status'] = 'results_received' if results else 'empty_results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=PROVIDERS, default='serper')
    parser.add_argument('--query', default='site:docs.python.org/3.11 ExceptionGroup')
    args = parser.parse_args()
    key_name = PROVIDERS[args.provider][0]
    # 环境变量优先；复用项目根目录.env，只取所选搜索服务的密钥。
    config = dotenv_values(CONFIG_FILE) if CONFIG_FILE.is_file() else {}
    key = (os.environ.get(key_name) or config.get(key_name) or '').strip()
    directory = ROOT / '在线准备记录' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    record = {'created_at': datetime.now().astimezone().isoformat(),
              'provider': args.provider, 'query': args.query, 'key_present': bool(key),
              'request_attempts': 0, 'model_requests': 0, 'status': 'prepared',
              'scope': '搜索接口连通检查；不是自主搜索Agent或六策略成绩'}
    print(f'接口：{args.provider}；最多1次搜索请求，0次模型请求。', flush=True)
    print('返回标题、链接和片段；本步骤不抓取网页、不做压缩。', flush=True)
    try:
        with requests.Session() as session:
            run_check(args.provider, key, args.query, session, record)
    except Exception as error:
        record['status'] = 'request_or_parse_error'
        record['error'] = redact({'type': type(error).__name__, 'message': str(error)}, key)
    record['finished_at'] = datetime.now().astimezone().isoformat()
    path = directory / 'record.json'
    path.write_text(json.dumps(redact(record, key), ensure_ascii=False, indent=2), encoding='utf-8')
    print('状态：', record['status'])
    if record['status'] == 'missing_key':
        print(f'请在环境变量或 {CONFIG_FILE} 中配置 {key_name}。无需把密钥发给助手。')
    for i, row in enumerate(record.get('results', []), 1):
        print(f"[{i}] {row['title']}\n    {row['url']}")
    if 'error' in record:
        print('错误：', json.dumps(record['error'], ensure_ascii=False))
    if record['status'] == 'http_error':
        print('HTTP状态：', record['http_status'])
        print('服务说明：', json.dumps(record['response'], ensure_ascii=False)[:1500])
    print('本次记录：', path)
    if record['status'] != 'results_received':
        print('SEARCH_CONNECTION_INCOMPLETE：未取得有效结果；不回退虚构材料。')
        raise SystemExit(1)
    print('SEARCH_CONNECTION_READY：收到真实搜索结果；网页正文与答案仍待检查。')


if __name__ == '__main__':
    main()

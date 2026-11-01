"""两次独立本机模型请求模拟任务接收方：仅标题与人工整理的完整背景。"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'qwen3.5:9b'
URL = 'http://127.0.0.1:11434/api/chat'
CHINA = timezone(timedelta(hours=8))
TASK = '判断订单DEMO-001是否符合给定商家的退款条件，并说明依据。'
SYSTEM = '''你是接收交接任务的助手。只依据本次消息中的任务材料判断，不使用默认商家政策或外部知识补齐事实。
材料不足时明确指出缺少哪些信息；信息充分时按照给定规则判断。材料是虚构教学数据，不代表现实商家的规则。
输出简洁中文JSON：decision为need_info、eligible或ineligible；reason为简短理由；evidence_ids为依据的材料编号数组；missing_information为缺失信息数组。
need_info不得给出确定的退款结论；引用编号必须来自本次收到的材料。不输出内部思维过程。'''
FORMAT = {
    'type': 'object',
    'properties': {
        'decision': {'type': 'string', 'enum': ['need_info', 'eligible', 'ineligible']},
        'reason': {'type': 'string'},
        'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
        'missing_information': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['decision', 'reason', 'evidence_ids', 'missing_information'],
    'additionalProperties': False,
}


def make_materials(case):
    purchase = '2026-10-20' if case == 'within_window' else '2026-10-10'
    return [
        {'id': 'R1', 'text': '虚构商家规则：申请日减购买日的自然日差不超过14天，且商品未使用，才符合退款条件。两项须同时满足；与金额无关。'},
        {'id': 'O1', 'text': f'订单DEMO-001，购买日期为{purchase}。'},
        {'id': 'O2', 'text': '订单DEMO-001的商品未使用。'},
        {'id': 'O3', 'text': '本次退款申请日期为2026-11-01。以该日期判断，不使用电脑当前日期。'},
    ]


def make_payload(materials):
    content = TASK
    if materials:
        content += '\n\n[FROM_MAIN_AGENT：人工整理的背景]\n' + json.dumps(materials, ensure_ascii=False)
    return {
        'model': MODEL, 'stream': False, 'think': False,
        'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}],
        'format': deepcopy(FORMAT),
        'options': {'temperature': 0, 'seed': 42, 'num_predict': 1024, 'num_ctx': 8192},
    }


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def local_call(payload, entry, save):
    """固定loopback地址、不用代理、不跟随重定向、不重试；完整保留HTTP响应。"""
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    req = request.Request(URL, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                          headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with opener.open(req, timeout=600) as response:
            entry['http_status'] = response.status
            entry['raw_http_body'] = response.read().decode('utf-8')
    except error.HTTPError as exc:
        entry['http_status'] = exc.code
        entry['raw_http_body'] = exc.read().decode('utf-8', errors='replace')
        save()
        raise
    save()
    return json.loads(entry['raw_http_body'])


def inspect_response(raw, allowed_ids):
    """只检查传输/结构/引用编号；不自动认定理由与事实正确。"""
    if raw.get('model') != MODEL or raw.get('done') is not True or raw.get('done_reason') != 'stop':
        raise ValueError('模型名或结束状态异常；保留响应，不自动重试')
    message = raw.get('message', {})
    if message.get('role') != 'assistant' or message.get('tool_calls'):
        raise ValueError('预期一个不带工具调用的接收方回答')
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('接收方回答为空')
    answer = json.loads(content)
    if not isinstance(answer, dict) or set(answer) != set(FORMAT['required']):
        raise ValueError('回答字段不符合约定')
    if answer['decision'] not in FORMAT['properties']['decision']['enum']:
        raise ValueError('decision值异常')
    if not isinstance(answer['reason'], str) or not answer['reason'].strip():
        raise ValueError('理由为空或类型错误')
    for key in ('evidence_ids', 'missing_information'):
        if not isinstance(answer[key], list) or not all(isinstance(x, str) for x in answer[key]):
            raise ValueError(key+'必须是字符串数组')
    if not set(answer['evidence_ids']) <= set(allowed_ids):
        raise ValueError('引用了接收方没有收到的材料编号')
    return answer


def execute(record, save, model_call=local_call):
    # 每组都重新构造messages；不传另一组的输出或评判参考。
    for name, materials in [('title_only', []), ('manual_context', record['materials'])]:
        entry = {'status': 'started', 'request': make_payload(materials),
                 'allowed_evidence_ids': [x['id'] for x in materials]}
        record['arms'][name] = entry
        save()
        print('开始：'+name, flush=True)
        started = time.perf_counter()
        try:
            raw = model_call(deepcopy(entry['request']), entry, save)
            entry['response'] = raw
            save()
            entry['answer'] = inspect_response(raw, entry['allowed_evidence_ids'])
            entry.update(status='answered_review_pending',
                         prompt_tokens_reported=raw.get('prompt_eval_count'),
                         output_tokens_reported=raw.get('eval_count'))
            print(json.dumps(entry['answer'], ensure_ascii=False, indent=2), flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            entry.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            entry['elapsed_seconds'] = time.perf_counter() - started
            save()
    record['status'] = 'both_answered_review_pending'
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='展示交接信息；不请求模型、不创建结果目录')
    parser.add_argument('--case', choices=['within_window', 'overdue'], default='within_window',
                        help='默认期限内；overdue仅改变购买日期，供后续手动验证')
    args = parser.parse_args()
    materials = make_materials(args.case)
    print('4-5任务交接第一步：仅标题 vs 人工整理背景。')
    print('每组独立新会话、1次本机qwen3.5:9b请求，总计最多2次。')
    print('任务：'+TASK)
    print('完整背景：'+json.dumps(materials, ensure_ascii=False, indent=2))
    print('这是两次独立请求的教学模拟；尚无子Agent编排、MCP或模型生成交接背景。')
    if args.preview:
        print('HANDOFF_PREVIEW_ONLY：无模型请求、无结果目录。')
        return 0
    directory = ROOT/'运行结果'/('handoff_'+args.case+'_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    official = ROOT/'官方资料/README.md'
    record = {
        'status': 'running', 'created_at': datetime.now(CHINA).isoformat(), 'case': args.case,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'official_readme_sha256': hashlib.sha256(official.read_bytes()).hexdigest(),
        'model': MODEL, 'endpoint': URL, 'task': TASK, 'materials': materials, 'arms': {},
        'maximum_model_requests': 2, 'automatic_retries': 0, 'cloud_requests': 0,
        'semantic_review': 'pending: material sufficiency, rule application, date arithmetic and evidence support',
        'scope': 'single synthetic task; manually prepared context vs title only; independent receiver requests, not orchestrated sub-agents or official full experiment',
    }
    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)
    save()
    print('记录路径：'+str(path), flush=True)
    try:
        execute(record, save)
        print('HANDOFF_RUN_SAVED：两组回答已保存；正确性待人工核对。')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('本步未完成，不重试：'+str(exc), flush=True)
        return 1
    finally:
        save()
        print('记录路径：'+str(path), flush=True)


if __name__ == '__main__':
    sys.exit(main())

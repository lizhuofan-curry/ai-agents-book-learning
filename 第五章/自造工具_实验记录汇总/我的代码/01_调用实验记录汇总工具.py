"""通过函数注册表调用只读汇总工具；0次模型/MCP/网络请求。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from 实验记录工具 import ROOT, SOURCE_ROOT, TOOL_DEFINITION, dispatch_tool, source_path

DEFAULT_RECORD = SOURCE_ROOT/'local_coding_revision_20261101_162450_914560/record.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', default=str(DEFAULT_RECORD), help='要汇总的本机编码record.json绝对路径')
    parser.add_argument('--preview', action='store_true', help='只显示定义和请求，不读取原始记录或创建结果目录')
    args = parser.parse_args()
    request = {'name': TOOL_DEFINITION['name'], 'arguments': {'record_path': args.record}}
    print('第五章W3自造工具：通过函数注册表汇总已保存的编码实验。')
    print('0次模型、MCP或网络请求；只读原记录，本次调用结果另存。')
    if args.preview:
        print(json.dumps({'definition': TOOL_DEFINITION, 'request': request}, ensure_ascii=False, indent=2))
        print('RECORD_SUMMARY_PREVIEW_ONLY：未调用工具、未创建结果目录。')
        return 0
    folder = ROOT/'运行结果'/('record_summary_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    path = folder/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tool_module_sha256': hashlib.sha256(Path(__file__).with_name('实验记录工具.py').read_bytes()).hexdigest(),
              'tool_definition': TOOL_DEFINITION, 'request': request,
              'model_requests': 0, 'mcp_requests': 0, 'network_requests': 0,
              'dispatch_attempts': 0, 'automatic_retries': 0,
              'scope': 'One direct registered function call; no model tool selection, no rerun of source experiment; not evidence of user independent authorship.'}

    def save():
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.replace(path)
    save()
    print('记录路径：' + str(path), flush=True)
    try:
        source = source_path(args.record)
        before = source.read_bytes()
        record['source_sha256_before'] = hashlib.sha256(before).hexdigest()
        record['dispatch_attempts'] = 1
        save()
        result = dispatch_tool(request['name'], request['arguments'])
        record['result'] = result
        after = source.read_bytes()
        record['checks'] = {
            'source_unchanged': before == after,
            'summary_identifies_read_source': result['source_sha256'] == record['source_sha256_before'],
            'registered_function_returned': result['summary_schema'] == 'coding_record_summary_v1',
        }
        passed = all(record['checks'].values())
        record['status'] = 'summary_call_checked_review_pending' if passed else 'finished_with_failed_checks'
        print('工具返回摘要：' + json.dumps({
            'source_status': result['source_status'],
            'model_requests': result['model_requests'],
            'tool_results': result['tool_events']['returned_results'],
            'tools': result['tool_events']['returned_results_by_tool'],
            'tool_sequence': result['tool_events']['returned_result_sequence'],
            'reported_prompt_tokens': result['usage_reported']['prompt_tokens'],
            'reported_output_tokens': result['usage_reported']['output_tokens'],
            'saved_completion_checks': {key: len(result['saved_completion_checks'][key])
                                        for key in ('passed', 'failed', 'unknown')},
            'completion_check_availability': result['saved_completion_checks']['availability'],
            'warnings': result['warnings'],
        }, ensure_ascii=False, indent=2))
        print('调用检查：' + json.dumps(record['checks'], ensure_ascii=False))
        print('RECORD_SUMMARY_SAVED：仅说明本次工具调用及原记录未改，源实验结论仍按原始检查和审阅。')
        return 0 if passed else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='failed_tool_call', error={'type': type(exc).__name__, 'message': str(exc)})
        print('工具调用失败，不重试：' + str(exc))
        return 1
    finally:
        save()
        print('记录路径：' + str(path), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

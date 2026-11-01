"""直接注册的只读工具：汇总第五章CodingAgent记录，不重新判定实验成功。"""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent/'CodingAgent入门/运行结果'
TOOL_DEFINITION = {
    'name': 'summarize_coding_record',
    'description': '只读汇总第五章本机编码记录的请求、工具事件和保存的检查状态；不调用模型、不运行原实验、不修改原记录。',
    'input_schema': {
        'type': 'object',
        'properties': {'record_path': {'type': 'string', 'description': 'CodingAgent入门/运行结果中一个local_coding_task或local_coding_revision批次的record.json绝对路径。'}},
        'required': ['record_path'],
        'additionalProperties': False,
    },
}


def source_path(record_path):
    if not isinstance(record_path, str) or not Path(record_path).is_absolute():
        raise ValueError('record_path须为绝对路径字符串')
    path = Path(record_path).resolve()
    if (not path.is_relative_to(SOURCE_ROOT.resolve()) or path.name != 'record.json'
            or path.parent.parent != SOURCE_ROOT.resolve()
            or not path.parent.name.startswith(('local_coding_task_', 'local_coding_revision_'))):
        raise ValueError('本工具仅支持CodingAgent入门/运行结果下的本机编码任务及修改批次record.json')
    return path


def checks_summary(checks):
    # 严格区分布尔true/false；空字典、缺失和字符串"true"不能算通过。
    if checks is None:
        return {'availability': 'missing', 'passed': [], 'failed': [], 'unknown': [], 'all_recorded_true': None}
    if not isinstance(checks, dict):
        raise ValueError('检查字段须为对象')
    passed = [k for k, v in checks.items() if v is True]
    failed = [k for k, v in checks.items() if v is False]
    unknown = [k for k, v in checks.items() if v is not True and v is not False]
    return {'availability': 'present' if checks else 'empty', 'passed': passed, 'failed': failed,
            'unknown': unknown, 'all_recorded_true': bool(checks) and not failed and not unknown}


def usage_summary(calls, key):
    values = [call.get(key) for call in calls]
    valid = [value for value in values if type(value) is int and value >= 0]
    missing = len(values) - len(valid)
    return {'reported_sum': sum(valid), 'calls_without_valid_count': missing,
            'coverage': 'no_requests' if not calls else ('complete' if missing == 0 else 'partial')}


def build_summary(record):
    if not isinstance(record, dict):
        raise ValueError('记录顶层须为对象')
    calls, events = record.get('model_calls'), record.get('events')
    if not isinstance(calls, list) or not all(isinstance(x, dict) for x in calls):
        raise ValueError('不支持的记录：model_calls须为对象列表')
    if not isinstance(events, list) or not all(isinstance(x, dict) for x in events):
        raise ValueError('不支持的记录：events须为对象列表')
    executions = [e for e in events if e.get('type') == 'tool_execution_complete']
    if any(not isinstance(e.get('tool'), str) or not isinstance(e.get('result'), dict) for e in executions):
        raise ValueError('工具执行完成事件缺少工具名或结果对象')
    sequence = [e['tool'] for e in executions]
    completion = checks_summary(record.get('completion_checks'))
    artifact = record.get('artifact_verification')
    if artifact is not None and not isinstance(artifact, dict):
        raise ValueError('artifact_verification须为对象')
    artifact_checks = checks_summary(artifact.get('checks') if artifact is not None else None)
    declared_budget = record.get('maximum_model_requests')
    valid_budget = type(declared_budget) is int and declared_budget >= 0
    budget_stop = any(e.get('type') == 'max_iterations_reached' for e in events)
    warnings = []
    if completion['availability'] != 'present':
        warnings.append('完成检查缺失或为空，不能据此判定实验通过。')
    if completion['failed'] or completion['unknown']:
        warnings.append('完成检查包含false或未知值，请保留失败并查看原记录。')
    if artifact_checks['all_recorded_true'] is not True:
        warnings.append('保存的产物检查未全部为true；本工具不运行代码重新验收。')
    if any(e.get('type') == 'error' for e in events) or budget_stop:
        warnings.append('存在引擎错误或预算停止事件。')
    if any(call.get('status') != 'completed' for call in calls):
        warnings.append('存在未完成或状态未知的模型请求；总数仍包含这些请求。')
    return {
        'summary_schema': 'coding_record_summary_v1',
        'source_status': record.get('status'),
        'model_requests': {
            'total_started': len(calls),
            'completed': sum(call.get('status') == 'completed' for call in calls),
            'not_completed_or_unknown': sum(call.get('status') != 'completed' for call in calls),
            'declared_maximum': declared_budget,
            'at_declared_limit': len(calls) == declared_budget if valid_budget else None,
            'exceeds_declared_limit': len(calls) > declared_budget if valid_budget else None,
            'budget_stop_event': budget_stop,
        },
        'tool_events': {
            'model_selections': sum(e.get('type') == 'tool_call' for e in events),
            'execution_starts': sum(e.get('type') == 'tool_execution_start' for e in events),
            'returned_results': len(executions),
            'returned_results_by_tool': dict(Counter(sequence)),
            'returned_result_sequence': sequence,
            'error_results': sum(bool(e['result'].get('error')) for e in executions),
            'nonzero_exit_results': sum(type(e['result'].get('exit_code')) is int
                                        and e['result']['exit_code'] != 0 for e in executions),
        },
        'usage_reported': {
            'prompt_tokens': usage_summary(calls, 'prompt_tokens_reported'),
            'output_tokens': usage_summary(calls, 'output_tokens_reported'),
            'interpretation': '模型报告值的累计；输入包含历史重复发送，不是不重复文本量或云端费用。',
        },
        'saved_completion_checks': completion,
        'saved_artifact_checks': artifact_checks,
        'agent_reported_done': any(e.get('type') == 'done' for e in events),
        'warnings': warnings,
        'scope': '只读机械汇总原记录；returned_results不等于成功工具数，检查true不等于语义审阅通过、理解已掌握或章节完成。',
    }


def summarize_coding_record(record_path):
    path = source_path(record_path)
    raw = path.read_bytes()
    summary = build_summary(json.loads(raw))
    return {'source_record': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(), **summary}


REGISTERED_TOOLS = {TOOL_DEFINITION['name']: summarize_coding_record}


def dispatch_tool(name, arguments):
    if name not in REGISTERED_TOOLS:
        raise ValueError('未注册的工具：' + str(name))
    if not isinstance(arguments, dict) or set(arguments) != {'record_path'}:
        raise ValueError('本工具参数须且仅须包含record_path')
    return REGISTERED_TOOLS[name](**arguments)

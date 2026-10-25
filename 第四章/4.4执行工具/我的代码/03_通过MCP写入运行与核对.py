"""客户端固定调用三个本地MCP执行工具；无模型请求。"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT.parents[1]
ENVIRONMENT = BOOK/'.venv-ch4-mcp'
SERVER = Path(__file__).with_name('02_执行工具SDK服务.py')
HELPER = Path(__file__).with_name('执行工具教学实现.py')
spec = importlib.util.spec_from_file_location('execution_demo_core', HELPER)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
CHINA = timezone(timedelta(hours=8))


def sdk_environment():
    if Path(sys.prefix).resolve() != ENVIRONMENT.resolve():
        python = ENVIRONMENT/'bin/python'
        if not python.is_file():
            raise RuntimeError('未找到已有第四章SDK环境')
        print('切换到已配置的SDK环境；不安装依赖。', flush=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.execv(str(python), [str(python), str(Path(__file__)), *sys.argv[1:]])
    if importlib.metadata.version('mcp') != '2.3.0':
        raise RuntimeError('本步骤固定mcp 2.3.0')


def dump(value):
    return value.model_dump(mode='json', by_alias=True, exclude_none=True)


async def exercise(workspace, record, save=lambda: None, verbose=True):
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client
    root = Path(workspace).resolve(strict=True)
    (root/'正确版本').mkdir()
    (root/'错误版本').mkdir()
    params = StdioServerParameters(command=sys.executable, args=['-u', str(SERVER)],
                                   env={'EXECUTION_TEACHING_ROOT': str(root), 'PYTHONDONTWRITEBYTECODE': '1'})
    record['calls'] = []
    with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as errors:
        try:
            async with Client(stdio_client(params, errlog=errors), read_timeout_seconds=20) as client:
                record.update(protocol_version=client.protocol_version, server_info=dump(client.server_info))
                record['catalog'] = dump(await client.list_tools())
                names = {t['name'] for t in record['catalog']['tools']}
                if names != {'file_write', 'run_python', 'verify_counts'} or len(record['catalog']['tools']) != 3:
                    raise ValueError('工具目录不符合预期')
                if verbose:
                    print('发现服务端工具：', ', '.join(sorted(names)), flush=True)
                save()

                async def invoke(label, name, args):
                    if len(record['calls']) >= 9:
                        raise ValueError('达到固定9次工具调用预算')
                    entry = {'label': label, 'name': name, 'arguments': args, 'status': 'started'}
                    record['calls'].append(entry)
                    save()
                    if verbose:
                        display = {k: v for k, v in args.items() if k != 'content'}
                        print('客户端调用：', label, name, json.dumps(display, ensure_ascii=False), flush=True)
                    raw = dump(await client.call_tool(name, args))
                    entry.update(status='completed', result=raw)
                    save()
                    blocks = [x['text'] for x in raw.get('content', []) if x.get('type') == 'text']
                    if len(blocks) != 1:
                        raise ValueError('预期恰好一条文本工具结果')
                    result = json.loads(blocks[0])
                    entry['decoded_result'] = result
                    if verbose:
                        print('服务端返回：', json.dumps(result, ensure_ascii=False), flush=True)
                    return result, raw

                async def require_write(label, variant, filename, content):
                    result, raw = await invoke(label, 'file_write', {'variant': variant, 'filename': filename, 'content': content})
                    if raw.get('isError') or result.get('success') is not True:
                        raise ValueError('预期写入未成功：'+label)
                    return result

                await require_write('准备正确版输入', 'correct', 'data.txt', tools.SAMPLE)
                await require_write('写入正确代码', 'correct', 'wordcount.py', tools.GOOD)
                rejected, rejected_raw = await invoke('观察语法拦截', 'file_write', {
                    'variant': 'correct', 'filename': 'broken.py', 'content': tools.BROKEN})
                good_run, good_run_raw = await invoke('运行正确版', 'run_python', {'variant': 'correct', 'filename': 'wordcount.py'})
                good_goal, good_goal_raw = await invoke('核对正确版', 'verify_counts', {'variant': 'correct'})
                await require_write('准备错误版输入', 'wrong', 'data.txt', tools.SAMPLE)
                await require_write('写入逻辑错误代码', 'wrong', 'wrong_count.py', tools.WRONG)
                bad_run, bad_run_raw = await invoke('运行逻辑错误版', 'run_python', {'variant': 'wrong', 'filename': 'wrong_count.py'})
                bad_goal, bad_goal_raw = await invoke('核对逻辑错误版', 'verify_counts', {'variant': 'wrong'})
                # 客户端再打开产物复核，避免只相信工具返回的success或goal_correct。
                actual_good = json.loads((root/'正确版本/counts.json').read_text())
                actual_bad = json.loads((root/'错误版本/counts.json').read_text())
                checks = {'syntax_rejection_visible_over_mcp': rejected_raw.get('isError') is True and rejected.get('error_type') == 'SyntaxError',
                          'rejected_script_not_written': not (root/'正确版本/broken.py').exists(),
                          'correct_execution_exited_zero': good_run_raw.get('isError') is False and good_run.get('returncode') == 0,
                          'correct_result_verified': good_goal_raw.get('isError') is False and good_goal.get('goal_correct') is True,
                          'wrong_execution_exited_zero': bad_run_raw.get('isError') is False and bad_run.get('returncode') == 0,
                          'wrong_goal_detected_without_protocol_error': bad_goal_raw.get('isError') is False and bad_goal.get('goal_correct') is False,
                          'client_checked_good_file': actual_good == tools.EXPECTED,
                          'client_checked_wrong_file': actual_bad == {**tools.EXPECTED, 'apple': 999}}
                record['checks'] = checks
                record['artifacts'] = {str(p.relative_to(root)): tools.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
                if not all(checks.values()):
                    raise ValueError('MCP执行与独立检查未全部满足')
                save()
        finally:
            errors.seek(0)
            record['server_stderr'] = errors.read()
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    sdk_environment()
    print('4-4第二步：通过真实MCP调用写文件、运行Python、核对结果三个执行工具。')
    print('客户端固定9次工具调用；SDK 2.3.0；无模型、网络或外部账号请求。')
    print('复用同一正确/错误词频代码，观察服务端如何返回执行状态与目标验证结果。')
    if args.preview:
        tools.sha(SERVER)
        compile(tools.GOOD, 'wordcount.py', 'exec')
        print('MCP_EXECUTION_PREVIEW_ONLY：环境与入口检查通过，未启动服务或生成工作区。')
        return 0
    if args.check:
        with tempfile.TemporaryDirectory(prefix='mcp_execution_check_') as temp:
            record = {}
            asyncio.run(asyncio.wait_for(exercise(temp, record, verbose=False), timeout=40))
            print('MCP_EXECUTION_OFFLINE_CHECK_PASSED：', record['protocol_version'], record['checks'])
        return 0
    directory = ROOT/'运行结果'/('mcp_execution_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    workspace = directory/'工作区'
    workspace.mkdir()
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
              'client_sha256': tools.sha(Path(__file__)), 'server_sha256': tools.sha(SERVER),
              'helper_sha256': tools.sha(HELPER), 'mcp_sdk_version': importlib.metadata.version('mcp'),
              'python': sys.version, 'executable': sys.executable, 'workspace': str(workspace),
              'model_requests': 0, 'network_requests': 0, 'automatic_retries': 0, 'maximum_tool_calls': 9,
              'transport': 'actual official SDK subprocess stdio; decoded messages recorded, not raw wire transcript',
              'scope': 'fixed known scripts and predetermined tool sequence; not model planning, arbitrary sandbox or full experiment'}

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    save()
    print('记录路径：', path, flush=True)
    started = time.perf_counter()
    try:
        asyncio.run(asyncio.wait_for(exercise(workspace, record, save), timeout=40))
        record['status'] = 'mcp_fixed_execution_verified'
        print('核对：', record['checks'])
        print('MCP_EXECUTION_RUN_SAVED：MCP调用及真实文件/进程结果保存。')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        def causes(exc):
            if isinstance(exc, BaseExceptionGroup):
                return [item for child in exc.exceptions for item in causes(child)]
            return [{'type': type(exc).__name__, 'message': str(exc)}]
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error), 'causes': causes(error)})
        print('本步未完成，不重试：', record['error'])
        return 1
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        save()
        print('记录路径：', path)


if __name__ == '__main__':
    raise SystemExit(main())

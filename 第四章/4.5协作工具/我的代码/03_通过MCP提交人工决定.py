"""真实stdio MCP：客户端读取人工输入，服务端保存决定并执行固定动作。"""
import argparse
import asyncio
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT.parents[1]
SDK_ENV = BOOK/'.venv-ch4-mcp'
PYTHON = SDK_ENV/'bin/python'
SERVER = Path(__file__).with_name('02_人工确认SDK服务.py')
HELPER = Path(__file__).with_name('01_终端人工确认与任务恢复.py')
spec = importlib.util.spec_from_file_location('terminal_human_input', HELPER)
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def dump(value):
    return value.model_dump(mode='json', by_alias=True, exclude_none=True)


async def exercise(record, save, directory, timeout, input_fn=None):
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable, args=['-u', str(SERVER)],
        env={'PYTHONDONTWRITEBYTECODE': '1', 'HITL_RUN_DIRECTORY': str(Path(directory).resolve())})
    with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as errors:
        try:
            async with Client(stdio_client(params, errlog=errors), read_timeout_seconds=10) as client:
                record['protocol_version'] = client.protocol_version
                record['catalog'] = dump(await client.list_tools())
                if {x['name'] for x in record['catalog']['tools']} != {'create_request','submit_decision','finish_request'}:
                    raise ValueError('工具目录不符')
                save()
                async def call(name, arguments):
                    item = {'name': name, 'arguments': arguments, 'status': 'started'}
                    record.setdefault('calls', []).append(item)
                    save()
                    result = dump(await client.call_tool(name, arguments))
                    item.update(status='returned', result=result)
                    save()
                    texts = [x['text'] for x in result.get('content', []) if x.get('type') == 'text']
                    if len(texts) != 1:
                        raise ValueError('返回格式异常')
                    observed = json.loads(texts[0])
                    print('服务端返回：', json.dumps(observed, ensure_ascii=False), flush=True)
                    return observed
                created = await call('create_request', {'timeout_seconds': timeout})
                if not created['success'] or created['status'] != 'pending':
                    raise ValueError('未成功创建请求')
                request_id = created['request_id']
                print('待决定：', created['message'])
                print('目标：', Path(directory)/created['target_filename'])
                print('客户端等待终端输入；服务端请求状态为pending。')
                print(f'请在约{created["remaining_seconds"]:.0f}秒内输入 同意/APPROVE 或 拒绝/REJECT，然后回车。', flush=True)
                human = local.new_record(max(0, created['remaining_seconds']))
                human['request_id'] = request_id
                record['human_input'] = human
                save()
                if input_fn is not None:
                    # 仅供离线检查注入；正式CLI没有自动回复选项。
                    await asyncio.to_thread(input_fn, human, save)
                elif not sys.stdin.isatty():
                    local.resolve(human, 'input_unavailable')
                else:
                    await asyncio.to_thread(local.wait_for_reply, human, save)
                decision = human['approval']['status']
                submitted = await call('submit_decision', {'request_id': request_id, 'decision': decision})
                # 超时可能由服务端先行判定；只有该拒绝是预期的终态竞争。
                if not submitted['success'] and submitted.get('current_status') != 'timeout':
                    raise ValueError('服务端未接受本次决定')
                finished = await call('finish_request', {'request_id': request_id})
                if not finished['success'] or not finished.get('finished'):
                    raise ValueError('服务端未完成动作核对')
                target = Path(directory)/created['target_filename']
                should_exist = finished['status'] == 'approved'
                if target.exists() != should_exist or finished['action']['executed'] != should_exist:
                    raise ValueError('真实文件与服务端决定不一致')
                if should_exist:
                    if hashlib.sha256(target.read_bytes()).hexdigest() != finished['action']['sha256']:
                        raise ValueError('产物哈希不一致')
                state = json.loads((Path(directory)/'server_state.json').read_text())
                if state['request']['status'] != finished['status'] or not state['request']['finished']:
                    raise ValueError('服务端落盘状态不一致')
                record.update(status='mcp_hitl_completed_'+finished['status'], server_decision=finished,
                              checks={'file_presence_matches_server_decision': True, 'server_state_matches_result': True,
                                      'request_id_preserved': human['request_id'] == finished['request_id']})
                save()
        finally:
            errors.seek(0)
            record['server_stderr'] = errors.read()
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        parser.error('--timeout必须在1到3600秒之间')
    if Path(sys.prefix).resolve() != SDK_ENV.resolve():
        if not PYTHON.is_file():
            raise RuntimeError('未找到已配置的第四章SDK环境')
        print('切换已有第四章SDK环境；不安装依赖。', flush=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__)), *sys.argv[1:]])
    if importlib.metadata.version('mcp') != '2.3.0':
        raise ValueError('需固定mcp 2.3.0')
    manifest_sha = local.check_sources(ROOT)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), HELPER, SERVER)}
    print('4-5第二步：真实MCP创建请求→终端人工回复→服务端继续或停止。')
    print('SDK 2.3.0；固定3次工具调用，无模型、外部网络、通知或自动回复。')
    if args.preview:
        print('MCP_HITL_PREVIEW_ONLY：环境及入口检查通过，未启动服务或等待输入。')
        return 0
    directory = ROOT/'运行结果'/('mcp_hitl_'+datetime.now(local.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': local.timestamp(), 'script_sha256': hashes,
              'manifest_sha256': manifest_sha, 'python': sys.version, 'executable': sys.executable,
              'mcp_sdk_version': '2.3.0', 'model_requests': 0, 'network_requests': 0,
              'notifications_sent': 0, 'automatic_retries': 0,
              'scope': 'local terminal human decision over real stdio MCP; fixed client sequence, no model decision',
              'message_record_scope': 'SDK-decoded calls and responses, not raw wire capture'}
    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)
    save()
    print('记录路径：', path, flush=True)
    try:
        asyncio.run(exercise(record, save, directory, args.timeout))
        print('MCP_HITL_RUN_SAVED：跨进程请求、人工回复及真实产物已核对。')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print('本步未完成，不重试：', record['error'])
        return 1
    finally:
        save()
        print('状态：', record['status'])
        print('记录路径：', path)


if __name__ == '__main__':
    raise SystemExit(main())

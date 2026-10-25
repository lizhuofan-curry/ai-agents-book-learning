"""同一读取任务换用官方MCP 2.3.0 SDK；无模型或网络请求。"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT.parents[1]
ENVIRONMENT = BOOK/'.venv-ch4-mcp'
PYTHON = ENVIRONMENT/'bin/python'
SERVER = Path(__file__).with_name('SDK文件读取服务.py')
HELPER = Path(__file__).with_name('文件读取服务.py')
FIXTURE = ROOT/'测试材料/工具说明.txt'
CHINA = timezone(timedelta(hours=8))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def use_sdk_environment():
    if Path(sys.prefix).resolve() != ENVIRONMENT.resolve():
        if not PYTHON.is_file():
            raise RuntimeError('未找到预配置的第四章SDK环境，请先检查环境说明')
        print('切换到已配置的第四章SDK环境；不安装或下载依赖。', flush=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__)), *sys.argv[1:]])
    if importlib.metadata.version('mcp') != '2.3.0':
        raise RuntimeError('需已固定的mcp 2.3.0环境')


def dump(value):
    return value.model_dump(mode='json', by_alias=True, exclude_none=True)


async def exercise(record, save=lambda: None):
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # SDK管理实际子进程与stdio；只传需要的环境变量，不传模型或外部账号凭据。
    params = StdioServerParameters(command=sys.executable, args=['-u', str(SERVER)],
                                   env={'PYTHONDONTWRITEBYTECODE': '1'})
    with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as errors:
        try:
            async with Client(stdio_client(params, errlog=errors), read_timeout_seconds=10) as client:
                record['protocol_version'] = client.protocol_version
                record['server_info'] = dump(client.server_info) if client.server_info else None
                record['server_capabilities'] = dump(client.server_capabilities)
                save()
                catalog = await client.list_tools()
                record['catalog'] = dump(catalog)
                save()
                if [tool.name for tool in catalog.tools] != ['file_reader']:
                    raise ValueError('目录缺失file_reader或多出工具')
                for key, filename in [('successful_call', '工具说明.txt'), ('failed_call', '不存在.txt')]:
                    arguments = {'file_path': filename}
                    record.setdefault('calls', []).append({'name': 'file_reader', 'arguments': arguments})
                    save()
                    result = await client.call_tool('file_reader', arguments)
                    record[key] = dump(result)
                    save()
                good, bad = record['successful_call'], record['failed_call']
                content = json.loads(good['content'][0]['text'])
                failure = json.loads(bad['content'][0]['text'])
                if good.get('isError') is not False or content['success'] is not True:
                    raise ValueError('正常读取失败')
                if content['message']['content'] != FIXTURE.read_text(encoding='utf-8') or content['message']['truncated']:
                    raise ValueError('返回内容与原文件不一致')
                if bad.get('isError') is not True or failure['success'] is not False:
                    raise ValueError('缺失文件未返回预期工具错误')
                if failure['metadata']['error_type'] != 'FileNotFoundError':
                    raise ValueError('缺失文件错误类型异常')
                record['checks'] = {'catalog_has_reader': True, 'content_matches_file': True,
                                    'missing_file_error_observed': True}
        finally:
            errors.seek(0)
            record['server_stderr'] = errors.read()
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    use_sdk_environment()
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
              'client_sha256': sha(Path(__file__)), 'server_sha256': sha(SERVER), 'reader_sha256': sha(HELPER),
              'fixture_sha256': sha(FIXTURE), 'mcp_sdk_version': importlib.metadata.version('mcp'),
              'python': sys.version, 'executable': sys.executable,
              'transport': 'real subprocess stdio through official SDK',
              'message_record_scope': 'SDK-decoded catalog and call results; not raw wire transcript',
              'model_requests': 0, 'network_requests': 0, 'automatic_retries': 0,
              'scope': 'one local file tool; not official 54-tool campaign or model decision test'}
    print('4-2第二步：官方MCP SDK 2.3.0，列工具并读取同一个测试文件。')
    print('SDK负责连接与消息处理；客户端仍预先指定调用；无大模型或网络请求。')
    if args.preview:
        print('MCP_SDK_PREVIEW_ONLY：依赖及文件检查通过，未启动服务或创建正式记录。')
        return 0
    if args.check:
        asyncio.run(asyncio.wait_for(exercise(record), timeout=30))
        print('MCP_SDK_OFFLINE_CHECK_PASSED：', record['protocol_version'], record['checks'])
        return 0
    directory = ROOT/'运行结果'/('sdk_file_read_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    save()
    print('记录路径：', path, flush=True)
    started = time.perf_counter()
    try:
        asyncio.run(asyncio.wait_for(exercise(record, save), timeout=30))
        record['status'] = 'sdk_stdio_file_read_verified'
        print('SDK版本：', record['mcp_sdk_version'])
        print('实际协议版本：', record['protocol_version'])
        print('发现工具：', record['catalog']['tools'][0]['name'])
        print('工具参数：', json.dumps(record['catalog']['tools'][0]['inputSchema'], ensure_ascii=False))
        print('读取结果：', record['successful_call']['content'][0]['text'])
        print('错误示例：', record['failed_call']['content'][0]['text'])
        print('核对：', record['checks'])
        print('MCP_SDK_FILE_READ_SAVED：SDK连接、目录与两次调用结果保存。')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print('本步未完成，不重试：', record['error'])
        return 1
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        save()
        print('记录路径：', path)


if __name__ == '__main__':
    raise SystemExit(main())

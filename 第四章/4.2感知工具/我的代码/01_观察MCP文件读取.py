"""真实子进程stdio入门：初始化→列工具→读取文件→观察不存在文件错误。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(__file__).with_name('文件读取服务.py')
FIXTURE = ROOT/'测试材料/工具说明.txt'
VERSION = '2025-06-18'
CHINA = timezone(timedelta(hours=8))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exercise(record, save=lambda: None):
    transcript = record.setdefault('transcript', [])
    with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as errors:
        process = subprocess.Popen([sys.executable, '-u', str(SERVER)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors, text=True, encoding='utf-8')
        record['server_pid'] = process.pid
        record['client_pid'] = __import__('os').getpid()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        sequence = 0

        def send(message):
            transcript.append({'direction': 'client_to_server', 'message': message})
            save()
            process.stdin.write(json.dumps(message, ensure_ascii=False)+'\n')
            process.stdin.flush()

        def rpc(method, params):
            nonlocal sequence
            sequence += 1
            send({'jsonrpc': '2.0', 'id': sequence, 'method': method, 'params': params})
            if not selector.select(timeout=10):
                raise TimeoutError('服务端10秒未返回')
            line = process.stdout.readline()
            if not line:
                raise RuntimeError('服务进程提前结束')
            reply = json.loads(line)
            transcript.append({'direction': 'server_to_client', 'message': reply})
            save()
            if reply.get('jsonrpc') != '2.0' or reply.get('id') != sequence or 'error' in reply:
                raise ValueError('JSON-RPC响应异常：'+str(reply))
            return reply['result']

        try:
            init = rpc('initialize', {'protocolVersion': VERSION, 'capabilities': {},
                       'clientInfo': {'name': 'learning-client', 'version': '0.1.0'}})
            record['initialize_result'] = init
            if init['protocolVersion'] != VERSION or 'tools' not in init['capabilities']:
                raise ValueError('协商版本或工具能力不符')
            send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
            catalog = rpc('tools/list', {})
            if [x['name'] for x in catalog['tools']] != ['file_reader']:
                raise ValueError('工具目录异常')
            success = rpc('tools/call', {'name': 'file_reader', 'arguments': {'file_path': '工具说明.txt'}})
            failed = rpc('tools/call', {'name': 'file_reader', 'arguments': {'file_path': '不存在.txt'}})
            returned = json.loads(success['content'][0]['text'])
            failure = json.loads(failed['content'][0]['text'])
            expected = FIXTURE.read_text(encoding='utf-8')
            if success.get('isError') is not False or returned['success'] is not True:
                raise ValueError('正常读取失败')
            if returned['message']['content'] != expected or returned['message']['truncated']:
                raise ValueError('返回正文与原文件不一致')
            if failed.get('isError') is not True or failure['success'] is not False or failure['metadata']['error_type'] != 'FileNotFoundError':
                raise ValueError('缺失文件未返回预期错误')
            record.update(catalog=catalog, successful_call=success, failed_call=failed,
                          checks={'separate_processes': record['server_pid'] != record['client_pid'],
                                  'protocol_agreed': True, 'content_matches_file': True, 'missing_file_error_observed': True})
        finally:
            selector.close()
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            process.stdout.close()
            errors.seek(0)
            record['server_stderr'] = errors.read()
            record['server_exit_code'] = process.returncode
            save()
        if process.returncode != 0:
            raise ValueError('服务未正常退出')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--check', action='store_true', help='实际stdio检查，但不创建正式记录')
    args = parser.parse_args()
    print('4-2第一步：真实本地stdio通信，列出工具并读取测试文件。')
    print('最小教学协议子集2025-06-18；不是官方SDK v2或全套54工具验收。')
    print('无SDK依赖、模型、网络或外部账号调用。固定2次工具调用：正常读取与不存在文件。')
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
              'client_sha256': sha(Path(__file__)), 'server_sha256': sha(SERVER),
              'fixture_sha256': sha(FIXTURE), 'protocol_requested': VERSION,
              'python': sys.version, 'executable': sys.executable, 'mcp_sdk': None,
              'implementation': 'stdlib teaching subset over real subprocess stdio',
              'model_requests': 0, 'network_requests': 0}
    if args.preview:
        print('MCP_FILE_PREVIEW_ONLY：文件入口可用，未启动服务或创建记录。')
        return 0
    if args.check:
        exercise(record)
        print('MCP_FILE_OFFLINE_CHECK_PASSED：真实stdio与内容一致性通过，未创建正式结果目录。')
        return 0
    directory = ROOT/'运行结果'/('file_read_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
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
        exercise(record, save)
        record['status'] = 'stdio_file_read_verified'
        print('协商协议：', record['initialize_result']['protocolVersion'])
        print('发现工具：', record['catalog']['tools'][0]['name'])
        print('工具参数：', json.dumps(record['catalog']['tools'][0]['inputSchema'], ensure_ascii=False))
        print('读取结果：', record['successful_call']['content'][0]['text'])
        print('错误示例：', record['failed_call']['content'][0]['text'])
        print('核对：返回正文与原文件逐字一致；不存在文件按预期返回错误。')
        print('MCP_FILE_READ_SAVED：初始化、工具目录、调用与返回消息均已保存。')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print('本步未完成：', record['error'])
        return 1
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        save()
        print('记录路径：', path)


if __name__ == '__main__':
    raise SystemExit(main())

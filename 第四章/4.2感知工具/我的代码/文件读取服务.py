"""仅用于教学的MCP stdio子集：initialize、tools/list、tools/call。

独立进程，stdout只写JSON-RPC消息；没有SDK、模型或网络请求。
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]/'测试材料'
VERSION = '2025-06-18'
TOOL = {'name': 'file_reader', 'description': '读取测试材料目录中的UTF-8文本文件。',
        'inputSchema': {'type': 'object', 'properties': {
            'file_path': {'type': 'string', 'description': '相对测试材料目录的文件路径'},
            'max_length': {'type': 'integer', 'minimum': 1, 'maximum': 50000, 'default': 50000}},
            'required': ['file_path'], 'additionalProperties': False}}


def read_file(arguments):
    try:
        if not isinstance(arguments, dict) or not {'file_path'} <= set(arguments) <= {'file_path', 'max_length'}:
            raise ValueError('参数必须含file_path，可选max_length')
        name, limit = arguments['file_path'], arguments.get('max_length', 50000)
        if not isinstance(name, str) or not name.strip():
            raise ValueError('文件名不能为空')
        if type(limit) is not int or not 1 <= limit <= 50000:
            raise ValueError('max_length须为1至50000的整数')
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('只接受测试材料中的相对路径')
        path = (ROOT/rel).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError('路径须为测试材料中的文件')
        if path.stat().st_size > 1024*1024:
            raise ValueError('本教学入口只读1MB内文本')
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        response = {'success': True, 'message': {'file_path': name, 'content': text[:limit],
                    'size_bytes': len(raw), 'truncated': len(text) > limit, 'encoding': 'utf-8'},
                    'metadata': {'source': 'local fixture'}}
        failed = False
    except (OSError, ValueError, UnicodeError) as error:
        response = {'success': False, 'message': str(error), 'metadata': {'error_type': type(error).__name__}}
        failed = True
    return {'content': [{'type': 'text', 'text': json.dumps(response, ensure_ascii=False)}], 'isError': failed}


def main():
    initialized = False
    ready = False
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get('jsonrpc') != '2.0' or not isinstance(request.get('method'), str):
                raise ValueError('无效JSON-RPC请求')
            method = request['method']
            if 'id' not in request:
                if method == 'notifications/initialized' and initialized:
                    ready = True
                continue
            params = request.get('params', {})
            if not isinstance(params, dict):
                raise ValueError('params必须为对象')
            if method == 'initialize' and not initialized:
                result = {'protocolVersion': VERSION, 'capabilities': {'tools': {'listChanged': False}},
                          'serverInfo': {'name': 'learning-file-reader', 'version': '0.1.0'}}
                initialized = True
            elif method == 'ping':
                result = {}
            elif not ready:
                raise ValueError('须先完成初始化')
            elif method == 'tools/list':
                result = {'tools': [TOOL]}
            elif method == 'tools/call' and params.get('name') == 'file_reader':
                result = read_file(params.get('arguments', {}))
            else:
                print(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                                  'error': {'code': -32601, 'message': '未知方法或工具'}}), flush=True)
                continue
            reply = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}
        except (ValueError, TypeError) as error:
            identifier = request.get('id') if isinstance(request, dict) else None
            if isinstance(request, dict) and 'id' not in request:
                continue
            reply = {'jsonrpc': '2.0', 'id': identifier, 'error': {
                'code': -32700 if isinstance(error, json.JSONDecodeError) else -32600, 'message': str(error)}}
        print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

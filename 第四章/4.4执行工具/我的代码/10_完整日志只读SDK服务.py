"""针对本轮指定完整日志提供搜索行号和按行读取；真实MCP stdio。"""
import hashlib
import json
import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

server = MCPServer('learning-log-reader', version='0.1.0', log_level='ERROR')


def lines():
    filename = os.environ.get('TEACHING_LOG_PATH')
    expected = os.environ.get('TEACHING_LOG_SHA256')
    if not filename or not expected:
        raise ValueError('未配置本轮日志路径或哈希')
    path = Path(filename)
    if path.is_symlink() or not path.is_file():
        raise ValueError('日志须为当前指定的真实文件')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('日志字节指纹变化')
    return raw.decode('utf-8').splitlines()


def packed(operation):
    try:
        data = {'success': True, 'message': operation()}
    except (OSError, ValueError, UnicodeError) as error:
        data = {'success': False, 'message': str(error), 'error_type': type(error).__name__}
    return CallToolResult(content=[TextContent(type='text', text=json.dumps(data, ensure_ascii=False))],
                          is_error=not data['success'])


@server.tool()
def search_log(pattern: str) -> CallToolResult:
    """在当前完整日志按字面关键词搜索，区分大小写。返回1起算文件行号，不返回原文；最多20个行号，含总匹配数和截断状态。"""
    def operation():
        if not isinstance(pattern, str) or not pattern.strip() or len(pattern) > 80:
            raise ValueError('关键词须1至80字符的非空文本')
        source = lines()
        matches = [i for i, line in enumerate(source, 1) if pattern in line]
        return {'line_numbers': matches[:20], 'match_count': len(matches),
                'truncated': len(matches) > 20, 'total_lines': len(source)}
    return packed(operation)


@server.tool()
def read_log_lines(start_line: int, end_line: int) -> CallToolResult:
    """读取当前完整日志的指定文件行。行号1起算，包含两端，每次最多20行；返回真实原文与行号。"""
    def operation():
        source = lines()
        if type(start_line) is not int or type(end_line) is not int or start_line < 1 or end_line < start_line:
            raise ValueError('行号范围无效')
        if end_line > len(source) or end_line-start_line+1 > 20:
            raise ValueError('行号越界或超过20行')
        return {'lines': [{'line_number': i, 'text': source[i-1]} for i in range(start_line, end_line+1)],
                'total_lines': len(source)}
    return packed(operation)


if __name__ == '__main__':
    server.run(transport='stdio')

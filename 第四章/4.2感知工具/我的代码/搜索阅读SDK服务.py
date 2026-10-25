"""两工具独立教学服务；仅访问搜索阅读材料，不改动上一阶段服务。"""
import json
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

ROOT = Path(__file__).resolve().parents[1]/'搜索阅读材料'
server = MCPServer('learning-search-reader', version='0.1.0', log_level='ERROR')


def response(ok, message, error_type=None):
    data = {'success': ok, 'message': message, 'metadata': {'source': 'local teaching fixtures'}}
    if error_type:
        data['metadata']['error_type'] = error_type
    return CallToolResult(content=[TextContent(type='text', text=json.dumps(data, ensure_ascii=False))], is_error=not ok)


def text_file(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('文件名不能为空')
    rel = Path(name)
    if rel.is_absolute() or len(rel.parts) != 1 or '..' in rel.parts:
        raise ValueError('仅接受材料目录内单个相对文件名')
    path = (ROOT/rel).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.suffix != '.txt':
        raise ValueError('只读材料目录内txt文件')
    if path.stat().st_size > 1024*1024:
        raise ValueError('文件超过教学读取上限')
    return path, path.read_text(encoding='utf-8')


@server.tool()
def grep(pattern: str) -> CallToolResult:
    """在全部本地测试文本中按关键词字面搜索，区分大小写。只返回匹配文件名和行号，不返回正文；需要内容时调用file_reader。"""
    try:
        if not isinstance(pattern, str) or not pattern.strip() or len(pattern) > 80:
            raise ValueError('关键词须为1至80字符的非空文本')
        matches = []
        for path in sorted(ROOT.glob('*.txt')):
            _, text = text_file(path.name)
            lines = [i for i, line in enumerate(text.splitlines(), 1) if pattern in line]
            if lines:
                matches.append({'file_path': path.name, 'line_numbers': lines})
        return response(True, {'pattern': pattern, 'matches': matches, 'files_scanned': len(list(ROOT.glob('*.txt')))})
    except (OSError, ValueError, UnicodeError) as error:
        return response(False, str(error), type(error).__name__)


@server.tool()
def file_reader(file_path: str, max_length: int = 50000) -> CallToolResult:
    """读取本地测试材料中的一个UTF-8文本文件。file_path为相对文件名，可使用grep返回的路径。"""
    try:
        if type(max_length) is not int or not 1 <= max_length <= 50000:
            raise ValueError('读取长度须为1至50000的整数')
        path, text = text_file(file_path)
        return response(True, {'file_path': file_path, 'content': text[:max_length],
                              'truncated': len(text) > max_length, 'size_bytes': path.stat().st_size})
    except (OSError, ValueError, UnicodeError) as error:
        return response(False, str(error), type(error).__name__)


if __name__ == '__main__':
    server.run(transport='stdio')

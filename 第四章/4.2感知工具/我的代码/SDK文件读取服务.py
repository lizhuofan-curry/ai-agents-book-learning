"""用官方MCP SDK注册file_reader，复用第一步的受限读取函数。"""
import importlib.util
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import CallToolResult

HELPER = Path(__file__).with_name('文件读取服务.py')
spec = importlib.util.spec_from_file_location('local_reader', HELPER)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
server = MCPServer('learning-sdk-file-reader', version='0.1.0', log_level='ERROR')


@server.tool()
def file_reader(file_path: str, max_length: int = 50000) -> CallToolResult:
    """读取测试材料目录中的UTF-8文本文件，返回正文或明确错误。"""
    return CallToolResult.model_validate(reader.read_file({'file_path': file_path, 'max_length': max_length}))


if __name__ == '__main__':
    server.run(transport='stdio')

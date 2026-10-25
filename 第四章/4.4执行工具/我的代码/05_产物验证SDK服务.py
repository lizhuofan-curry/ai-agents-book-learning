"""为当前固定产物开放只读核对工具；不暴露版本标签。"""
import importlib.util
import json
import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

HELPER = Path(__file__).with_name('执行工具教学实现.py')
spec = importlib.util.spec_from_file_location('output_verifier_core', HELPER)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
server = MCPServer('learning-output-verifier', version='0.1.0', log_level='ERROR')


@server.tool()
def verify_counts() -> CallToolResult:
    """读取当前任务的输入和真实counts.json，独立计数比较。goal_correct表示产物是否符合词频统计任务，不修改或重新执行程序。"""
    try:
        configured = os.environ.get('EXECUTION_VERIFY_ROOT')
        if not configured:
            raise ValueError('未配置本轮产物目录')
        root = Path(configured).resolve(strict=True)
        for filename in ('data.txt', 'counts.json'):
            path = root/filename
            if path.is_symlink() or not path.is_file():
                raise ValueError('输入及产物须为当前目录内真实文件')
        result = core.verify_counts(root)
    except (OSError, ValueError) as error:
        result = {'error_type': type(error).__name__, 'error': str(error), 'goal_correct': None}
    return CallToolResult(content=[TextContent(type='text', text=json.dumps(result, ensure_ascii=False))],
                          is_error='error_type' in result)


if __name__ == '__main__':
    server.run(transport='stdio')

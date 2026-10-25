"""官方SDK注册固定教学执行工具，复用第一步内部实现。"""
import importlib.util
import json
import os
from pathlib import Path
from typing import Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

HELPER = Path(__file__).with_name('执行工具教学实现.py')
spec = importlib.util.spec_from_file_location('fixed_execution_tools', HELPER)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
server = MCPServer('learning-execution-tools', version='0.1.0', log_level='ERROR')
Variant = Literal['correct', 'wrong']


def workspace(variant):
    configured = os.environ.get('EXECUTION_TEACHING_ROOT')
    if not configured or variant not in ('correct', 'wrong'):
        raise ValueError('须配置本轮工作区及有效版本')
    root = Path(configured).resolve(strict=True)
    name = '正确版本' if variant == 'correct' else '错误版本'
    target = root/name
    if target.is_symlink():
        raise ValueError('不允许版本目录为符号链接')
    target = target.resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_dir():
        raise ValueError('版本目录须位于本轮工作区')
    return target


def packed(operation):
    try:
        result = operation()
    except (OSError, ValueError, TypeError) as error:
        result = {'success': False, 'error_type': type(error).__name__, 'error': str(error)}
    # 验证已执行但goal_correct=false，不是协议或验证工具执行失败。
    failed = result.get('success') is False or 'error_type' in result
    return CallToolResult(content=[TextContent(type='text', text=json.dumps(result, ensure_ascii=False))],
                          is_error=failed)


@server.tool()
def file_write(variant: Variant, filename: str, content: str) -> CallToolResult:
    """在本轮指定版本工作区写入文件；Python先校验语法，不覆盖已有文件。"""
    return packed(lambda: tools.file_write(workspace(variant), filename, content))


@server.tool()
def run_python(variant: Variant, filename: str) -> CallToolResult:
    """在独立Python子进程运行固定教学脚本；只允许wordcount.py或wrong_count.py且源码必须匹配。"""
    return packed(lambda: tools.run_fixed_python(workspace(variant), filename))


@server.tool()
def verify_counts(variant: Variant) -> CallToolResult:
    """读取真实counts.json并按输入独立核对。goal_correct表示目标正确，不等于进程退出状态。"""
    return packed(lambda: tools.verify_counts(workspace(variant)))


if __name__ == '__main__':
    server.run(transport='stdio')

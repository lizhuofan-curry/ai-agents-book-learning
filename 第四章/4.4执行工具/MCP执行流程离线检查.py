import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/'我的代码'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('mcp_execution_client', '03_通过MCP写入运行与核对.py')
server = load('mcp_execution_server', '02_执行工具SDK服务.py')


class MCPExecutionChecks(unittest.TestCase):
    def test_actual_stdio_file_changes_and_goal_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            record = {}
            asyncio.run(m.exercise(temp, record, verbose=False))
            self.assertEqual(len(record['calls']), 9)
            self.assertTrue(all(record['checks'].values()))
            self.assertTrue(record['calls'][2]['result']['isError'])
            # 逻辑错误脚本正常退出，验证工具也正常完成，但验证目标为false。
            self.assertFalse(record['calls'][7]['result']['isError'])
            self.assertFalse(record['calls'][8]['result']['isError'])
            self.assertFalse(record['calls'][8]['decoded_result']['goal_correct'])
            self.assertEqual(len(record['artifacts']), 6)

    def test_missing_configuration_and_symlink_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(server.verify_counts('correct').is_error)
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            (Path(temp)/'正确版本').symlink_to(outside, target_is_directory=True)
            with patch.dict(os.environ, {'EXECUTION_TEACHING_ROOT': temp}):
                result = server.file_write('correct', 'x.txt', 'test')
                self.assertTrue(result.is_error)
                self.assertFalse((Path(outside)/'x.txt').exists())

    def test_missing_script_returns_tool_error(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'正确版本').mkdir()
            with patch.dict(os.environ, {'EXECUTION_TEACHING_ROOT': temp}):
                result = server.run_python('correct', 'wordcount.py')
                self.assertTrue(result.is_error)
                data = json.loads(result.content[0].text)
                self.assertEqual(data['error_type'], 'FileNotFoundError')


if __name__ == '__main__':
    unittest.main()

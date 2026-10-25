"""模拟模型+真实SDK stdio检查；不访问Ollama或生成正式结果目录。"""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('model_mcp', HERE/'我的代码/04_本机模型选择并调用MCP工具.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ModelBridgeChecks(unittest.TestCase):
    def test_fake_model_through_real_sdk_stdio(self):
        requests = []

        async def fake(request, record, save):
            requests.append(deepcopy(request))
            if len(requests) == 1:
                self.assertNotIn('content_matches_fixture', json.dumps(request))
                self.assertNotIn(m.sdk.FIXTURE.read_text(), json.dumps(request, ensure_ascii=False))
                self.assertEqual(request['tools'][0]['function']['name'], 'file_reader')
                return {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'function': {'name': 'file_reader', 'arguments': {'file_path': '工具说明.txt'}}}]}
            self.assertNotIn('tools', request)
            tool = request['messages'][-1]
            self.assertEqual(tool['role'], 'tool')
            self.assertEqual(tool['tool_name'], 'file_reader')
            self.assertEqual(json.loads(tool['content'])['message']['content'], m.sdk.FIXTURE.read_text())
            return {'role': 'assistant', 'content': '模拟回答：只用于连通性检查，不是模型实测。'}

        record = {}
        asyncio.run(m.exercise(record, lambda: None, fake))
        self.assertEqual(len(requests), 2)
        self.assertEqual(record['status'], 'model_mcp_answer_saved_review_pending')
        self.assertTrue(record['file_check']['complete_content'])

    def test_catalog_schema_forwarded(self):
        catalog = {'tools': [{'name': 'file_reader', 'description': 'test', 'inputSchema': {
            'type': 'object', 'properties': {'file_path': {'type': 'string'}}, 'required': ['file_path']}}]}
        self.assertEqual(m.tools_for_model(catalog)[0]['function']['parameters'], catalog['tools'][0]['inputSchema'])

    def test_no_call_wrong_path_and_multiple_calls_rejected(self):
        catalog = {'tools': [{'name': 'file_reader', 'inputSchema': {'type': 'object'}}]}
        call = {'function': {'name': 'file_reader', 'arguments': {'file_path': '../private.txt'}}}
        for message in ({'content': '猜测文件内容'}, {'tool_calls': [call]}, {'tool_calls': [call, call]}):
            with self.assertRaises(ValueError):
                m.selected_call(message, catalog)

    def test_boolean_length_and_duplicate_json_rejected(self):
        catalog = {'tools': [{'name': 'file_reader', 'inputSchema': {'type': 'object'}}]}
        with self.assertRaises(ValueError):
            m.selected_call({'tool_calls': [{'function': {'name': 'file_reader', 'arguments': {
                'file_path': '工具说明.txt', 'max_length': True}}}]}, catalog)
        with self.assertRaises(ValueError):
            json.loads('{"a":1,"a":2}', object_pairs_hook=m.unique)


if __name__ == '__main__':
    unittest.main()

"""模拟模型决策，真实SDK stdio串联；不访问Ollama。"""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('search_read_entry', HERE/'我的代码/07_模型搜索再读取.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def tool(name, arguments):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]}


class SearchReadChecks(unittest.TestCase):
    def test_fake_model_with_real_mcp_search_and_read(self):
        requests = []
        discovered = []

        async def fake(request, record, save):
            requests.append(deepcopy(request))
            if len(requests) == 1:
                for filename in m.fixture_hashes():
                    self.assertNotIn(filename, json.dumps(request, ensure_ascii=False))
                self.assertEqual({t['function']['name'] for t in request['tools']}, {'grep', 'file_reader'})
                return tool('grep', {'pattern': 'MCP'})
            observation = json.loads(request['messages'][-1]['content'])['message']
            if len(requests) == 2:
                self.assertEqual(request['messages'][-1]['tool_name'], 'grep')
                self.assertEqual(len(observation['matches']), 1)
                self.assertNotIn('content', observation)
                discovered.append(observation['matches'][0]['file_path'])
                return tool('file_reader', {'file_path': discovered[0]})
            self.assertNotIn('tools', request)
            self.assertEqual(request['messages'][-1]['tool_name'], 'file_reader')
            self.assertIn('MCP', observation['content'])
            return {'role': 'assistant', 'content': '模拟回答，不是模型实测。'}

        record = {}
        asyncio.run(m.exercise(record, lambda: None, fake))
        self.assertEqual(len(requests), 3)
        self.assertEqual([c['name'] for c in record['tool_calls']], ['grep', 'file_reader'])
        self.assertEqual(record['tool_calls'][1]['arguments']['file_path'], discovered[0])
        self.assertEqual(record['status'], 'search_read_answer_saved_review_pending')

    def test_early_answer_preserved_without_fake_tools(self):
        async def fake(request, record, save):
            return {'role': 'assistant', 'content': '这是未读取就给出的模拟回答。'}
        record = {}
        asyncio.run(m.exercise(record, lambda: None, fake))
        self.assertEqual(record['tool_calls'], [])
        self.assertEqual(record['status'], 'answer_without_verified_chain_review_pending')

    def test_traversal_unknown_tool_and_multiple_calls_rejected(self):
        catalog = {'tools': [{'name': n, 'inputSchema': {'type': 'object'}} for n in ('grep', 'file_reader')]}
        invalid = [tool('file_reader', {'file_path': '../secret.txt'}), tool('write', {}),
                   {'tool_calls': [tool('grep', {'pattern': 'MCP'})['tool_calls'][0]]*2}]
        for message in invalid:
            with self.assertRaises(ValueError):
                m.selected_call(message, catalog)

    def test_no_regex_or_empty_search(self):
        spec = importlib.util.spec_from_file_location('search_reader_server', HERE/'我的代码/搜索阅读SDK服务.py')
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        # Test the underlying registered Python handler directly for literal-search semantics.
        result = server.grep('MCP|记录')
        self.assertEqual(json.loads(result.content[0].text)['message']['matches'], [])
        self.assertTrue(server.grep('').is_error)


if __name__ == '__main__':
    unittest.main()

"""模拟模型+真实MCP stdio检查；不调用Ollama。"""
import asyncio
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/'我的代码'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('empty_search_entry', '09_模型处理搜索无结果.py')
previous = load('successful_search_entry', '07_模型搜索再读取.py')


def tool(name, arguments):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]}


class EmptySearchChecks(unittest.TestCase):
    def test_same_settings_and_only_question_changed(self):
        self.assertEqual(m.SYSTEM, previous.SYSTEM)
        self.assertEqual(m.MODEL, previous.MODEL)
        self.assertEqual(m.fixture_hashes(), previous.fixture_hashes())
        self.assertEqual(m.SERVER, previous.SERVER)
        self.assertEqual(m.payload([], []), previous.payload([], []))
        for path in m.FIXTURES.glob('*.txt'):
            self.assertNotIn(m.TARGET, path.read_text())

    def test_real_empty_result_then_model_stops(self):
        requests = []

        async def fake(request, record, save):
            requests.append(request)
            if len(requests) == 1:
                self.assertNotIn('未找到', request['messages'][0]['content'])
                for name in m.fixture_hashes():
                    self.assertNotIn(name, request['messages'][1]['content'])
                return tool('grep', {'pattern': m.TARGET})
            last = request['messages'][-1]
            self.assertEqual(last['role'], 'tool')
            observed = json.loads(last['content'])
            self.assertTrue(observed['success'])
            self.assertEqual(observed['message']['matches'], [])
            self.assertEqual(observed['message']['files_scanned'], 3)
            return {'role': 'assistant', 'content': '模拟回答：当前材料未检索到该词。'}

        record = {}
        asyncio.run(m.exercise(record, lambda: None, fake))
        self.assertEqual(len(requests), 2)
        self.assertEqual([x['name'] for x in record['tool_calls']], ['grep'])
        self.assertEqual(record['status'], 'empty_search_answer_saved_review_pending')

    def test_guessed_read_retained_and_not_marked_clean_stop(self):
        count = 0

        async def fake(request, record, save):
            nonlocal count
            count += 1
            if count == 1:
                return tool('grep', {'pattern': m.TARGET})
            if count == 2:
                return tool('file_reader', {'file_path': '虚构材料.txt'})
            self.assertTrue(json.loads(request['messages'][-1]['content'])['success'] is False)
            return {'role': 'assistant', 'content': '模拟错误行为后的回答。'}

        record = {}
        asyncio.run(m.exercise(record, lambda: None, fake))
        self.assertEqual(record['status'], 'answer_needs_empty_search_review')
        self.assertEqual(record['evidence_check']['read_tool_calls'], 1)
        self.assertEqual(record['tool_calls'][1]['arguments']['file_path'], '虚构材料.txt')

    def test_third_tool_request_not_executed(self):
        async def fake(request, record, save):
            return tool('grep', {'pattern': m.TARGET})
        record = {}
        with self.assertRaises(ExceptionGroup) as raised:
            asyncio.run(m.exercise(record, lambda: None, fake))
        def leaves(error):
            if isinstance(error, BaseExceptionGroup):
                return [leaf for child in error.exceptions for leaf in leaves(child)]
            return [error]
        self.assertTrue(any(isinstance(e, ValueError) and '预算耗尽' in str(e) for e in leaves(raised.exception)))
        self.assertEqual(len(record['tool_calls']), 2)


if __name__ == '__main__':
    unittest.main()

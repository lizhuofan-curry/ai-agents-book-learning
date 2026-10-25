"""模拟模型与真实MCP验证工具往返；不访问Ollama或创建正式结果。"""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('verification_model', HERE/'我的代码/06_模型依据验证结果判断完成.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class VerificationChecks(unittest.TestCase):
    def test_fake_model_real_faulty_output_and_mcp(self):
        requests = []

        async def fake(request, record, save):
            requests.append(request)
            if len(requests) == 1:
                visible = json.dumps(request, ensure_ascii=False)
                for hidden in ('999', 'wrong_count.py', 'goal_correct": false'):
                    self.assertNotIn(hidden, visible)
                self.assertEqual(request['tools'][0]['function']['name'], 'verify_counts')
                return {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'function': {'name': 'verify_counts', 'arguments': {}}}]}
            self.assertNotIn('tools', request)
            result = json.loads(request['messages'][-1]['content'])
            self.assertFalse(result['goal_correct'])
            self.assertEqual(result['actual']['apple'], 999)
            self.assertEqual(result['expected']['apple'], 4)
            return {'role': 'assistant', 'content': '模拟判断：结果未满足任务要求。'}

        with tempfile.TemporaryDirectory() as temp:
            record = {}
            m.prepare_fixture(temp, record)
            before = record['prepared_fixture_hashes']
            asyncio.run(m.exercise(temp, record, lambda: None, fake))
            self.assertEqual(len(requests), 2)
            self.assertEqual(record['status'], 'verification_answer_saved_review_pending')
            self.assertFalse(record['tool_call']['result']['isError'])
            self.assertEqual(before, {p.name: m.execution.sha(p) for p in Path(temp).iterdir()})

    def test_early_unverified_answer_not_accepted_as_verified(self):
        async def fake(request, record, save):
            return {'role': 'assistant', 'content': '模拟错误判断：退出码0所以完成。'}
        with tempfile.TemporaryDirectory() as temp:
            record = {}
            m.prepare_fixture(temp, record)
            asyncio.run(m.exercise(temp, record, lambda: None, fake))
            self.assertEqual(record['status'], 'answer_without_verification_review_pending')
            self.assertNotIn('tool_call', record)

    def test_verifier_can_report_correct_output(self):
        async def fake(request, record, save):
            if 'tools' in request:
                return {'role': 'assistant', 'tool_calls': [{'function': {'name': 'verify_counts', 'arguments': {}}}]}
            observed = json.loads(request['messages'][-1]['content'])
            self.assertTrue(observed['goal_correct'])
            return {'role': 'assistant', 'content': '模拟正确产物判断。'}
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(m.execution.file_write(temp, 'data.txt', m.execution.SAMPLE)['success'])
            self.assertTrue(m.execution.file_write(temp, 'wordcount.py', m.execution.GOOD)['success'])
            self.assertTrue(m.execution.run_fixed_python(temp, 'wordcount.py')['success'])
            record = {}
            asyncio.run(m.exercise(temp, record, lambda: None, fake))
            self.assertTrue(record['observed_verification']['goal_correct'])


if __name__ == '__main__':
    unittest.main()

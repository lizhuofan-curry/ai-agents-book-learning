"""模拟模型验证两组隔离、候选加载、参数传递及预算；不请求Ollama。"""
import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).resolve().parent/'我的代码/02_全量与按需工具选择对照.py'
spec = importlib.util.spec_from_file_location('tool_choice_compare', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def call(name, args):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': args}}]}


def scripted(actions, requests):
    async def fake(request, arm, save):
        index = len(requests)
        requests.append(deepcopy(request))
        arm['calls'].append({'request': deepcopy(request), 'schema_metrics': m.catalog_step.metrics(request.get('tools', []))})
        if index < len(actions):
            return deepcopy(actions[index])
        return {'role': 'assistant', 'content': '模拟测试回答：只做了教学模拟，没有真实下载。'}
    return fake


class ChoiceCompareChecks(unittest.TestCase):
    def setUp(self):
        self.full, self.initial, _ = m.prepare()

    def test_full_exposes126_and_records_simulated_binding(self):
        requests = []
        actions = [call('arxiv_search', {'query': 'Transformer', 'max_results': 1}),
                   call('download_file', {'url': 'https://example.invalid/teaching/transformer-1.pdf', 'path': 'paper.pdf'})]
        arm = {}
        asyncio.run(m.run_arm('full', self.full, self.initial, arm, lambda: None, scripted(actions, requests)))
        self.assertEqual(len(requests[0]['tools']), 126)
        self.assertEqual(arm['status'], 'answered_review_pending')
        self.assertTrue(arm['metrics']['search_then_simulated_download_with_returned_url'])
        self.assertEqual(arm['metrics']['real_files_downloaded'], 0)

    def test_discovery_loads_candidates_and_keeps_same_settings(self):
        requests = []
        actions = [call('discover_tools', {'need': 'arxiv'}),
                   call('arxiv_search', {'query': 'Transformer', 'max_results': 1}),
                   call('discover_tools', {'need': 'download'}),
                   call('download_file', {'url': 'https://example.invalid/teaching/transformer-1.pdf', 'path': 'paper.pdf'})]
        arm = {}
        asyncio.run(m.run_arm('discovery', self.full, self.initial, arm, lambda: None, scripted(actions, requests)))
        self.assertEqual(len(requests), 5)
        self.assertEqual(len(requests[0]['tools']), 3)
        initial_names = [t['function']['name'] for t in requests[0]['tools']]
        self.assertNotIn('arxiv_search', initial_names)
        self.assertNotIn('download_file', initial_names)
        self.assertEqual(arm['metrics']['discovery_actions'], 2)
        self.assertTrue(arm['metrics']['search_then_simulated_download_with_returned_url'])
        self.assertNotIn('tools', requests[-1])
        self.assertTrue(all(r['options'] == requests[0]['options'] for r in requests))
        self.assertTrue(all(r['messages'][0]['content'] == m.SYSTEM for r in requests))

    def test_unloaded_tool_rejected_without_autofill(self):
        requests = []
        arm = {}
        asyncio.run(m.run_arm('discovery', self.full, self.initial, arm, lambda: None,
                             scripted([call('arxiv_search', {'query': 'Transformer', 'max_results': 1})], requests)))
        self.assertEqual(arm['status'], 'incomplete')
        self.assertEqual(arm['actions'], [])

    def test_empty_or_no_overlap_discovery_not_oracle_filled(self):
        index = m.ToolIndex(self.full)
        self.assertEqual(index.search('zzzzznotfoundzzzzz'), [])
        with self.assertRaises(ValueError):
            index.search(' ')

    def test_unreturned_url_and_early_answer_not_counted_as_chain(self):
        requests = []
        arm = {}
        asyncio.run(m.run_arm('full', self.full, self.initial, arm, lambda: None,
                             scripted([call('download_file', {'url': 'https://example.invalid/made-up.pdf', 'path': 'x.pdf'})], requests)))
        self.assertFalse(arm['metrics']['search_then_simulated_download_with_returned_url'])
        early = {}
        asyncio.run(m.run_arm('discovery', self.full, self.initial, early, lambda: None, scripted([], [])))
        self.assertEqual(early['actions'], [])
        self.assertFalse(early['metrics']['search_then_simulated_download_with_returned_url'])


if __name__ == '__main__':
    unittest.main()

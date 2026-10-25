import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).resolve().parent/'我的代码/07_子任务完成检查提示对照.py'
spec = importlib.util.spec_from_file_location('subtask_prompt', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PromptChecks(unittest.TestCase):
    def test_only_system_append_changed_at_first_request(self):
        source, full, initial, _ = m.prepare()
        requests = []
        original = m.compare.SYSTEM

        async def fake(request, arm, save):
            requests.append(deepcopy(request))
            arm['calls'].append({'request': deepcopy(request), 'schema_metrics': m.compare.catalog_step.metrics(request.get('tools', []))})
            return {'role': 'assistant', 'content': '模拟提前结束，不表示成功。'}

        arm = {}
        asyncio.run(m.run_checked(full, initial, arm, lambda: None, fake))
        baseline = deepcopy(source['arms']['discovery']['calls'][0]['request'])
        expected = deepcopy(baseline)
        expected['messages'][0]['content'] += '\n'+m.ADDED_PROMPT
        self.assertEqual(requests[0], expected)
        self.assertEqual(m.compare.SYSTEM, original)
        self.assertEqual(len(requests), 1)
        self.assertEqual(arm['actions'], [])
        self.assertFalse(arm['metrics']['search_then_simulated_download_with_returned_url'])

    def test_no_target_tool_or_query_in_added_prompt(self):
        for forbidden in ('arxiv_search', 'download_file', '下载论文PDF文件', 'Transformer'):
            self.assertNotIn(forbidden, m.ADDED_PROMPT)

    def test_index_backend_budgets_and_candidates_preserved(self):
        source, full, initial, _ = m.prepare()
        self.assertEqual(m.compare.MAX_CALLS, 5)
        self.assertEqual(m.compare.MAX_ACTIONS, 4)
        self.assertEqual(len(initial), 3)
        action = source['arms']['discovery']['actions'][0]
        ranking = m.compare.ToolIndex(full).search(action['arguments']['need'], excluded={t['function']['name'] for t in initial})
        self.assertEqual(ranking, action['ranking'])


if __name__ == '__main__':
    unittest.main()

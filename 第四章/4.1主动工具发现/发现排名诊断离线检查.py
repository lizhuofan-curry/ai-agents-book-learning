import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).resolve().parent/'我的代码/05_发现需求与候选排名诊断.py'
spec = importlib.util.spec_from_file_location('diagnose_discovery', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DiagnosisChecks(unittest.TestCase):
    def test_saved_model_ranking_replayed_exactly(self):
        queries, trace, _ = m.prepare()
        self.assertEqual(queries[0]['origin'], 'saved_model_need')
        self.assertEqual(queries[0]['need'], '查找Transformer相关论文')
        self.assertEqual(queries[0]['top5'][0]['name'], 'arxiv_search')
        self.assertEqual(trace['actual_discovery_calls'], 1)

    def test_manual_query_labeled_and_visibility_preserved(self):
        queries, _, _ = m.prepare()
        self.assertEqual([q['origin'] for q in queries], [
            'saved_model_need', 'task_text_hypothetical_query', 'manual_posthoc_download_need'])
        self.assertEqual(len(queries[-1]['excluded_available_names']), 8)
        for q in queries:
            self.assertTrue(set(q['excluded_available_names']).isdisjoint(item['name'] for item in q['full_positive_ranking']))

    def test_no_overlap_no_target_rank_invented(self):
        full, _, _ = m.compare.prepare()
        index = m.compare.ToolIndex(full)
        result = m.view(index, 'zzzzznotfoundzzzzz', set(), 'test', 'test')
        self.assertEqual(result['top5'], [])
        self.assertIsNone(result['target_rank_among_positive_scores'])
        self.assertFalse(result['target_in_top_k']['20'])


if __name__ == '__main__':
    unittest.main()

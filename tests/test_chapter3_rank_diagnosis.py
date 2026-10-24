"""排名诊断的合成语料检查；不执行正式四条诊断查询。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'第三章/3.8Agentic_RAG/我的代码/10_漏检条文排名诊断.py'
spec = importlib.util.spec_from_file_location('diagnostic_step', PATH)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class RankChecks(unittest.TestCase):
    def make_retriever(self, directory, content):
        h = diag.helper()
        path = h.OFFICIAL/'offline_retriever.py'
        spec = importlib.util.spec_from_file_location('synthetic_rank_retriever', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._tokenize = h.tokenize
        Path(directory, 'sample.md').write_text(content, encoding='utf-8')
        return h, module.OfflineRetriever(directory)

    def test_source_target_presence_only(self):
        h, _, _, target = diag.prepare()
        self.assertEqual(h.leading(target['text']), '第六十五条')
        self.assertEqual(target['chunk_id'], '7-刑法/刑法_chunk_67')

    def test_query_frequency_and_contribution_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            h, retriever = self.make_retriever(tmp, 'alpha alpha beta\n\nbeta gamma')
            chunk = retriever.chunks[0]
            once = diag.contributions(retriever, 'alpha', chunk, h.tokenize)
            twice = diag.contributions(retriever, 'alpha alpha', chunk, h.tokenize)
            self.assertAlmostEqual(twice['score'], 2*once['score'])
            row = diag.diagnose(retriever, 'alpha alpha', chunk, h)
            self.assertAlmostEqual(row['target_detail']['score'], retriever.search('alpha alpha', 1)[0]['score'])

    def test_zero_score_no_invented_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            h, retriever = self.make_retriever(tmp, 'alpha beta\n\ngamma delta')
            row = diag.diagnose(retriever, 'unknown', retriever.chunks[0], h)
            self.assertIsNone(row['target_rank'])
            self.assertTrue(all(not hit for hit in row['top_k_target_hit'].values()))

    def test_top_k_cutoff_and_tie_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            h, retriever = self.make_retriever(tmp, '\n\n'.join(['alpha beta']*6))
            row = diag.diagnose(retriever, 'alpha', retriever.chunks[0], h)
            self.assertEqual(row['target_rank'], 6)
            self.assertFalse(row['top_k_target_hit']['5'])
            self.assertTrue(row['top_k_target_hit']['10'])
            self.assertEqual(len(row['top5_details']), 5)


if __name__ == '__main__':
    unittest.main()

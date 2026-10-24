"""3-8离线协议检查，查询仅使用自造语料，不运行正式七题。"""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'第三章/3.8Agentic_RAG/我的代码/01_单次与分解检索对照.py'
spec = importlib.util.spec_from_file_location('retrieval_step', PATH)
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


class ProtocolChecks(unittest.TestCase):
    def test_sources_and_gold_presence_without_queries(self):
        retriever, dataset, manifest = step.load_sources()
        self.assertEqual(len(retriever.documents), 1)
        self.assertEqual(len(dataset['cases']), 7)
        self.assertEqual(len(manifest['files']), 8)

    def test_exact_article_boundary(self):
        hits = [{'text': '第一百三十三条之一 测试'}, {'text': '第二十条 测试'}]
        result = step.evidence(hits, ['第一百三十三条', '第一百三十三条之一'])
        self.assertEqual(result['covered'], ['第一百三十三条之一'])
        self.assertEqual(result['recall'], 0.5)

    def test_decomposition_union_deduplicated(self):
        class Retriever:
            def search(self, query, top_k):
                return [{'chunk_id': 'x', 'text': '第一条 甲', 'score': 1}]
        case = {'id': 'synthetic', 'question': '甲', 'naive_query': '甲',
                'difficulty': 'test', 'subqueries': ['甲', '乙'], 'gold_articles': ['第一条']}
        row = step.run_case(Retriever(), case, 2)
        self.assertEqual(len(row['decomposed']['union_hits']), 1)
        self.assertEqual(len(row['decomposed']['steps']), 2)
        self.assertEqual(row['wider_single']['candidate_position_budget'], 4)
        self.assertEqual(row['decomposed']['evidence']['recall'], 1)

    def test_bm25_against_independent_ascii_calculation(self):
        spec = importlib.util.spec_from_file_location('synthetic_bm25', step.OFFICIAL/'offline_retriever.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._tokenize = step.tokenize
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'sample.md').write_text('alpha alpha beta\n\nbeta gamma delta', encoding='utf-8')
            retriever = module.OfflineRetriever(tmp)
            hits = retriever.search('alpha', 2)
            # 两个3词文档，alpha只在第一个出现2次。
            expected = math.log(1+(2-1+0.5)/(1+0.5))*2*2.5/(2+1.5)
            self.assertEqual(len(hits), 1)
            self.assertAlmostEqual(hits[0]['score'], expected)
            self.assertEqual(retriever.search('missing', 2), [])

    def test_macro_averaging(self):
        rows = [{'difficulty': 'easy', **{m: {'evidence': {'recall': 1}} for m in ('naive', 'decomposed', 'wider_single')}},
                {'difficulty': 'hard', **{m: {'evidence': {'recall': 1/3}} for m in ('naive', 'decomposed', 'wider_single')}}]
        self.assertAlmostEqual(step.summarize(rows)['overall']['naive_macro_recall'], 2/3)


if __name__ == '__main__':
    unittest.main()

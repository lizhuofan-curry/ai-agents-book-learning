"""3-11受控前缀检索的离线检查，不执行正式8题。"""
import importlib.util
import math
from pathlib import Path
import unittest

P=Path(__file__).resolve().parents[1]/'第三章/3.11上下文与双层记忆/我的代码/01_记忆原文与背景检索对照.py'
spec=importlib.util.spec_from_file_location('memory_ctx_test',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class MemoryContextChecks(unittest.TestCase):
    def test_frozen_dataset(self):
        _,data,_=m.prepare();self.assertEqual(len(data['chunks']),12);self.assertEqual(len(data['queries']),8)

    def test_bm25_formula_and_tie_order(self):
        module,_,_=m.prepare();index=module.BM25([['a','a','b'],['b','c','d']])
        self.assertAlmostEqual(index.score(['a'],0),math.log(2)*2*2.5/3.5)
        self.assertEqual(index.rank(['absent']),[0,1])

    def test_tokens_do_not_cross_punctuation(self):
        module,_,_=m.prepare();tokens=module.tokenize('甲乙，丙丁')
        self.assertIn('甲乙',tokens);self.assertNotIn('乙丙',tokens)

    def test_synthetic_comparison_retains_full_rankings(self):
        module,_,_=m.prepare()
        data={'chunks':[{'id':'a','text':'yes','context':'flight alpha'},
                        {'id':'b','text':'hotel beta','context':'hotel beta'}],
              'queries':[{'q':'flight','gold':'a'}]}
        rows,summary=m.compare(module,data)
        self.assertEqual(len(rows[0]['plain']['ranking']),2)
        self.assertEqual(summary['contextual']['n_queries'],1)


if __name__=='__main__':unittest.main()

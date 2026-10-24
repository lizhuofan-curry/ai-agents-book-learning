"""上下文检索离线合成检查，不执行正式15题。"""
import importlib.util
import math
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'第三章/3.10上下文感知检索/我的代码/01_原始与背景前缀检索对照.py'
spec=importlib.util.spec_from_file_location('contextual_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ContextualChecks(unittest.TestCase):
    def test_sources_and_exact_same_plain_texts(self):
        chunks,queries,_,_=m.load_sources()
        self.assertEqual(len(chunks),22);self.assertEqual(len(queries),15)
        for c in chunks:self.assertEqual(c['contextual'],c['prefix']+'\n\n'+c['plain'])

    def test_bm25_independent_small_example(self):
        chunks=[{'chunk_id':'a','plain':'alpha alpha beta'}, {'chunk_id':'b','plain':'beta gamma delta'}]
        index=m.BM25(chunks,'plain',str.split);rows=index.rank('alpha')
        self.assertEqual(rows[0]['chunk_id'],'a')
        self.assertAlmostEqual(rows[0]['score'],math.log(2)*2*2.5/3.5)
        self.assertEqual(rows[1]['score'],0)

    def test_ties_and_metrics(self):
        chunks=[{'chunk_id':'b','plain':'x'},{'chunk_id':'a','plain':'x'}]
        rows=m.BM25(chunks,'plain',str.split).rank('absent')
        self.assertEqual([r['chunk_id'] for r in rows],['a','b'])
        result=m.evaluate(rows,'b')
        self.assertEqual(result['gold_rank'],2)
        self.assertFalse(result['hits']['1']);self.assertTrue(result['hits']['3'])

    def test_prefix_can_help_or_hurt_no_preset_outcome(self):
        chunks=[{'chunk_id':'a','plain':'alpha','contextual':'alpha'},
                {'chunk_id':'b','plain':'beta','contextual':'alpha alpha alpha beta'}]
        plain=m.BM25(chunks,'plain',str.split).rank('alpha')
        ctx=m.BM25(chunks,'contextual',str.split).rank('alpha')
        self.assertEqual(plain[0]['chunk_id'],'a')
        self.assertEqual(ctx[0]['chunk_id'],'b')


if __name__=='__main__':unittest.main()

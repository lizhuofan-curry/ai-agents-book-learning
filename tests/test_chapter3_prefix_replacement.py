"""只替换一个前缀的合成验证，不运行15道正式问题。"""
import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).resolve().parents[1]/'第三章/3.10上下文感知检索/我的代码/04_单块新前缀替换检索对照.py'
spec=importlib.util.spec_from_file_location('replace_prefix',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ReplacementChecks(unittest.TestCase):
    def test_frozen_input_one_change_only(self):
        _,old,new,_,_,_,generated=m.prepare()
        self.assertEqual(sum(a!=b for a,b in zip(old,new)),1)
        self.assertEqual([c['plain'] for c in old],[c['plain'] for c in new])
        self.assertEqual(next(c for c in new if c['chunk_id']==m.TARGET)['prefix'],generated['candidate']['prefix'])

    def test_copy_not_mutation_and_invalid_target(self):
        old=[{'chunk_id':'a','plain':'original','prefix':'old','contextual':'old\n\noriginal'}]
        new=m.replace_one(old,'a','new')
        self.assertEqual(old[0]['prefix'],'old')
        self.assertEqual(new[0]['contextual'],'new\n\noriginal')
        with self.assertRaises(ValueError):m.replace_one(old,'missing','new')
        with self.assertRaises(ValueError):m.replace_one(old,'a',' ')

    def test_synthetic_three_arm_metrics(self):
        h=m.load_step()
        old=[{'chunk_id':'a','plain':'alpha','prefix':'alpha','contextual':'alpha\n\nalpha'},
             {'chunk_id':'b','plain':'beta','prefix':'beta','contextual':'beta\n\nbeta'}]
        new=m.replace_one(old,'b','alpha alpha alpha')
        rows,summary,change=m.compare(h,old,new,[{'id':'synthetic','query':'alpha','gold_chunk_id':'a'}],str.split)
        self.assertEqual(rows[0]['plain']['gold_rank'],1)
        self.assertEqual(sum(change.values()),1)
        self.assertEqual(set(summary),{'plain','historical','single_new'})


if __name__=='__main__':unittest.main()

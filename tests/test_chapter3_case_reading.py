"""3-12来源及正文隔离检查，不执行模型发现。"""
import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).resolve().parents[1]/'第三章/3.12结构化知识提取/我的代码/01_阅读合成案例与字段边界.py'
spec=importlib.util.spec_from_file_location('case_reading',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class CaseReadingChecks(unittest.TestCase):
    def test_fixed_sources(self):
        rows,_=m.load_sources();self.assertEqual(len(rows),66)

    def test_selection_and_fact_only_view(self):
        rows,_=m.load_sources();selected=m.select_samples(rows);visible=m.fact_only(selected)
        self.assertEqual([r['charge'] for r in selected],list(m.CHARGES))
        for i,row in enumerate(visible):
            self.assertEqual(set(row),{'sample_id','fact'})
            self.assertEqual(row['fact'],selected[i]['fact'])
            self.assertEqual(row['sample_id'],f'sample_{i+1}')

    def test_duplicate_ids_and_missing_text_rejected(self):
        rows,_=m.load_sources();rows[1]['id']=rows[0]['id']
        with self.assertRaises(ValueError):m.validate(rows)
        rows,_=m.load_sources();rows[0]['fact']=''
        with self.assertRaises(ValueError):m.validate(rows)


if __name__=='__main__':unittest.main()

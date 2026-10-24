"""单前缀协议离线检查，不调用本地模型。"""
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'第三章/3.10上下文感知检索/我的代码/02_单片段生成背景前缀.py'
spec=importlib.util.spec_from_file_location('single_prefix_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class SinglePrefixChecks(unittest.TestCase):
    def test_source_and_input_fields_no_evaluation_leak(self):
        h,_,source,plain=m.prepare();chunks,queries,_,_=h.load_sources()
        payload=m.make_request(source,plain)
        self.assertEqual(len(payload['messages']),2)
        self.assertEqual(payload['messages'][1]['content'],
            '完整源文档：\n<document>\n'+source+'\n</document>\n\n目标片段：\n<chunk>\n'+plain+'\n</chunk>')
        self.assertNotIn(m.TARGET,payload['messages'][1]['content'])
        self.assertNotIn(queries[8]['query'],payload['messages'][1]['content'])
        self.assertEqual(payload['options']['num_predict'],1024)

    def test_all_exact_quote_occurrences(self):
        anchors=m.check_output({'prefix':'定位','source_quotes':['甲乙']},'甲乙丙甲乙')
        self.assertEqual(anchors[0]['source_char_ranges'],[{'start':0,'end':2},{'start':3,'end':5}])

    def test_invalid_prefix_and_fabricated_quote_rejected(self):
        for data in [{'prefix':'','source_quotes':['甲']},
                     {'prefix':'a'*201,'source_quotes':['甲']},
                     {'prefix':'背景','source_quotes':['不存在']},
                     {'prefix':'背景','source_quotes':[]}]:
            with self.assertRaises(ValueError):m.check_output(data,'甲乙')

    def test_extra_and_duplicate_json_fields_rejected(self):
        with self.assertRaises(ValueError):m.unique([('prefix','a'),('prefix','b')])
        with self.assertRaises(ValueError):m.check_output({'prefix':'背景','source_quotes':['甲'],'answer':'泄露'},'甲')


if __name__=='__main__':unittest.main()

"""可追溯背景/卡片协议检查，不调用模型。"""
import importlib.util
import json
from pathlib import Path
import unittest

P=Path(__file__).resolve().parents[1]/'第三章/3.11上下文与双层记忆/我的代码/03_完整会话生成背景与记忆卡片.py'
spec=importlib.util.spec_from_file_location('memory_cards',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class CardChecks(unittest.TestCase):
    def test_source_only_prompt(self):
        session,messages,target=m.prepare();request=m.make_request(session,messages,target)
        self.assertEqual(len(messages),94)
        self.assertEqual(target['source_message_numbers'],list(range(73,95)))
        for text in ['expected_answer','gold','FS-447291','Honda','Tesla']:
            self.assertNotIn(text,request['messages'][0]['content'])
        self.assertEqual(request['options']['num_predict'],2048)

    def test_anchor_bound_to_message(self):
        messages=[{'message_number':1,'role':'user','content':'alpha beta alpha'},
                  {'message_number':2,'role':'assistant','content':'other'}]
        result=m.anchor([{'message_number':1,'quote':'alpha'}],messages)
        self.assertEqual(len(result[0]['source_char_ranges']),2)
        with self.assertRaises(ValueError):m.anchor([{'message_number':2,'quote':'alpha'}],messages)
        with self.assertRaises(ValueError):m.anchor([{'message_number':True,'quote':'alpha'}],messages)

    def test_card_schema_and_quote_validation(self):
        messages=[{'message_number':1,'role':'user','content':'yes'}]
        citation={'message_number':1,'quote':'yes'}
        data={'prefix':'背景','prefix_evidence':[citation],
              'cards':[{'subject':'test','fact':'合成事实','status':'confirmed','evidence':[citation]}],
              'uncertainties':''}
        self.assertEqual(len(m.validate(data,messages)['cards']),1)
        data['cards'][0]['status']='invented'
        with self.assertRaises(ValueError):m.validate(data,messages)
        with self.assertRaises(ValueError):m.unique([('x',1),('x',2)])


if __name__=='__main__':unittest.main()

"""冻结输入隔离及虚构依据拒绝检查；不连接模型。"""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1]/'第三章/3.12结构化知识提取/我的代码/02_三条正文发现候选字段.py'
spec = importlib.util.spec_from_file_location('factor_discovery', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DiscoveryChecks(unittest.TestCase):
    def setUp(self):
        self.rows = m.prepare()
        self.data = {'fields': [
            {'name': f'测试字段{i}', 'definition': '仅供机械校验的模拟输出', 'value_type': 'string',
             'evidence': [{'sample_id': 'sample_1', 'quote': self.rows[0]['fact'][:8]}]}
            for i in range(4)], 'uncertainties': '模拟输出，不代表实际模型结果'}

    def test_request_contains_only_reading_view(self):
        request = m.make_request(self.rows)
        visible = json.loads(request['messages'][1]['content'])
        self.assertEqual(visible, self.rows)
        for row in visible:
            self.assertEqual(set(row), {'sample_id', 'fact'})
        self.assertNotIn('34170', request['messages'][0]['content'])
        self.assertEqual(request['model'], 'qwen3.5:9b')

    def test_exact_quote_offsets(self):
        anchors = m.validate(self.data, self.rows)
        for anchor in anchors:
            self.assertEqual(self.rows[0]['fact'][anchor['start']:anchor['end']], anchor['quote'])

    def test_fabricated_and_wrong_source_quotes_rejected(self):
        for sid, quote in [('sample_1', '不在正文里的虚构依据'), ('sample_2', '34170元')]:
            candidate = copy.deepcopy(self.data)
            candidate['fields'][0]['evidence'] = [{'sample_id': sid, 'quote': quote}]
            with self.assertRaises(ValueError):
                m.validate(candidate, self.rows)

    def test_duplicate_names_and_json_keys_rejected(self):
        self.data['fields'][1]['name'] = self.data['fields'][0]['name']
        with self.assertRaises(ValueError):
            m.validate(self.data, self.rows)
        with self.assertRaises(ValueError):
            json.loads('{"fields":[],"fields":[]}', object_pairs_hook=m.unique)


if __name__ == '__main__':
    unittest.main()

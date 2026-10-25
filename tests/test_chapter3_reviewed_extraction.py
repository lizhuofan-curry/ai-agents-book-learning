import copy
import importlib.util
import json
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1]/'第三章/3.12结构化知识提取/我的代码/07_审阅字段逐案抽取.py'
spec = importlib.util.spec_from_file_location('reviewed_extraction', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExtractionChecks(unittest.TestCase):
    def setUp(self):
        self.rows, self.plan = m.prepare()
        self.data = {'cases': [{'sample_id': r['sample_id'], 'values': {
            f['name']: {'value': None, 'evidence_ids': []} for f in self.plan['fields']}} for r in self.rows],
            'uncertainties': 'mock only'}

    def test_no_source_loss_or_reference_columns(self):
        visible = json.loads(m.make_request(self.rows, self.plan)['messages'][1]['content'])
        for source, row in zip(self.rows, visible['cases']):
            self.assertEqual(''.join(s['text'] for s in row['spans']), source['fact'])
            self.assertEqual(set(row), {'sample_id', 'spans'})

    def test_canonical_quote_copy(self):
        item = self.data['cases'][0]['values']['prior_record']
        item.update(value='正文中提及的记录', evidence_ids=[self.rows[0]['spans'][1]['span_id']])
        result = m.validate(self.data, self.rows, self.plan)
        evidence = result[0]['copied_evidence']['prior_record'][0]
        self.assertEqual(self.rows[0]['fact'][evidence['start']:evidence['end']], evidence['text'])

    def test_cross_case_reference_and_boolean_number_rejected(self):
        candidate = copy.deepcopy(self.data)
        candidate['cases'][0]['values']['prior_record'].update(value='test', evidence_ids=['sample_2_s1'])
        with self.assertRaises(ValueError):
            m.validate(candidate, self.rows, self.plan)
        candidate = copy.deepcopy(self.data)
        candidate['cases'][0]['values']['amount_cny'].update(value=True, evidence_ids=['sample_1_s1'])
        with self.assertRaises(ValueError):
            m.validate(candidate, self.rows, self.plan)

    def test_duplicate_case_and_unsupported_nonnull_rejected(self):
        self.data['cases'][1]['sample_id'] = 'sample_1'
        with self.assertRaises(ValueError):
            m.validate(self.data, self.rows, self.plan)
        self.setUp()
        self.data['cases'][0]['values']['amount_cny']['value'] = 34170
        with self.assertRaises(ValueError):
            m.validate(self.data, self.rows, self.plan)


if __name__ == '__main__':
    unittest.main()

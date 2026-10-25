"""新样本隔离、固定参数与跨案例引用防护；不调用模型。"""
import importlib.util
import json
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1]/'第三章/3.12结构化知识提取/我的代码/09_新案例冻结字段抽取.py'
spec = importlib.util.spec_from_file_location('new_case_extractor', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NewCaseChecks(unittest.TestCase):
    def test_fixed_selection_and_no_overlap(self):
        source, _ = m.reader.load_sources()
        selected = m.select_new(source)
        for charge, case in zip(m.reader.CHARGES, selected):
            self.assertEqual(case, [r for r in source if r['charge'] == charge][1])
        self.assertTrue(set(r['id'] for r in selected).isdisjoint(r['id'] for r in m.reader.select_samples(source)))

    def test_unchanged_settings_schema_and_system(self):
        oldrows, oldplan = m.shared.prepare()
        rows, plan = m.prepare()
        oldrequest = m.shared.make_request(oldrows, oldplan)
        request = m.make_request(rows, plan)
        for key in ('model', 'options', 'format', 'stream', 'think'):
            self.assertEqual(oldrequest[key], request[key])
        self.assertEqual(oldrequest['messages'][0], request['messages'][0])
        self.assertEqual(plan, oldplan)

    def test_only_text_and_neutral_ids_enter_request(self):
        rows, plan = m.prepare()
        visible = json.loads(m.make_request(rows, plan)['messages'][1]['content'])
        self.assertEqual(set(visible), {'field_definitions', 'cases'})
        for row, source in zip(visible['cases'], rows):
            self.assertEqual(set(row), {'sample_id', 'spans'})
            self.assertTrue(row['sample_id'].startswith('new_'))
            self.assertEqual(''.join(s['text'] for s in row['spans']), source['fact'])
            for span in row['spans']:
                self.assertEqual(set(span), {'span_id', 'text'})

    def test_old_case_reference_rejected(self):
        rows, plan = m.prepare()
        result = {'cases': [{'sample_id': r['sample_id'], 'values': {
            f['name']: {'value': None, 'evidence_ids': []} for f in plan['fields']}} for r in rows],
            'uncertainties': 'mock'}
        result['cases'][0]['values']['amount_cny'] = {'value': 1, 'evidence_ids': ['sample_1_s5']}
        with self.assertRaises(ValueError):
            m.validate(result, rows, plan)


if __name__ == '__main__':
    unittest.main()

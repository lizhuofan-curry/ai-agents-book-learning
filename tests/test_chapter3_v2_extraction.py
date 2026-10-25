"""v2增量字段、输入隔离与人数边界检查；无模型请求。"""
import importlib.util
import json
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1]/'第三章/3.12结构化知识提取/我的代码/13_v2字段同案例抽取.py'
spec = importlib.util.spec_from_file_location('v2_case_extractor', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class V2Checks(unittest.TestCase):
    def test_old_definitions_unchanged_and_same_cases(self):
        rows, plan = m.prepare()
        oldplan = json.loads(m.shared.PLAN.read_text())
        self.assertEqual(plan['fields'][:9], oldplan['fields'])
        self.assertEqual(len(plan['fields']), 13)
        original = m.load_module('previous_new_case_entry', '09_新案例冻结字段抽取.py')
        self.assertEqual(rows, original.prepare()[0])

    def test_no_previous_answers_or_gold_in_input(self):
        rows, plan = m.prepare()
        request = m.make_request(rows, plan)
        visible = json.loads(request['messages'][1]['content'])
        self.assertEqual(set(visible), {'field_definitions', 'cases'})
        for case, source in zip(visible['cases'], rows):
            self.assertEqual(set(case), {'sample_id', 'spans'})
            self.assertEqual(''.join(s['text'] for s in case['spans']), source['fact'])
        self.assertEqual(request['options'], m.shared.make_request(rows, plan)['options'])

    def test_victim_count_positive_integer_only(self):
        rows, plan = m.prepare()
        for value in (0, -1, 1.5, True):
            candidate = {'cases': [{'sample_id': r['sample_id'], 'values': {
                f['name']: {'value': None, 'evidence_ids': []} for f in plan['fields']}} for r in rows],
                'uncertainties': 'mock'}
            candidate['cases'][2]['values']['victim_count_reported'] = {
                'value': value, 'evidence_ids': ['new_3_s7']}
            with self.assertRaises(ValueError):
                m.validate(candidate, rows, plan)


if __name__ == '__main__':
    unittest.main()

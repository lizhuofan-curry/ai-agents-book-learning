"""成本公式、缓存计数与未知本机金额检查；只用公开样例或临时合成数据。"""
from copy import deepcopy
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cost_step', ROOT/'我的代码/01_官方四场景成本离线复算.py')
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)
PRICES = {'input':1, 'cached':.1, 'output':4}


class CostChecks(unittest.TestCase):
    def test_cached_input_is_subtracted_before_charging_input(self):
        value = step.span_cost({'prompt_tokens':100,'cached_tokens':40,'completion_tokens':10}, PRICES)
        self.assertEqual(value, Decimal('0.000104'))

    def test_unknown_negative_boolean_and_excess_cache_rejected(self):
        valid = {'prompt_tokens':100,'cached_tokens':40,'completion_tokens':10}
        for key, value in [('prompt_tokens',None),('cached_tokens',101),('cached_tokens',-1),('completion_tokens',True)]:
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                step.span_cost({**valid,key:value}, PRICES)

    def test_price_change_only_changes_arithmetic(self):
        span = {'prompt_tokens':100,'cached_tokens':40,'completion_tokens':10}
        before = dict(span)
        a = step.span_cost(span, PRICES)
        b = step.span_cost(span, {**PRICES,'output':8})
        self.assertEqual(b-a, Decimal('0.00004'))
        self.assertEqual(span, before)

    def test_public_sample_and_corrupted_cost_component(self):
        sample = json.loads((step.OFFICIAL/'sample_trace.json').read_text(encoding='utf-8'))
        prices = sample['pricing']
        rows = step.verify_official_output(sample, sample, prices)
        self.assertEqual([r['key'] for r in rows], ['naive','kv','compress','both'])
        self.assertAlmostEqual(rows[0]['recomputed_scenario_usd'], .0037758)
        wrong = deepcopy(sample)
        wrong['scenarios'][0]['component_costs']['uncached_input_tokens'] += 1
        with self.assertRaises(ValueError):
            step.verify_official_output(sample, wrong, prices)

    def test_local_cache_split_money_and_missing_time_stay_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            folder = project/'第三章/个人Wiki混合RAG/本地运行结果/example'
            folder.mkdir(parents=True)
            response = {'total_duration':None}
            record = {'model_calls':[{'query_id':f'Q{i}', 'response':response,
                'raw_http_body':json.dumps(response),'prompt_tokens_reported':10,
                'output_tokens_reported':2,'answer':{'status':'insufficient'}} for i in range(1,4)]}
            review = {'question_reviews':[{'query_id':f'Q{i}','review_outcome':'synthetic'} for i in range(1,4)]}
            record_path, review_path = folder/'record.json', folder/'review.json'
            record_path.write_text(json.dumps(record));review_path.write_text(json.dumps(review))
            plan = {'source_record':str(record_path.relative_to(project)),
                    'source_record_sha256':step.sha(record_path.read_bytes()),
                    'source_review':str(review_path.relative_to(project)),
                    'source_review_sha256':step.sha(review_path.read_bytes())}
            with patch.object(step, 'PROJECT', project):
                result, _ = step.load_personal(plan)
            self.assertEqual(result['total_reported_prompt_tokens'], 30)
            self.assertIsNone(result['server_duration_sum_s'])
            self.assertIsNone(result['rows'][0]['cached_input_tokens'])
            self.assertIsNone(result['rows'][0]['local_billed_money'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

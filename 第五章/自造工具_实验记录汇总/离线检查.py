"""汇总口径检查；真实记录只读，失败样本为内存合成数据，不冒充模型实验。"""
from copy import deepcopy
import hashlib
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'我的代码'))
from 实验记录工具 import SOURCE_ROOT, TOOL_DEFINITION, build_summary, checks_summary, dispatch_tool, source_path


def sample():
    return {'status': 'review_pending', 'model_calls': [
                {'status': 'completed', 'prompt_tokens_reported': 10, 'output_tokens_reported': 2},
                {'status': 'incomplete', 'prompt_tokens_reported': None}],
            'events': [{'type': 'tool_call', 'tool': 'Bash'}, {'type': 'tool_execution_start', 'tool': 'Bash'},
                       {'type': 'tool_execution_complete', 'tool': 'Bash', 'result': {'exit_code': 1, 'output': '失败'}}],
            'completion_checks': {'done': False}, 'maximum_model_requests': 2}


class SummaryChecks(unittest.TestCase):
    def test_actual_records_have_known_counts_and_no_changes(self):
        cases = [
            ('local_coding_task_20261101_161002_541836', 3, {'Write': 1, 'Bash': 1}, 20883, 224, 6),
            ('local_coding_revision_20261101_162450_914560', 6, {'Read': 1, 'Edit': 2, 'Bash': 2}, 45101, 545, 8),
        ]
        for name, count, tools, prompt, output, checks in cases:
            with self.subTest(name=name):
                path = SOURCE_ROOT/name/'record.json'
                before = path.read_bytes()
                result = dispatch_tool(TOOL_DEFINITION['name'], {'record_path': str(path)})
                self.assertEqual(result['model_requests']['total_started'], count)
                self.assertEqual(result['tool_events']['returned_results_by_tool'], tools)
                self.assertEqual(result['tool_events']['returned_results'], sum(tools.values()))
                self.assertEqual(result['usage_reported']['prompt_tokens']['reported_sum'], prompt)
                self.assertEqual(result['usage_reported']['output_tokens']['reported_sum'], output)
                self.assertEqual(len(result['saved_completion_checks']['passed']), checks)
                self.assertIn('review_pending', result['source_status'])
                self.assertEqual(result['source_sha256'], hashlib.sha256(before).hexdigest())
                self.assertEqual(path.read_bytes(), before)

    def test_partial_usage_and_unfinished_request_not_hidden(self):
        record = sample()
        before = deepcopy(record)
        result = build_summary(record)
        self.assertEqual(result['model_requests']['total_started'], 2)
        self.assertEqual(result['model_requests']['completed'], 1)
        self.assertEqual(result['model_requests']['not_completed_or_unknown'], 1)
        self.assertEqual(result['usage_reported']['prompt_tokens'],
                         {'reported_sum': 10, 'calls_without_valid_count': 1, 'coverage': 'partial'})
        self.assertEqual(result['usage_reported']['output_tokens']['calls_without_valid_count'], 1)
        self.assertTrue(any('未完成' in w for w in result['warnings']))
        self.assertEqual(record, before)

    def test_empty_missing_and_truthy_values_are_not_passes(self):
        self.assertIsNone(checks_summary(None)['all_recorded_true'])
        self.assertFalse(checks_summary({})['all_recorded_true'])
        result = checks_summary({'ok': True, 'bad': False, 'text': 'true', 'number': 1})
        self.assertEqual(result['passed'], ['ok'])
        self.assertEqual(result['failed'], ['bad'])
        self.assertEqual(result['unknown'], ['text', 'number'])
        self.assertFalse(result['all_recorded_true'])
        self.assertFalse(build_summary(sample())['saved_completion_checks']['all_recorded_true'])

    def test_selected_started_and_returned_are_different(self):
        record = sample()
        record['events'].extend([{'type': 'tool_call', 'tool': 'Read'},
                                 {'type': 'tool_execution_start', 'tool': 'Read'}])
        result = build_summary(record)['tool_events']
        self.assertEqual(result['model_selections'], 2)
        self.assertEqual(result['execution_starts'], 2)
        self.assertEqual(result['returned_results'], 1)
        self.assertEqual(result['nonzero_exit_results'], 1)
        self.assertEqual(result['error_results'], 0)
        record['events'][2]['result']['error'] = '合成工具错误'
        self.assertEqual(build_summary(record)['tool_events']['error_results'], 1)

    def test_exact_budget_not_a_budget_stop(self):
        record = sample()
        result = build_summary(record)
        self.assertTrue(result['model_requests']['at_declared_limit'])
        self.assertFalse(result['model_requests']['exceeds_declared_limit'])
        self.assertFalse(result['model_requests']['budget_stop_event'])
        record['events'].append({'type': 'max_iterations_reached'})
        self.assertTrue(build_summary(record)['model_requests']['budget_stop_event'])

    def test_input_paths_and_dispatch_are_scoped(self):
        with self.assertRaises(ValueError):
            source_path('record.json')
        with self.assertRaises(ValueError):
            source_path(str(ROOT/'record.json'))
        with self.assertRaises(ValueError):
            source_path(str(SOURCE_ROOT/'../record.json'))
        with self.assertRaises(ValueError):
            dispatch_tool('not_registered', {'record_path': 'unused'})
        with self.assertRaises(ValueError):
            dispatch_tool(TOOL_DEFINITION['name'], {'record_path': 'unused', 'extra': True})

    def test_other_record_shapes_rejected(self):
        for value in ([], {}, {'model_calls': {}, 'events': []}, {'model_calls': [], 'events': ['bad']}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_summary(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)

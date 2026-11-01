"""交接请求隔离与失败留档检查；全部使用假响应，不请求真实模型。"""
import contextlib
from copy import deepcopy
from datetime import date
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'task_handoff', ROOT/'第四章/4.5协作工具/我的代码/08_任务交接信息对照.py')
step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(step)


def response(decision='need_info', ids=None, reason='stop'):
    return {'model': step.MODEL, 'done': True, 'done_reason': reason,
            'message': {'role': 'assistant', 'content': json.dumps({
                'decision': decision, 'reason': 'SYNTHETIC_RESPONSE_MARKER',
                'evidence_ids': ids or [], 'missing_information': ['测试缺口'] if decision == 'need_info' else [],
            })}, 'prompt_eval_count': 10, 'eval_count': 20}


class HandoffChecks(unittest.TestCase):
    def test_independent_requests_no_cross_arm_answer(self):
        record = {'materials': step.make_materials('within_window'), 'arms': {}, 'status': 'running'}
        requests = []
        def fake(payload, entry, save):
            requests.append(deepcopy(payload))
            return response() if len(requests) == 1 else response('eligible', ['R1', 'O1', 'O2', 'O3'])
        with contextlib.redirect_stdout(io.StringIO()):
            step.execute(record, lambda: None, fake)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]['messages'][0], requests[1]['messages'][0])
        self.assertEqual([m['role'] for m in requests[1]['messages']], ['system', 'user'])
        self.assertNotIn('SYNTHETIC_RESPONSE_MARKER', json.dumps(requests[1]))
        self.assertNotIn('2026-10-20', requests[0]['messages'][1]['content'])
        self.assertNotIn('eligible', requests[1]['messages'][1]['content'])
        self.assertEqual(record['status'], 'both_answered_review_pending')

    def test_unseen_citation_rejected(self):
        with self.assertRaisesRegex(ValueError, '没有收到'):
            step.inspect_response(response('eligible', ['O1']), [])

    def test_truncation_saved_and_no_retry(self):
        record = {'materials': step.make_materials('within_window'), 'arms': {}}
        snapshots = []
        calls = []
        def fake(payload, entry, save):
            calls.append(payload)
            return response(reason='length')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, '结束状态'):
            step.execute(record, lambda: snapshots.append(deepcopy(record)), fake)
        self.assertEqual(len(calls), 1)
        arm = snapshots[-1]['arms']['title_only']
        self.assertEqual(arm['status'], 'incomplete')
        self.assertEqual(arm['response']['done_reason'], 'length')

    def test_transport_failure_keeps_first_arm_and_partial_http(self):
        record = {'materials': step.make_materials('within_window'), 'arms': {}}
        calls = []
        def fake(payload, entry, save):
            calls.append(payload)
            if len(calls) == 2:
                entry.update(http_status=503, raw_http_body='synthetic failure')
                raise OSError('synthetic transport error')
            return response()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError):
            step.execute(record, lambda: None, fake)
        self.assertEqual(len(calls), 2)
        self.assertEqual(record['arms']['title_only']['status'], 'answered_review_pending')
        self.assertEqual(record['arms']['manual_context']['raw_http_body'], 'synthetic failure')

    def test_variant_only_changes_purchase_date(self):
        normal = step.make_materials('within_window')
        overdue = step.make_materials('overdue')
        self.assertEqual([i for i in range(4) if normal[i] != overdue[i]], [1])
        self.assertEqual((date(2026, 11, 1)-date(2026, 10, 20)).days, 12)
        self.assertEqual((date(2026, 11, 1)-date(2026, 10, 10)).days, 22)

    def test_preview_no_model_or_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(step, 'ROOT', Path(directory)), patch.object(step, 'execute') as run:
                with patch('sys.argv', ['handoff', '--preview']), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(step.main(), 0)
                run.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()

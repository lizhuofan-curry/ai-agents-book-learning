"""独立接收方输入隔离与规则核验；假响应，不新增真实模型调用。"""
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'generated_receiver', ROOT/'第四章/4.5协作工具/我的代码/15_模型交接稿接收与核验.py')
step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(step)


def response(decision='eligible', ids=None, reason='stop'):
    answer = {'decision': decision, 'reason': '合成测试理由，不证明语义正确',
              'evidence_ids': ids if ids is not None else ['H2','H4','H5','H6'], 'missing_information': []}
    return {'model': step.transport.MODEL, 'done': True, 'done_reason': reason,
            'message': {'role': 'assistant', 'content': json.dumps(answer,ensure_ascii=False)},
            'prompt_eval_count': 100, 'eval_count': 50}


class GeneratedReceiverChecks(unittest.TestCase):
    def setUp(self):
        self.generated, self.facts = step.prepare()

    def record(self):
        return {'handoff_as_received': deepcopy(self.generated['handoff']), 'original_rule_facts': deepcopy(self.facts)}

    def test_payload_exact_handoff_without_original_sources_or_expected_answer(self):
        payload = step.make_payload(self.generated['handoff'])
        content = payload['messages'][1]['content']
        self.assertEqual([m['role'] for m in payload['messages']], ['system','user'])
        self.assertTrue(content.endswith(json.dumps(self.generated['handoff'],ensure_ascii=False)))
        self.assertNotIn('TRAIN-07', content)
        self.assertNotIn('expected_decision', content)
        self.assertNotIn('source_messages', content)
        self.assertNotIn('12 天', content)

    def test_one_call_and_immutable_handoff(self):
        record = self.record()
        before = deepcopy(record['handoff_as_received'])
        calls = []
        def fake(payload, entry, save):
            calls.append(payload)
            return response()
        with contextlib.redirect_stdout(io.StringIO()):
            step.receive(record,lambda:None,fake)
        self.assertEqual(len(calls),1)
        self.assertEqual(record['handoff_as_received'],before)
        self.assertEqual(record['decision_verification']['gate'],'decision_verified_reason_review_pending')
        self.assertFalse(record['decision_verification']['can_execute_action'])
        self.assertFalse(record['decision_verification']['can_forward_full_result'])

    def test_wrong_decision_blocked_using_original_rule(self):
        answer = {'decision':'ineligible','reason':'合成','evidence_ids':['H2','H4','H5','H6'],'missing_information':[]}
        checked = step.check_decision(answer,self.generated['handoff'],self.facts)
        self.assertEqual(checked['gate'],'blocked_decision_mismatch')
        self.assertEqual(answer['decision'],'ineligible')

    def test_missing_supporting_citations_blocked(self):
        answer = {'decision':'eligible','reason':'合成','evidence_ids':['H2'],'missing_information':[]}
        self.assertEqual(step.check_decision(answer,self.generated['handoff'],self.facts)['gate'],'blocked_missing_supporting_citations')

    def test_unseen_source_reference_rejected_with_raw_response_kept(self):
        record = self.record()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            step.receive(record,lambda:None,lambda *args:response(ids=['S2']))
        self.assertEqual(record['status'],'incomplete')
        self.assertIn('response',record['receiver_call'])

    def test_truncation_no_retry(self):
        record = self.record()
        calls = []
        def fake(*args):
            calls.append(1)
            return response(reason='length')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            step.receive(record,lambda:None,fake)
        self.assertEqual(len(calls),1)
        self.assertEqual(record['receiver_call']['response']['done_reason'],'length')

    def test_preview_no_model_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(step,'ROOT',Path(directory)), patch.object(step,'prepare',return_value=(self.generated,self.facts)):
                with patch.object(step,'receive') as receive, patch('sys.argv',['receiver','--preview']), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(step.main(),0)
                    receive.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()),[])


if __name__ == '__main__':
    unittest.main()

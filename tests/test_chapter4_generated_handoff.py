"""交接稿生成阶段隔离、来源引用及失败留档检查；无真实模型调用。"""
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
    'generated_handoff', ROOT/'第四章/4.5协作工具/我的代码/13_模型整理交接背景.py')
step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(step)


def draft(source):
    return {'handoff_task': source[0]['text'],
            'materials': [{'id': 'H'+str(i), 'text': item['text'], 'source_ids': [item['id']]}
                          for i,item in enumerate(source[1:5], 1)], 'missing_information': []}


def response(result, reason='stop'):
    return {'model': step.transport.MODEL, 'done': True, 'done_reason': reason,
            'message': {'role': 'assistant', 'content': json.dumps(result,ensure_ascii=False)},
            'prompt_eval_count': 100, 'eval_count': 200}


class GeneratedHandoffChecks(unittest.TestCase):
    def test_only_source_materials_not_old_answers_or_evaluations(self):
        source, _ = step.prepare()
        old = json.loads(step.BASE.read_text())
        payload = step.make_payload(source)
        content = payload['messages'][1]['content']
        self.assertEqual(len(source), 7)
        self.assertEqual([x['id'] for x in source], ['S1','S2','S3','S4','S5','S6','S7'])
        self.assertNotIn(old['arms']['manual_context']['answer']['reason'], content)
        self.assertNotIn('decision', content)
        self.assertEqual([m['role'] for m in payload['messages']], ['system','user'])

    def test_single_generation_does_not_call_receiver_or_forward(self):
        source, _ = step.prepare()
        record = {'source_messages': source, 'can_forward_handoff': False, 'receiver_requests': 0}
        calls = []
        def fake(payload, entry, save):
            calls.append(deepcopy(payload))
            return response(draft(source))
        with contextlib.redirect_stdout(io.StringIO()):
            step.generate(record, lambda: None, fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(record['receiver_requests'], 0)
        self.assertFalse(record['can_forward_handoff'])
        self.assertEqual(record['status'], 'generated_handoff_review_pending')

    def test_unknown_source_and_duplicate_fact_ids_rejected(self):
        source, _ = step.prepare()
        for kind in ['unknown', 'duplicate']:
            result = draft(source)
            if kind == 'unknown':
                result['materials'][0]['source_ids'] = ['S999']
            else:
                result['materials'][1]['id'] = result['materials'][0]['id']
            with self.assertRaises(ValueError):
                step.inspect_handoff(response(result), source)

    def test_omission_reported_not_filled_from_original(self):
        source, _ = step.prepare()
        result = draft(source)
        result['materials'].pop(2)
        parsed, checks = step.inspect_handoff(response(result), source)
        self.assertEqual(checks['required_fact_source_ids_not_cited'], ['S4'])
        self.assertEqual(parsed, result)
        self.assertEqual(len(parsed['materials']), 3)

    def test_existing_citations_not_claimed_semantically_correct(self):
        source, _ = step.prepare()
        result = draft(source)
        result['materials'][1]['text'] = '订单DEMO-001，购买日期为2000-01-01。'
        parsed, checks = step.inspect_handoff(response(result), source)
        self.assertEqual(parsed['materials'][1]['text'], result['materials'][1]['text'])
        self.assertIn('pending human review', checks['semantic_faithfulness'])

    def test_truncation_response_preserved_without_retry(self):
        source, _ = step.prepare()
        record = {'source_messages': source}
        calls = []
        saved = []
        def fake(payload, entry, save):
            calls.append(payload)
            return response(draft(source), reason='length')
        with self.assertRaises(ValueError):
            step.generate(record, lambda: saved.append(deepcopy(record)), fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(saved[-1]['generation_call']['response']['done_reason'], 'length')
        self.assertEqual(record['status'], 'incomplete')

    def test_preview_does_not_generate_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(step, 'ROOT', Path(directory)), patch.object(step, 'generate') as generate:
                with patch('sys.argv', ['generated_handoff', '--preview']), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(step.main(), 0)
                generate.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()

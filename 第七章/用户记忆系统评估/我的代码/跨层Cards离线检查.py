"""离线假HTTP/故障检查；所有卡片和答案均为测试夹具，不是模型结果。"""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest

import 本机记忆适配 as base
import 批量JSONCards适配 as batch


def run(official, selected, plan, provenance, network):
    from transformers import AutoTokenizer
    cache = batch.verify_inputs()[1]
    tokenizer = AutoTokenizer.from_pretrained(str(cache), local_files_only=True, trust_remote_code=False)
    by_prompt = {x['card_prompt_sha256']: c for x, c in zip(plan['cases'], selected)}
    seen = []

    def fake_post(payload):
        seen.append(deepcopy(payload))
        if payload.get('format') == 'json':
            import hashlib
            case = by_prompt[hashlib.sha256(payload['messages'][0]['content'].encode()).hexdigest()]
            count = selected.index(case) + 1  # 各题1/2/3卡，不依赖旧11/14数量。
            cards = [{'category': 'offline_test_fixture', 'card_key': f'{case.test_id}:{i}',
                'backstory': 'Fake HTTP fixture, not a model-produced memory.', 'date_created': '',
                'person': 'fixture', 'relationship': 'fixture', 'facts': {'fixture_only': i},
                'source_conversation_ids': [case.conversation_histories[0].conversation_id],
                'status': 'uncertain', 'memory_tier': 'core'} for i in range(count)]
            content = json.dumps({'cards': cards}, ensure_ascii=False)
        else:
            content = 'Offline test fixture answer. Not a measured model response.'
        count = len(tokenizer.apply_chat_template(payload['messages'], tokenize=True,
            add_generation_prompt=True, enable_thinking=False))
        return 200, json.dumps({'model': base.MODEL, 'done': True, 'done_reason': 'stop',
            'message': {'role': 'assistant', 'content': content}, 'prompt_eval_count': count, 'eval_count': 1})

    class Tests(unittest.TestCase):
        def setUp(self):
            seen.clear()
            self.temp = tempfile.TemporaryDirectory(prefix='cards_batch_offline_')
            self.directory = Path(self.temp.name)
            self.record = batch.fresh_record(plan, provenance)
            self.path = self.directory / 'record.json'

        def tearDown(self):
            self.temp.cleanup()

        def save(self):
            batch.persist(self.path, self.record)

        def execute(self, post=fake_post, tok=tokenizer):
            with redirect_stdout(io.StringIO()):
                batch.run_cases(official, selected, self.record, plan, tok, self.directory, self.save, lambda: None, post)

        def test_01_variable_cards_isolated_inputs_six_request_cap(self):
            self.execute()
            self.assertEqual(len(seen), 6)
            self.assertEqual(self.record['summary']['answers_saved'], 3)
            self.assertEqual([len(r['cards']) for r in self.record['cases']], [1, 2, 3])
            for c, row in zip(selected, self.record['cases']):
                self.assertEqual([x['stage'] for x in row['model_calls']], ['card_build', 'cards_answer'])
                request = row['model_calls'][1]['request']
                self.assertEqual(request['messages'], [{'role': 'system', 'content': official.SYSTEM_RULES
                    + '\nResident Advanced JSON Cards:\n' + json.dumps(row['cards'], ensure_ascii=False)},
                    {'role': 'user', 'content': c.user_question}])
                self.assertTrue(all(row['checks'].values()))
            self.assertIsNone(self.record['summary']['formal_task_pass_rate'])
            self.assertIsNone(self.record['summary']['local_billed_money'])

        def test_02_resume_completed_sends_nothing(self):
            self.execute()
            reloaded = batch.load_checkpoint(self.path)
            batch.check_resume(reloaded, plan, provenance, self.directory)
            self.record = reloaded
            seen.clear()
            self.execute()
            self.assertEqual(seen, [])

        def test_03_interrupted_started_request_is_never_resent(self):
            def interrupt(payload):
                seen.append(payload)
                raise KeyboardInterrupt('Simulated interruption after request reservation')
            with self.assertRaises(KeyboardInterrupt):
                self.execute(interrupt)
            self.assertEqual(len(seen), 1)
            self.record = batch.load_checkpoint(self.path)
            self.assertEqual(self.record['cases'][0]['model_calls'][0]['status'], 'started')
            batch.check_resume(self.record, plan, provenance, self.directory)
            seen.clear()
            self.execute()
            self.assertEqual(len(seen), 4)
            self.assertEqual(self.record['cases'][0]['status'], 'interrupted_not_retried')
            self.assertEqual(self.record['summary']['reserved_model_requests'], 5)

        def test_04_dynamic_answer_overlength_blocks_before_http(self):
            class DynamicTooLong:
                def apply_chat_template(self, messages, **kwargs):
                    return [0] * 16384 if len(messages) == 2 else tokenizer.apply_chat_template(messages, **kwargs)
            self.execute(tok=DynamicTooLong())
            self.assertEqual(len(seen), 3)  # 提取已发生；三个回答均未发送。
            self.assertTrue(all(len(r['model_calls']) == 1 and 'cards' in r for r in self.record['cases']))
            self.assertEqual(self.record['summary']['answers_saved'], 0)

        def test_05_http_truncation_json_usage_failures_no_retry(self):
            mutations = ['http', 'truncated', 'bad_json', 'missing_usage', 'unknown_source', 'wrong_model', 'count_drift']
            for kind in mutations:
                with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                    self.directory = Path(d)
                    self.path = self.directory / 'record.json'
                    self.record = batch.fresh_record(plan, provenance)
                    seen.clear()
                    def faulty(payload):
                        status, raw = fake_post(payload)
                        if kind == 'http':
                            return 503, 'offline fixture unavailable'
                        r = json.loads(raw)
                        if kind == 'truncated': r['done_reason'] = 'length'
                        if kind == 'bad_json': r['message']['content'] = '{'
                        if kind == 'missing_usage': r.pop('prompt_eval_count')
                        if kind == 'wrong_model': r['model'] = 'unexpected'
                        if kind == 'count_drift': r['prompt_eval_count'] += 1
                        if kind == 'unknown_source':
                            cards = json.loads(r['message']['content'])
                            cards['cards'][0]['source_conversation_ids'] = ['absent']
                            r['message']['content'] = json.dumps(cards)
                        return status, json.dumps(r)
                    self.execute(faulty)
                    self.assertEqual(len(seen), 3)
                    self.assertTrue(all(len(r['model_calls']) == 1 for r in self.record['cases']))
                    self.assertTrue(all(r['model_calls'][0]['status'] == 'failed_or_incomplete' for r in self.record['cases']))
                    self.assertTrue(all('raw_http_body' in r['model_calls'][0] for r in self.record['cases']))

        def test_06_checkpoint_tamper_and_protocol_drift_rejected(self):
            self.execute()
            obj = json.loads(self.path.read_text())
            obj['cases'][0]['model_calls'] = []
            self.path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError): batch.load_checkpoint(self.path)
            changed = deepcopy(provenance)
            changed['plan_sha256'] = 'wrong'
            with self.assertRaises(ValueError): batch.check_resume(self.record, plan, changed, self.directory)
            (self.directory / self.record['cases'][0]['case_id'] / 'answer.txt').write_text('changed')
            with self.assertRaises(ValueError): batch.check_resume(self.record, plan, provenance, self.directory)

        def test_07_stage_and_total_budget_cannot_dispatch_again(self):
            self.execute()
            before = len(seen)
            chat = batch.BatchChat(official, selected[0], self.record['cases'][0], self.record,
                plan, tokenizer, self.save, lambda: None, fake_post)
            with self.assertRaises(base.StopPipeline): chat.complete([], json_object=True)
            self.assertEqual(len(seen), before)

        def test_08_strict_json_and_unknown_usage_not_zero(self):
            for text in ['{"cards":[],"cards":[]}', '{"x":NaN}', '{"x":Infinity}']:
                with self.assertRaises(ValueError): batch.strict_json(text)
            self.record['cases'] = [{'status': 'interrupted_not_retried', 'model_calls': [{'status': 'started'}]}]
            batch.summarize(self.record)
            self.assertEqual(self.record['summary']['prompt_tokens']['unknown_calls'], 1)
            self.assertEqual(self.record['summary']['completed_model_requests'], 0)

    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    path = base.ROOT / '本地运行结果' / ('cards_batch_offline_' + datetime.now(batch.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    path.mkdir(exist_ok=False)
    record = {'actor': 'Codex offline fault/fixture checks; not a model experiment or human task review',
        'provenance': provenance, 'test_script_sha256': base.sha(Path(__file__)),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'all_passed': result.wasSuccessful(), 'generation_requests': 0, 'judge_requests': 0,
        'embedding_forward_batches': 0, 'network_audit': network, 'test_output': output.getvalue()}
    base.dump(path / 'record.json', record)
    print(output.getvalue(), end='')
    print('CARDS_BATCH_OFFLINE_SAVED：仅假HTTP检查；0次模型/网络请求。记录：' + str(path / 'record.json'))
    return 0 if result.wasSuccessful() else 1

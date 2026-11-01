"""本机适配的离线回归：假响应、不连接Ollama、不调用裁判。"""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ['PYTHON_DOTENV_DISABLED'] = '1'
sys.path.insert(0, str(Path(__file__).parent / '我的代码'))
import 本机记忆适配 as local


def fake_response(content, *, reason='stop', tokens=True):
    value = {'model': local.MODEL, 'done': True, 'done_reason': reason,
        'message': {'role': 'assistant', 'content': content},
        'prompt_eval_count': 100, 'eval_count': 50}
    if not tokens:
        del value['prompt_eval_count']
    return 200, json.dumps(value)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = local.load_official()
        cls.case = cls.official.UserMemoryEvaluationFramework().get_test_case('layer1_01_bank_account')
        cls.plan = local.load_plan()

    def cards(self):
        return [{'category': 'accounts', 'card_key': 'checking', 'backstory': 'Offline fixture',
            'date_created': '2024-11-15', 'person': 'Michael James Robertson', 'relationship': 'self',
            'facts': {'checking_account_number': '4429853327'},
            'source_conversation_ids': ['bank_setup_001'], 'status': 'current', 'memory_tier': 'core'}]

    def run_fake(self, replies):
        record = {'model_calls': []}
        requests = []

        def post(payload):
            requests.append(deepcopy(payload))
            if len(requests) > len(replies):
                self.fail('unexpected extra HTTP request')
            return replies[len(requests) - 1]

        with tempfile.TemporaryDirectory(prefix='memory-adapter-offline-') as name:
            directory = Path(name)
            save = lambda: local.dump(directory / 'record.json', record)
            caught = None
            try:
                with redirect_stdout(io.StringIO()):
                    local.run_cards_path(self.official, self.case, self.plan, record, save, directory, post)
            except (Exception, local.StopPipeline) as exc:
                caught = exc
            saved_cards = json.loads((directory / 'cards.json').read_text()) if (directory / 'cards.json').exists() else None
        return record, requests, caught, saved_cards

    def test_two_request_path_isolates_question_and_preserves_unknown_cost(self):
        cards = self.cards()
        record, requests, error, saved = self.run_fake([
            fake_response(json.dumps({'cards': cards})), fake_response('Your checking account is 4429853327.')])
        self.assertIsNone(error)
        self.assertEqual(len(requests), 2)
        self.assertEqual(saved, {'cards': cards})
        self.assertTrue(all(record['checks'].values()))
        self.assertNotIn(self.case.user_question, requests[0]['messages'][0]['content'])
        self.assertNotIn(self.case.evaluation_criteria, requests[0]['messages'][0]['content'])
        self.assertEqual(len(requests[1]['messages']), 2)
        self.assertEqual(requests[1]['messages'][1]['content'], self.case.user_question)
        self.assertEqual(record['reported_usage']['prompt_tokens'], 200)
        self.assertIsNone(record['reported_usage']['cached_input_tokens'])
        self.assertIsNone(record['agent_result']['local_billed_money'])
        self.assertNotIn('cost_usd', record['agent_result'])

    def test_bad_extraction_never_sends_second_request(self):
        for reply in [fake_response('not JSON'), fake_response('{}'), (503, 'offline error'),
                fake_response('{"cards":[]}', reason='length'), fake_response('{"cards":[]}', tokens=False)]:
            with self.subTest(reply=reply):
                record, requests, error, _ = self.run_fake([reply])
                self.assertIsInstance(error, local.StopPipeline)
                self.assertEqual(len(requests), 1)
                self.assertEqual(len(record['model_calls']), 1)
                self.assertEqual(len(record['blocked_dispatches']), 1)

    def test_unknown_source_id_stops_before_answer(self):
        cards = self.cards()
        cards[0]['source_conversation_ids'] = ['invented-conversation']
        record, requests, error, _ = self.run_fake([fake_response(json.dumps({'cards': cards}))])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(requests), 1)
        self.assertNotIn('answer', record)

    def test_empty_cards_stop_before_answer(self):
        record, requests, error, _ = self.run_fake([fake_response('{"cards":[]}')])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(requests), 1)
        self.assertNotIn('answer', record)

    def test_answer_truncation_preserves_cards_and_raw_response(self):
        record, requests, error, saved = self.run_fake([
            fake_response(json.dumps({'cards': self.cards()})), fake_response('unfinished', reason='length')])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(requests), 2)
        self.assertIsNotNone(saved)
        self.assertEqual(record['model_calls'][1]['status'], 'failed_or_incomplete')
        self.assertIn('unfinished', record['model_calls'][1]['raw_http_body'])
        self.assertNotIn('answer', record)

    def test_third_dispatch_is_blocked_without_http(self):
        record = {'model_calls': [{'stage': 'card_build'}, {'stage': 'cards_answer'}]}
        chat = local.LocalChat(self.official, self.plan, record, lambda: None,
            lambda payload: self.fail('budget must stop before HTTP'))
        chat.stage = 'cards_answer'
        with self.assertRaises(local.StopPipeline):
            chat.complete([])


if __name__ == '__main__':
    print('离线适配检查：全部使用假HTTP响应，0次真实模型或网络请求。', flush=True)
    def no_network(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            raise RuntimeError('Tests prohibit all network access')
    sys.addaudithook(no_network)
    unittest.main()

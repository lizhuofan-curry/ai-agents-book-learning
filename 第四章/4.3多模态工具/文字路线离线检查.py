"""只检查请求隔离、顺序、预算与失败留档；使用假HTTP响应，不连接模型。"""
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('text_route', ROOT/'我的代码/04_先提取文字再回答.py')
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)


class FakeSession:
    requests = []
    responses = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        assert self.trust_env is False and kwargs['allow_redirects'] is False
        type(self).requests.append(deepcopy(kwargs['json']))
        status, body = type(self).responses.pop(0)
        return type('Response', (), {'status_code': status, 'text': body})()


def reply(text, reason='stop'):
    return (200, json.dumps({'model': route.native.MODEL, 'done': True, 'done_reason': reason,
                            'message': {'role': 'assistant', 'content': text},
                            'prompt_eval_count': 10, 'eval_count': 5}))


def fake_call(request, stage, save):
    return route.native.ask(request, stage, save, FakeSession)


class Checks(unittest.TestCase):
    def setUp(self):
        FakeSession.requests = []
        FakeSession.responses = []

    def test_extraction_uses_original_image_without_question_or_answer(self):
        picture = route.native.load_input(ROOT)
        request = route.extraction_payload(picture)
        self.assertEqual(base64.b64decode(request['messages'][1]['images'][0], validate=True), picture)
        texts = ''.join(item['content'] for item in request['messages'])
        for forbidden in (route.native.QUESTION, 'Q4', '180', '标准参考', 'REVENUE'):
            self.assertNotIn(forbidden, texts)

    def test_answer_has_only_verbatim_text_and_question(self):
        text = '测试转写：单位看不清；数据未知。'
        request = route.answer_payload(text)
        self.assertEqual(len(request['messages']), 2)
        self.assertEqual(json.loads(request['messages'][1]['content']),
                         {'图表转写文字': text, '问题': route.native.QUESTION})
        self.assertTrue(all('images' not in item for item in request['messages']))
        self.assertEqual(request['options'], route.native.payload(b'')['options'])
        with self.assertRaises(ValueError):
            route.answer_payload(' ')

    def test_success_has_two_fresh_requests_and_rejects_rerun(self):
        transcription = '离线占位转写，不作为正式模型结果'
        FakeSession.responses = [reply(transcription), reply('离线占位回答')]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            route.run_pipeline(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(record['model_requests'], 2)
            self.assertEqual(len(FakeSession.requests), 2)
            self.assertEqual(json.loads(FakeSession.requests[1]['messages'][1]['content'])['图表转写文字'], transcription)
            self.assertNotIn('images', FakeSession.requests[1]['messages'][1])
            self.assertEqual((Path(directory)/'图片转写原文.txt').read_text(), transcription+'\n')
            self.assertEqual(record['comparison']['extract_then_answer']['prompt_tokens_reported'], 20)
            with self.assertRaises(RuntimeError):
                route.run_pipeline(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(len(FakeSession.requests), 2)

    def test_failed_extraction_stops_before_answer(self):
        FakeSession.responses = [(500, '离线故障占位')]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            with self.assertRaises(ValueError):
                route.run_pipeline(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(record['model_requests'], 1)
            self.assertEqual(len(FakeSession.requests), 1)
            self.assertEqual(record['stages'][0]['raw_http_body'], '离线故障占位')
            self.assertFalse((Path(directory)/'text_answer_request.json').exists())

    def test_truncated_answer_preserves_extraction_and_raw_reply(self):
        FakeSession.responses = [reply('保留原始转写'), reply('未完整回答', 'length')]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            with self.assertRaises(ValueError):
                route.run_pipeline(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(record['model_requests'], 2)
            self.assertEqual(record['extracted_text'], '保留原始转写')
            self.assertIn('length', record['stages'][1]['raw_http_body'])
            self.assertNotIn('final_answer', record)
            self.assertFalse((Path(directory)/'纯文本模型回答.txt').exists())


if __name__ == '__main__':
    unittest.main()

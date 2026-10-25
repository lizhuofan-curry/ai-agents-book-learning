"""请求隔离、固定输入及失败留档检查；全部使用本地假响应，不连接Ollama。"""
import base64
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_vision', ROOT/'我的代码'/'02_本机模型直接看图.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class FakeSession:
    body = ''
    posts = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        type(self).posts += 1
        assert url == native.URL and self.trust_env is False
        assert kwargs['allow_redirects'] is False and kwargs['timeout'] == (10, 600)
        return type('Response', (), {'status_code': 200, 'text': self.body})()


class Checks(unittest.TestCase):
    def test_request_contains_only_original_image_question_and_generic_prompt(self):
        picture = native.load_input(ROOT)
        request = native.payload(picture)
        self.assertEqual(request['messages'], [
            {'role': 'system', 'content': native.SYSTEM},
            {'role': 'user', 'content': native.QUESTION,
             'images': [base64.b64encode(picture).decode('ascii')]}])
        self.assertEqual(base64.b64decode(request['messages'][1]['images'][0], validate=True), picture)
        self.assertNotIn('tools', request)
        # 人工可核对的文本部分没有附上答案、参考JSON或生成器。
        text = native.SYSTEM + native.QUESTION
        for forbidden in ('Q4', '180', '120', '150', '95', '标准参考', 'REVENUE'):
            self.assertNotIn(forbidden, text)

    def test_changed_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = ROOT/'运行结果'/native.SOURCE_NAME
            dest = Path(temp)/'运行结果'/native.SOURCE_NAME
            dest.mkdir(parents=True)
            shutil.copyfile(source/'record.json', dest/'record.json')
            (dest/'sample_chart.png').write_bytes(b'changed image')
            with self.assertRaisesRegex(ValueError, '固定图片变化'):
                native.load_input(temp)

    def test_one_request_and_no_second_request(self):
        FakeSession.body = json.dumps({'model': native.MODEL, 'done': True, 'done_reason': 'stop',
                                      'message': {'role': 'assistant', 'content': '离线占位回答'}})
        FakeSession.posts = 0
        record = {'model_requests': 0}
        native.ask({}, record, lambda: None, FakeSession)
        self.assertEqual(record['final_answer'], '离线占位回答')
        with self.assertRaises(RuntimeError):
            native.ask({}, record, lambda: None, FakeSession)
        self.assertEqual(FakeSession.posts, 1)

    def test_truncated_reply_is_preserved_and_not_accepted(self):
        FakeSession.body = json.dumps({'model': native.MODEL, 'done': True, 'done_reason': 'length',
                                      'message': {'role': 'assistant', 'content': '未完整回答'}})
        FakeSession.posts = 0
        record, snapshots = {'model_requests': 0}, []
        with self.assertRaises(ValueError):
            native.ask({}, record, lambda: snapshots.append(dict(record)), FakeSession)
        self.assertEqual(FakeSession.posts, 1)
        self.assertEqual(record['raw_http_body'], FakeSession.body)
        self.assertTrue(any('raw_http_body' in item for item in snapshots))
        self.assertNotIn('final_answer', record)


if __name__ == '__main__':
    unittest.main()

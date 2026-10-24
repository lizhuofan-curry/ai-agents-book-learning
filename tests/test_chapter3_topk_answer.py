"""固定上下文两组对照的离线验证，不调用模型。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step = load(ROOT/'第三章/3.8Agentic_RAG/我的代码/13_固定查询Top5与Top10回答对照.py', 'topk_step')
fake = load(Path(__file__).with_name('test_chapter3_live_rag.py'), 'topk_fakes')


class TopKChecks(unittest.TestCase):
    def test_fixed_source_and_nested_context(self):
        live, h, source, query, hits = step.prepare()
        self.assertEqual(len(hits), 10)
        self.assertEqual(h.leading(hits[7]['text']), step.TARGET)
        a = step.messages_for(source['question'], hits[:5], live.SYSTEM)
        b = step.messages_for(source['question'], hits, live.SYSTEM)
        self.assertEqual(a[:2], b[:2])
        self.assertEqual(query, source['arm']['searches'][2]['query'])
        self.assertNotIn(step.TARGET, a[0]['content'])

    def run_fake(self, action, hits, reason='stop'):
        live = step.load_live()
        session = fake.FakeSession([action], reason)
        # fake session响应名为fake，临时改变独立加载测试模块常量。
        previous = step.MODEL
        step.MODEL = 'fake'
        arm = {}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                step.run_answer(live, session, '合成问题', '合成查询', hits, arm, lambda: None)
        finally:
            step.MODEL = previous
        return arm, session

    def test_one_answer_call_and_unseen_ref_rejected(self):
        hits = [{'chunk_id': 'fake_1', 'text': '合成证据'}]
        arm, session = self.run_fake(fake.answer(), hits)
        self.assertEqual(arm['status'], 'answered_review_pending')
        self.assertEqual(len(session.requests), 1)
        arm, _ = self.run_fake(fake.answer('fake_2'), hits)
        self.assertEqual(arm['status'], 'failed')

    def test_search_and_truncation_rejected_without_retry(self):
        hits = [{'chunk_id': 'fake_1', 'text': '合成证据'}]
        for action, reason in [(fake.search('不能追加'), 'stop'), (fake.answer(), 'length')]:
            arm, session = self.run_fake(action, hits, reason)
            self.assertEqual(arm['status'], 'failed')
            self.assertEqual(len(session.requests), 1)


if __name__ == '__main__':
    unittest.main()

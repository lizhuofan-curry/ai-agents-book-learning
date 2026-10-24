"""证据缺口提示适配的离线检查；不执行正式问题查询或模型调用。"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'第三章/3.8Agentic_RAG/我的代码/07_证据缺口提示对照.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gap = load(PATH, 'gap_test')
fake = load(Path(__file__).with_name('test_chapter3_live_rag.py'), 'fake_live_helpers')


class GapPromptChecks(unittest.TestCase):
    def test_frozen_baseline_and_configuration(self):
        live, retrieval, _, _, baseline, original = gap.prepare()
        old = baseline['arms']['agentic']['calls'][0]['request']
        new = live.payload(gap.MODEL, [])
        self.assertEqual({k:v for k,v in old.items() if k!='messages'},
                         {k:v for k,v in new.items() if k!='messages'})
        self.assertEqual(live.SYSTEM, original+'\n'+gap.GAP_PROMPT)
        self.assertEqual(retrieval.sha(gap.BASELINE), gap.BASELINE_SHA256)
        fresh = gap.load_live()
        self.assertEqual(fresh.SYSTEM, original)  # 未改磁盘03或污染新加载模块。

    def test_prompt_has_no_gold_or_prelabelled_queries(self):
        live, _, _, case, _, _ = gap.prepare()
        for target in case['gold_articles']+case['subqueries']:
            self.assertNotIn(target, live.SYSTEM)
        for field in ('gold_articles', 'subqueries'):
            self.assertNotIn(field, live.SYSTEM)

    def test_runtime_prompt_and_autonomous_stop(self):
        live, _, _, _, _, _ = gap.prepare()
        arm = {}
        session = fake.FakeSession([fake.search('合成缺口一'), fake.search('合成缺口二'), fake.answer()])
        before = hashlib.sha256(gap.BASELINE.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            live.run_arm('agentic', '合成问题', fake.FakeRetriever(), session, 'fake', arm, lambda: None)
        self.assertEqual(arm['status'], 'answered_review_pending')
        self.assertEqual(len(arm['searches']), 2)  # 未被强制查满3次。
        self.assertIn(gap.GAP_PROMPT, session.requests[0]['messages'][0]['content'])
        self.assertEqual(before, hashlib.sha256(gap.BASELINE.read_bytes()).hexdigest())

    def test_budget_still_not_forced_answer(self):
        live, _, _, _, _, _ = gap.prepare()
        session = fake.FakeSession([fake.search(s) for s in ('甲','乙','丙','丁')])
        arm = {}
        with contextlib.redirect_stdout(io.StringIO()):
            live.run_arm('agentic', '合成问题', fake.FakeRetriever(), session, 'fake', arm, lambda: None)
        self.assertEqual(arm['status'], 'budget_exhausted')
        self.assertEqual(len(session.requests), 4)
        self.assertNotIn('answer', arm)


if __name__ == '__main__':
    unittest.main()

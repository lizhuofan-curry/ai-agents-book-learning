"""记忆任务适配的合成测试，不运行官方问题或真实模型。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step=load(ROOT/'第三章/3.9用户记忆迭代检索/我的代码/04_单次与模型自主记忆检索.py','live_memory_test')
fake=load(Path(__file__).with_name('test_chapter3_live_rag.py'),'memory_fake_responses')


class LiveMemoryChecks(unittest.TestCase):
    def test_sources_and_task_prompt(self):
        live,retrieval,_,record,chunks,_=step.prepare()
        self.assertEqual(len(chunks),6)
        self.assertEqual(live.TOP_K,3)
        for marker in retrieval.MARKERS.values():self.assertNotIn(marker,live.SYSTEM)
        self.assertNotIn('Honda',live.SYSTEM)
        self.assertNotIn('Tesla',live.SYSTEM)
        self.assertNotIn('条文',live.SYSTEM)

    def test_adapter_only_returns_selected_chunk(self):
        class Backend:
            def search(self,query,k):return [('x',1)]
        chunks={'x':{'messages':[{'role':'user','content':'raw'}], 'conversation_id':'s',
                     'session_timestamp':'2020-01-01','source_message_numbers':[1]}}
        hits=step.MemorySearch(Backend(),chunks).search('q',3)
        self.assertEqual(hits[0]['text'],'Customer: raw')
        self.assertEqual(hits[0]['session_timestamp'],'2020-01-01')

    def test_two_independent_histories_and_live_search(self):
        live,*_=step.prepare()
        arms=[]
        for mode,actions in [('ordinary',[fake.answer()]),('agentic',[fake.search('new query'),fake.answer()])]:
            arm={};session=fake.FakeSession(actions)
            with contextlib.redirect_stdout(io.StringIO()):
                live.run_arm(mode,'synthetic memory question',fake.FakeRetriever(),session,'fake',arm,lambda:None)
            self.assertEqual(arm['status'],'answered_review_pending')
            arms.append(arm)
        self.assertEqual(len(arms[0]['calls']),1)
        self.assertEqual(len(arms[1]['calls']),2)
        self.assertEqual(len(arms[1]['calls'][0]['request']['messages']),2)
        self.assertIsNot(arms[0]['messages'],arms[1]['messages'])

    def test_budget_still_enforced(self):
        live,*_=step.prepare();arm={}
        session=fake.FakeSession([fake.search(q) for q in ('a','b','c','d')])
        with contextlib.redirect_stdout(io.StringIO()):
            live.run_arm('agentic','synthetic',fake.FakeRetriever(),session,'fake',arm,lambda:None)
        self.assertEqual(arm['status'],'budget_exhausted')
        self.assertEqual(len(arm['searches']),3)
        self.assertNotIn('answer',arm)


if __name__=='__main__':unittest.main()

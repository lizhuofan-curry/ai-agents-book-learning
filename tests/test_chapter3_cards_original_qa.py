"""固定卡片/原文对照的合成检查，不调用本机模型。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


m=load(ROOT/'第三章/3.11上下文与双层记忆/我的代码/07_卡片与原始会话问答对照.py','cards_qa_test')
fake=load(Path(__file__).with_name('test_chapter3_live_rag.py'),'cards_qa_fake')


class CardsQAChecks(unittest.TestCase):
    def test_same_cards_and_metadata_only_originals_added(self):
        _,_,session,cards,originals=m.prepare()
        a=m.materials('cards_only',session,cards,originals)
        b=m.materials('cards_and_original',session,cards,originals)
        self.assertEqual({k:v for k,v in b.items() if k!='original_messages'},a)
        self.assertEqual(len(b['original_messages']),94)
        self.assertNotIn('original_messages',a)
        self.assertNotIn('candidate',a)
        self.assertEqual(cards[0]['source_id'],'C1')

    def run_fake(self,ref,mode):
        _,control,_,_,_=m.prepare();arm={}
        provided={'cards':[{'source_id':'C1','fact':'synthetic'}]}
        if mode=='cards_and_original':provided['original_messages']=[{'source_id':'M1','content':'synthetic'}]
        session=fake.FakeSession([fake.answer(ref)]);old=m.MODEL;m.MODEL='fake'
        try:
            with contextlib.redirect_stdout(io.StringIO()):m.run_arm(control,session,provided,arm,lambda:None)
        finally:m.MODEL=old
        return arm,session

    def test_original_refs_only_allowed_in_second_arm(self):
        a,_=self.run_fake('M1','cards_only');self.assertEqual(a['status'],'failed')
        b,s=self.run_fake('M1','cards_and_original');self.assertEqual(b['status'],'answered_review_pending');self.assertEqual(len(s.requests),1)

    def test_cards_refs_allowed_in_both_arms(self):
        for mode in ('cards_only','cards_and_original'):
            arm,s=self.run_fake('C1',mode);self.assertEqual(arm['status'],'answered_review_pending')
            self.assertEqual(len(s.requests),1)
            self.assertNotIn('Honda',s.requests[0]['messages'][0]['content'])
            self.assertNotIn('Tesla',s.requests[0]['messages'][0]['content'])


if __name__=='__main__':unittest.main()

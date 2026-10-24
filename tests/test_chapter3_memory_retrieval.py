"""规则记忆检索的合成验证，不执行正式车辆问题。"""
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'第三章/3.9用户记忆迭代检索/我的代码/02_单次与规则多步记忆检索.py'
spec=importlib.util.spec_from_file_location('memory_retrieval',PATH)
step=importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


class MemoryRetrievalChecks(unittest.TestCase):
    def test_sources_and_six_chunks(self):
        _,_,chunks,backend,_,_=step.prepare()
        self.assertEqual(len(chunks),6)
        self.assertEqual(len(backend.doc_ids),6)

    def test_entity_rule_and_query_generation(self):
        _,_,_,_,extract,_=step.prepare()
        self.assertIn('Honda Accord',extract(['2019 Honda Accord service']))
        class Fake:
            def search(self,query,top_k):return [('x',1)]
        chunks={'x':{'messages':[{'role':'user','content':'2019 Honda Accord service'}],
                     'conversation_id':'s','session_timestamp':'t','source_message_numbers':[1]}}
        result=step.compare(Fake(),chunks,'synthetic question',extract)
        self.assertEqual(result['rule_multistep']['steps'][1]['query'],'Honda Accord service appointment scheduled')
        self.assertEqual(len(result['rule_multistep']['union_hits']),1)

    def test_marker_presence_not_answer_correctness(self):
        hits=[{'text':'FS-447291 Model 3'}]
        self.assertEqual(step.coverage(hits)['marker_recall'],1)
        self.assertEqual(step.coverage([])['marker_recall'],0)
        self.assertNotIn('answer_correct',step.coverage(hits))

    def test_index_text_uses_only_messages(self):
        chunk={'messages':[{'role':'user','content':'raw words'}],
               'session_metadata':{'test_title':'do not index'},'chunk_id':'do not index'}
        self.assertEqual(step.index_text(chunk),'Customer: raw words')


if __name__=='__main__':unittest.main()

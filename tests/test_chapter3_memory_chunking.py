"""固定轮次分块的合成检查，不执行正式车辆用例分块。"""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/'第三章/3.9用户记忆迭代检索/我的代码/01_阅读会话与固定轮次分块.py'
spec = importlib.util.spec_from_file_location('memory_chunks',PATH)
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


class ChunkChecks(unittest.TestCase):
    def test_sources(self):
        data,manifest = step.load_case()
        self.assertEqual(len(data['conversation_histories']),2)
        self.assertEqual(sum(len(c['messages']) for c in data['conversation_histories']),194)

    def test_overlap_no_missing_and_timestamps(self):
        msgs=[{'role':role,'content':f'{i}-{role}'} for i in range(25) for role in ('user','assistant')]
        conv={'conversation_id':'synthetic','timestamp':'2020-01-01','messages':msgs}
        chunks=step.split_conversation(conv,'test')
        self.assertEqual([(c['start_round'],c['end_round']) for c in chunks],[(1,20),(19,25)])
        self.assertEqual(set(chunks[0]['source_message_numbers']) & set(chunks[1]['source_message_numbers']),{37,38,39,40})
        self.assertTrue(all(c['session_timestamp']=='2020-01-01' for c in chunks))

    def test_partial_round_empty_and_invalid_window(self):
        conv={'conversation_id':'x','messages':[{'role':'user','content':'unfinished'}]}
        chunks=step.split_conversation(conv,'test')
        self.assertEqual(chunks[0]['source_message_numbers'],[1])
        self.assertEqual(step.split_conversation({'conversation_id':'empty','messages':[]},'test'),[])
        with self.assertRaises(ValueError):step.split_conversation(conv,'test',2,2)

    def test_no_cross_session_or_expected_answer_metadata(self):
        conv={'conversation_id':'a','messages':[{'role':'user','content':'a'}],
              'expected_answer':'must_not_leak','title':'must_not_leak'}
        chunk=step.split_conversation(conv,'test')[0]
        self.assertNotIn('must_not_leak',str(chunk))
        other=step.split_conversation({'conversation_id':'b','messages':[{'role':'user','content':'b'}]},'test')[0]
        self.assertNotEqual(chunk['chunk_id'],other['chunk_id'])


if __name__=='__main__':unittest.main()

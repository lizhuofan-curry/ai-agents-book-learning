"""Hybrid假HTTP/查询向量检查；不做真实模型或嵌入前向。"""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest

import 本机记忆适配 as base
import 批量JSONCards适配 as journal
import 跨层Hybrid适配 as batch


def run(official,selected,plan,provenance,embedding,source,cache,network):
    import numpy as np
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(cache),local_files_only=True,trust_remote_code=False)
    seen=[];query_events=[]
    source_digest=journal.digest({cid:{'cards':x['cards'],'chunks':x['chunks']} for cid,x in source.items()})
    choices={case.user_question:False for case in selected}
    class FakeCached:
        def __init__(self,official,case,row,directory,src):
            self.row,self.directory,self.src,self.cid=row,directory,src,case.test_id
            self.calls=0;self.last_usage,self.last_latency_ms=official.Usage(),0.0
        def embed(self,texts):
            self.calls+=1
            if self.calls==1:
                assert list(texts)==[c['text'] for c in self.src['chunks']]
                np.save(self.directory/'documents.npy',self.src['vectors'],allow_pickle=False)
                self.row['document_vectors_reused']=True
                return self.src['vectors'].tolist()
            assert self.calls==2 and texts==['offline hybrid fixture query']
            query_events.append(self.cid)
            values=np.random.default_rng(314).normal(size=(1,512)).astype(np.float32)
            values/=np.linalg.norm(values,axis=1,keepdims=True)
            np.save(self.directory/'query.npy',values,allow_pickle=False)
            self.row['embedding_forward_batches']=1  # 模拟协议计数；本测试不执行真实前向。
            self.row['batches']=[{'kind':'query','status':'completed','fixture_only':True}]
            return values.tolist()
    def fake_post(payload):
        seen.append(deepcopy(payload))
        first=bool(payload.get('tools'))
        question=next(m['content'] for m in payload['messages'] if m['role']=='user')
        message={'role':'assistant','content':'Offline Hybrid fixture answer, not a measured model output.'}
        if first and choices[question]:
            message={'role':'assistant','content':'','tool_calls':[{'function':{'name':'search_memory',
                'arguments':{'query':'offline hybrid fixture query'}}}]}
        n=len(tokenizer.apply_chat_template(payload['messages'],tools=payload.get('tools'),tokenize=True,
            add_generation_prompt=True,enable_thinking=False))
        return 200,json.dumps({'model':base.MODEL,'done':True,'done_reason':'stop','message':message,
            'prompt_eval_count':n-(6 if first else 4),'eval_count':1})
    class Tests(unittest.TestCase):
        def setUp(self):
            seen.clear();query_events.clear()
            for q in choices: choices[q]=False
            self.temp=tempfile.TemporaryDirectory(prefix='hybrid_cross_layer_offline_')
            self.directory=Path(self.temp.name);self.path=self.directory/'record.json'
            self.record=batch.new_record(plan,provenance)
        def tearDown(self): self.temp.cleanup()
        def save(self): journal.persist(self.path,self.record)
        def execute(self,post=fake_post,tok=tokenizer):
            with redirect_stdout(io.StringIO()):
                batch.run_cases(official,selected,self.record,plan,tok,self.directory,self.save,lambda:None,
                    embedding,source,post,FakeCached)
        def test_01_direct_paths_preserve_variable_core_and_no_new_queries(self):
            self.execute()
            self.assertEqual((len(seen),len(query_events)),(3,0))
            self.assertEqual(self.record['summary']['paths'],['core_cards_only']*3)
            self.assertEqual([r['core_card_count'] for r in self.record['cases']],[6,7,4])
            self.assertTrue(all(all(r['checks'].values()) for r in self.record['cases']))
            self.assertEqual(source_digest,journal.digest({cid:{'cards':x['cards'],'chunks':x['chunks']} for cid,x in source.items()}))
        def test_02_optional_search_paths_only_new_query_and_actual_top5(self):
            for q in choices: choices[q]=True
            self.execute()
            self.assertEqual((len(seen),len(query_events)),(6,3))
            self.assertEqual(self.record['summary']['paths'],['core_cards_plus_search']*3)
            self.assertEqual(self.record['summary']['new_embedding_forward_batches'],3)
            self.assertTrue(all(all(r['checks'].values()) for r in self.record['cases']))
        def test_03_mixed_choices_are_not_forced_to_search(self):
            choices[selected[1].user_question]=True
            self.execute()
            self.assertEqual((len(seen),len(query_events)),(4,1))
            self.assertEqual(self.record['summary']['paths'],['core_cards_only','core_cards_plus_search','core_cards_only'])
        def test_04_resume_skips_completed_and_interrupted_cases(self):
            def interrupt(payload):
                seen.append(payload)
                raise KeyboardInterrupt('offline interruption')
            with self.assertRaises(KeyboardInterrupt): self.execute(interrupt)
            self.record=journal.load_checkpoint(self.path)
            batch.check_resume(self.record,plan,provenance,self.directory)
            seen.clear();self.execute()
            self.assertEqual(len(seen),2)
            self.assertEqual(self.record['cases'][0]['status'],'interrupted_not_retried')
            seen.clear();self.execute();self.assertEqual(seen,[])
        def test_05_initial_or_tool_history_overlength_stops_before_dispatch(self):
            class InitialLong:
                def apply_chat_template(self,messages,**kwargs): return [0]*16384
            self.execute(tok=InitialLong())
            self.assertEqual((len(seen),len(query_events)),(0,0))
            with tempfile.TemporaryDirectory() as folder:
                self.directory=Path(folder);self.path=self.directory/'record.json'
                self.record=batch.new_record(plan,provenance)
                for q in choices: choices[q]=True
                class HistoryLong:
                    def apply_chat_template(self,messages,**kwargs):
                        return [0]*16384 if len(messages)>2 else tokenizer.apply_chat_template(messages,**kwargs)
                self.execute(tok=HistoryLong())
                self.assertEqual((len(seen),len(query_events)),(3,3))
                self.assertEqual(self.record['summary']['answers_saved'],0)
        def test_06_invalid_response_and_extra_search_no_retry(self):
            for kind in ['bad_query','extra_tool','truncated','missing_usage','http','bad_json']:
                with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                    self.directory=Path(folder);self.path=self.directory/'record.json'
                    self.record=batch.new_record(plan,provenance);seen.clear();query_events.clear()
                    for q in choices: choices[q]=True
                    def faulty(payload):
                        status,raw=fake_post(payload)
                        if kind=='http': return 503,'offline unavailable'
                        if kind=='bad_json': return 200,'{'
                        r=json.loads(raw)
                        if kind=='bad_query': r['message']['tool_calls'][0]['function']['arguments']={'query':''}
                        if kind=='extra_tool': r['message']['tool_calls']*=2
                        if kind=='truncated': r['done_reason']='length'
                        if kind=='missing_usage': r.pop('prompt_eval_count')
                        return status,json.dumps(r)
                    self.execute(faulty)
                    self.assertEqual((len(seen),len(query_events)),(3,0))
                    self.assertTrue(all('raw_http_body' in r['model_calls'][0] for r in self.record['cases']))
        def test_07_protocol_and_artifact_changes_reject_resume(self):
            self.execute()
            changed=deepcopy(provenance);changed['plan_sha256']='changed'
            with self.assertRaises(ValueError): batch.check_resume(self.record,plan,changed,self.directory)
            (self.directory/self.record['cases'][0]['case_id']/'core_cards.json').write_text('changed')
            with self.assertRaises(ValueError): batch.check_resume(self.record,plan,provenance,self.directory)
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    directory=base.ROOT/'本地运行结果'/('cross_hybrid_offline_'+datetime.now(journal.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(exist_ok=False)
    record={'actor':'Codex fake HTTP/query vectors; not actual model results or task scores','provenance':provenance,
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_passed':result.wasSuccessful(),
        'generation_requests':0,'embedding_forward_batches':0,'judge_requests':0,'network_audit':network,'test_output':output.getvalue()}
    base.dump(directory/'record.json',record)
    print(output.getvalue(),end='')
    print('CROSS_HYBRID_OFFLINE_SAVED：仅假HTTP/查询向量检查；0次真实生成/前向/网络。记录：'+str(directory/'record.json'))
    return 0 if result.wasSuccessful() else 1

"""三案例RAG假HTTP、假向量与故障检查；不做真实前向或模型请求。"""
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
import 跨层RAG适配 as batch


def run(official,selected,plan,provenance,embedding,cache,network):
    import numpy as np
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(cache),local_files_only=True,trust_remote_code=False)
    seen=[]
    embed_calls=[]
    class FakeBGE:
        def __init__(self,official,case,row,directory):
            self.row,self.directory=row,directory
            self.calls=0
            self.expected=official.conversation_chunks(case,6,2)
            self.last_usage,self.last_latency_ms=official.Usage(),0.0
        def embed(self,texts):
            kind='documents' if self.calls==0 else 'query'
            if self.calls==0:
                assert list(texts)==[c.text for c in self.expected]
            else:
                assert self.calls==1 and texts==['offline fixture query']
            self.calls+=1
            embed_calls.append((kind,len(texts)))
            # 人造向量仅用于验证索引、余弦排序及消息映射，不评估语义质量。
            values=np.random.default_rng(42 if kind=='documents' else 43).normal(size=(len(texts),512)).astype(np.float32)
            values/=np.linalg.norm(values,axis=1,keepdims=True)
            np.save(self.directory/(kind+'.npy'),values,allow_pickle=False)
            return values.tolist()
    def fake_post(payload):
        seen.append(deepcopy(payload))
        first=bool(payload.get('tools'))
        content='' if first else 'Offline RAG fixture answer, not a measured model output.'
        message={'role':'assistant','content':content}
        if first: message['tool_calls']=[{'function':{'name':'search_memory','arguments':{'query':'offline fixture query'}}}]
        local=len(tokenizer.apply_chat_template(payload['messages'],tools=payload.get('tools'),tokenize=True,
            add_generation_prompt=True,enable_thinking=False))
        return 200,json.dumps({'model':base.MODEL,'done':True,'done_reason':'stop','message':message,
            'prompt_eval_count':local-(6 if first else 4),'eval_count':1})
    class Tests(unittest.TestCase):
        def setUp(self):
            seen.clear();embed_calls.clear()
            self.temp=tempfile.TemporaryDirectory(prefix='cross_rag_offline_')
            self.directory=Path(self.temp.name)
            self.path=self.directory/'record.json'
            self.record=batch.new_record(plan,provenance)
        def tearDown(self): self.temp.cleanup()
        def save(self): journal.persist(self.path,self.record)
        def execute(self,post=fake_post,tok=tokenizer):
            with redirect_stdout(io.StringIO()):
                batch.run_cases(official,selected,self.record,plan,tok,self.directory,self.save,lambda:None,
                    embedding,post,FakeBGE)
        def test_01_variable_chunks_actual_search_and_top5_only(self):
            self.execute()
            self.assertEqual(len(seen),6)
            self.assertEqual(embed_calls,[('documents',12),('query',1),('documents',24),('query',1),('documents',31),('query',1)])
            self.assertEqual(self.record['summary']['retrieval_requests'],3)
            self.assertEqual(self.record['summary']['answers_saved'],3)
            for row in self.record['cases']:
                self.assertTrue(all(row['checks'].values()))
                self.assertEqual([c['reported_input_minus_local_template'] for c in row['model_calls']],[-6,-4])
                self.assertTrue(all(not c['length_check']['exact_tool_template_equivalence_claimed'] for c in row['model_calls']))
        def test_02_completed_resume_no_dispatch_or_embedding(self):
            self.execute()
            self.record=journal.load_checkpoint(self.path)
            batch.check_resume(self.record,plan,provenance,self.directory)
            seen.clear();embed_calls.clear()
            self.execute()
            self.assertEqual((seen,embed_calls),([],[]))
        def test_03_interrupt_never_resends_prior_case(self):
            def interrupt(payload):
                seen.append(payload)
                raise KeyboardInterrupt('fixture interrupted request')
            with self.assertRaises(KeyboardInterrupt): self.execute(interrupt)
            self.record=journal.load_checkpoint(self.path)
            batch.check_resume(self.record,plan,provenance,self.directory)
            seen.clear();embed_calls.clear()
            self.execute()
            self.assertEqual(len(seen),4)
            self.assertEqual([x for x in embed_calls if x[0]=='documents'],[('documents',24),('documents',31)])
            self.assertEqual(self.record['cases'][0]['status'],'interrupted_not_retried')
        def test_04_dynamic_tool_history_overlength_not_sent(self):
            class LongHistory:
                def apply_chat_template(self,messages,**kwargs):
                    return [0]*16384 if len(messages)>2 else tokenizer.apply_chat_template(messages,**kwargs)
            self.execute(tok=LongHistory())
            self.assertEqual(len(seen),3)
            self.assertEqual(self.record['summary']['answers_saved'],0)
            self.assertTrue(all('http_dispatch_reserved' not in row['model_calls'][1] for row in self.record['cases']))
        def test_05_invalid_actual_tool_usage_and_truncation_no_retry(self):
            for kind in ['no_tool','bad_query','extra_tool','missing_usage','large_drift','truncated','http','bad_json']:
                with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                    self.directory=Path(folder);self.path=self.directory/'record.json'
                    self.record=batch.new_record(plan,provenance);seen.clear();embed_calls.clear()
                    def faulty(payload):
                        status,raw=fake_post(payload)
                        if kind=='http': return 503,'offline unavailable'
                        if kind=='bad_json': return 200,'{'
                        r=json.loads(raw)
                        if kind=='no_tool': r['message'].pop('tool_calls')
                        if kind=='bad_query': r['message']['tool_calls'][0]['function']['arguments']={'query':''}
                        if kind=='extra_tool': r['message']['tool_calls']*=2
                        if kind=='missing_usage': r.pop('prompt_eval_count')
                        if kind=='large_drift': r['prompt_eval_count']+=300
                        if kind=='truncated': r['done_reason']='length'
                        return status,json.dumps(r)
                    self.execute(faulty)
                    self.assertEqual(len(seen),3)
                    self.assertEqual(self.record['summary']['retrieval_requests'],0)
                    self.assertTrue(all('raw_http_body' in row['model_calls'][0] for row in self.record['cases']))
        def test_06_changed_protocol_or_artifact_reject_resume(self):
            self.execute()
            changed=deepcopy(provenance);changed['plan_sha256']='changed'
            with self.assertRaises(ValueError): batch.check_resume(self.record,plan,changed,self.directory)
            (self.directory/self.record['cases'][0]['case_id']/'documents.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError): batch.check_resume(self.record,plan,provenance,self.directory)
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    directory=base.ROOT/'本地运行结果'/('cross_rag_offline_'+datetime.now(journal.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(exist_ok=False)
    record={'actor':'Codex fake HTTP/vectors; not an experiment or semantic evaluation','provenance':provenance,
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_passed':result.wasSuccessful(),
        'generation_requests':0,'embedding_forward_batches':0,'judge_requests':0,'network_audit':network,
        'test_output':output.getvalue()}
    base.dump(directory/'record.json',record)
    print(output.getvalue(),end='')
    print('CROSS_RAG_OFFLINE_SAVED：仅假HTTP/假向量检查；0次真实生成/前向/网络。记录：'+str(directory/'record.json'))
    return 0 if result.wasSuccessful() else 1

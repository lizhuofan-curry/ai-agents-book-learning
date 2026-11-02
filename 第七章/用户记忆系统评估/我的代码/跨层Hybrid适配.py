"""同三题Hybrid：只复用原core卡片/文档向量，自主选择是否检索。"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import 本机记忆适配 as base
import RAG本机适配 as rag
import Hybrid本机适配 as legacy
import 批量JSONCards适配 as journal
import 跨层RAG适配 as cross

PLAN=base.ROOT/'测试材料/同三案例Hybrid方案.json'
ARTIFACTS=cross.ARTIFACTS|{'core_cards.json'}


def verify_inputs():
    import numpy as np
    plan=journal.strict_json(PLAN.read_text())
    rplan,cache,origin,embedding,_=cross.verify_inputs()
    fixed={'protocol_version':'hybrid-cross-layer-smoke-v1','model':base.MODEL,'num_ctx':16384,
        'num_predict':1024,'temperature':0,'seed':42,'think':False,'maximum_model_requests':6,
        'maximum_retrieval_requests':3,'maximum_new_embedding_forward_batches':3,
        'document_forward_batch_cap':0,'query_forward_batch_cap':3,'rounds_per_chunk':6,'overlap':2,
        'top_k':5,'reranker':'none','reuse_original_core_cards':True,'reuse_frozen_document_vectors':True,
        'automatic_retries':False,'judge_requests':0,'engineering_headroom_tokens':256,
        'local_tool_template_exact_equivalence_claimed':False}
    if any(type(plan.get(k)) is not type(v) or plan[k]!=v for k,v in fixed.items()):
        raise ValueError('Hybrid固定三题、复用方式或预算变化')
    if base.sha(Path(legacy.__file__))!=plan['legacy_hybrid_chat_sha256']:
        raise ValueError('原Hybrid工具映射实现变化')
    parent=(base.ROOT/'本地运行结果').resolve()
    source={}
    for name,prefix in [('cards','cards_cross_layer_'),('rag','rag_cross_layer_')]:
        spec=plan['sources'][name];directory=(parent/spec['run']).resolve()
        if directory.parent!=parent or not directory.name.startswith(prefix):
            raise ValueError('冻结来源目录越界')
        for filename,key in [('record.json','record_sha256'),('核对记录.json','execution_review_sha256'),('来源审阅.json','semantic_review_sha256')]:
            if base.sha(directory/filename)!=spec[key]: raise ValueError('冻结原结果或审阅变化')
        record=journal.load_checkpoint(directory/'record.json')
        review=journal.strict_json((directory/'核对记录.json').read_text())
        if review.get('source_record_sha256')!=spec['record_sha256'] or not all(review['checks'].values()):
            raise ValueError('来源执行核对未通过')
        source[name]={'directory':directory,'record':record}
    if source['rag']['record']['plan']!=rplan or source['rag']['record']['provenance']!=origin:
        raise ValueError('冻结RAG构建实现/模型资产变化')
    if [x['case_id'] for x in plan['cases']]!=[x['case_id'] for x in rplan['cases']]:
        raise ValueError('必须是原同三题，不挑题扩展')
    data={}
    for spec,crow,rrow in zip(plan['cases'],source['cards']['record']['cases'],source['rag']['record']['cases']):
        if spec['case_id']!=crow['case_id'] or spec['case_id']!=rrow['case_id']:
            raise ValueError('三路案例顺序不一致')
        directory=source['rag']['directory']/spec['case_id']
        for filename,expected in rrow['artifacts_sha256'].items():
            if filename not in cross.ARTIFACTS or base.sha(directory/filename)!=expected:
                raise ValueError('原RAG产物变化')
        if base.sha(directory/'documents.npy')!=spec['document_vector_sha256'] or base.sha(directory/'chunks.json')!=spec['chunk_file_sha256']:
            raise ValueError('文档向量或原文顺序变化')
        chunks=journal.strict_json((directory/'chunks.json').read_text())
        vectors=np.load(directory/'documents.npy',allow_pickle=False)
        core=[c for c in crow['cards'] if c['memory_tier']=='core']
        if (len(chunks)!=spec['expected_chunks'] or vectors.shape!=(len(chunks),512)
                or vectors.dtype!=np.float32 or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6)
                or len(core)!=spec['core_card_count'] or not core):
            raise ValueError('冻结卡片/文档向量结构不符')
        if rrow['embedding_files_verified']!=origin['bge_files']:
            raise ValueError('冻结文档向量模型资产不一致')
        build=crow['model_calls'][0]
        data[spec['case_id']]={'cards':deepcopy(crow['cards']),'core':deepcopy(core),'chunks':chunks,'vectors':vectors,
            'inherited_construction_usage':{'card_build_source_run':plan['sources']['cards']['run'],
                'card_build_requests':1,'card_build_prompt_tokens':build['response']['prompt_eval_count'],
                'card_build_output_tokens':build['response']['eval_count'],'document_source_run':plan['sources']['rag']['run'],
                'document_embedding_batches':rrow['document_forward_batch_cap'],
                'document_embedding_input_tokens':rrow['embedding_usage']['documents']['tokenizer_input_tokens'],
                'scope':'Previously incurred construction, not incremental requests or zero full-path cost.'}}
    provenance={'rag_and_source_provenance':origin,'plan_sha256':base.sha(PLAN),
        'implementation_sha256':{n:base.sha(Path(__file__).with_name(n)) for n in
            ['11_同三案例Hybrid对照.py','跨层Hybrid适配.py','跨层Hybrid离线检查.py','Hybrid本机适配.py']}}
    return plan,cache,provenance,embedding,data


def new_record(plan,provenance):
    return {'status':'prepared','plan':deepcopy(plan),'provenance':deepcopy(provenance),'cases':[],
        'network_audit':{'allowed_events':[],'blocked_events':[]},'judge_requests':0,'external_network_requests':0,
        'semantic_review':'pending source-level review; source errors deliberately preserved',
        'full_sixty_or_three_arm_runner_completed':False}


def check_resume(record,plan,provenance,directory):
    if record['plan']!=plan or record['provenance']!=provenance:
        raise ValueError('断点方案、实现或冻结产物变化')
    ids=[r['case_id'] for r in record['cases']]
    if ids!=[r['case_id'] for r in plan['cases']][:len(ids)] or any(a>b for a,b in zip(cross.counts(record),[6,3,3])):
        raise ValueError('断点案例顺序或预算异常')
    for row in record['cases']:
        if ([c['stage'] for c in row['model_calls']] not in [[],['hybrid_decision'],['hybrid_decision','hybrid_answer']]
                or row['retrieval_requests']>1 or row['embedding_forward_batches']>1):
            raise ValueError('Hybrid单题阶段或预算异常')
        for name,value in row.get('artifacts_sha256',{}).items():
            if name not in ARTIFACTS or base.sha(directory/row['case_id']/name)!=value:
                raise ValueError('断点产物变化，不覆盖或重发')


class CachedBGE:
    def __init__(self,official,row,record,save,directory,chunks,embedding,source):
        self.official,self.row,self.record,self.save,self.directory=official,row,record,save,directory
        self.chunks,self.embedding,self.source=chunks,embedding,source
        self.calls=0
        self.last_usage,self.last_latency_ms=official.Usage(),0.0
    def embed(self,texts):
        self.calls+=1
        if self.calls==1:
            if list(texts)!=[c['text'] for c in self.source['chunks']]:
                raise ValueError('冻结文档向量与本题原文顺序不符')
            import numpy as np
            started=time.perf_counter()
            np.save(self.directory/'documents.npy',self.source['vectors'],allow_pickle=False)
            self.row['document_vectors_reused']=True
            self.last_latency_ms=(time.perf_counter()-started)*1000
            self.last_usage=self.official.Usage()
            self.save()
            return self.source['vectors'].tolist()
        if self.calls!=2 or len(texts)!=1 or cross.counts(self.record)[2]>=3:
            raise base.StopPipeline('最多复用文档一次和新查询一次，全批最多3批新前向')
        live=cross.VariableBGE(self.official,self.row,self.record,self.save,self.directory,self.chunks,self.embedding)
        live.calls=1  # 文档来自缓存；仅嵌入这次模型真实发出的新查询。
        values=live.embed(texts)
        self.last_usage,self.last_latency_ms=live.last_usage,live.last_latency_ms
        return values


def run_one(official,case,row,record,plan,tokenizer,directory,save,guard,embedding,source,post,embedder=None):
    chunks=official.conversation_chunks(case,6,2)
    if [asdict(c) for c in chunks]!=source['chunks']:
        raise ValueError('原切块与冻结向量顺序不一致')
    core=official.select_core_cards(source['cards'],case.test_id)
    if core!=source['core'] or len(core)!=row['core_card_count']:
        raise ValueError('须使用原core，不修复、不重分层、不重新提取')
    base.dump(directory/'chunks.json',source['chunks'])
    base.dump(directory/'core_cards.json',core)
    row['inherited_construction_usage']=source['inherited_construction_usage']
    if embedder is None: embedder=CachedBGE(official,row,record,save,directory,chunks,embedding,source)
    class Index(official.VectorMemoryIndex):
        def search(self,query,candidate_k=20):
            if row['retrieval_requests']>=1 or cross.counts(record)[1]>=3:
                raise base.StopPipeline('Hybrid每题最多一次实际查询')
            row['retrieval_requests']+=1;save()
            ranked,usage,latency=super().search(query,candidate_k)
            result={'query':query,'candidates':[{'chunk_id':c.chunk_id,'cosine':score} for c,score in ranked],
                'returned_top5_ids':[c.chunk_id for c,_ in ranked[:5]]}
            row['retrievals']=[result];base.dump(directory/'retrieval.json',result);save()
            print('实际Top5：'+' → '.join(result['returned_top5_ids']),flush=True)
            return ranked,usage,latency
    index=Index(chunks,embedder)
    chat=legacy.HybridChat(official,plan,row,save,cross.guarded_post(post,tokenizer,row,record,save,guard))
    result=official.MemoryAgent(chat).rag(case.user_question,index,official.NoReranker(),hybrid_cards=core,
        top_k=5,allow_followup_searches=False,max_search_rounds=1)
    searched=result.tool_calls==1
    row['observed_path']='core_cards_plus_search' if searched else 'core_cards_only'
    row['answer']=result.answer
    row['agent_result']={'steps':result.steps,'tool_calls':result.tool_calls,'trace':result.trace,
        'retrieved_chunks':[asdict(c) for c in result.retrieved_chunks]}
    base.dump(directory/'retrieved_top5.json',row['agent_result']['retrieved_chunks'])
    (directory/'answer.txt').write_text(result.answer,encoding='utf-8')
    first=row['model_calls'][0]
    expected=[{'role':'system','content':official.SYSTEM_RULES+'\nResident core Advanced JSON Cards:\n'+json.dumps(core,ensure_ascii=False)},
        {'role':'user','content':case.user_question}]
    checks={'original_core_cards_without_repair':core==source['core'],
        'independent_initial_context_core_and_question_only':first['official_messages']==expected,
        'model_choice_determined_observed_path':result.tool_calls in [0,1] and result.steps==len(row['model_calls'])==1+result.tool_calls,
        'all_calls_completed_within_budget':len(row['model_calls'])<=2 and all(c['status']=='completed' for c in row['model_calls']),
        'answer_matches_raw_response':row['answer']==row['model_calls'][-1]['response']['message']['content'],
        'cached_documents_no_new_document_batches':row.get('document_vectors_reused') is True and all(b['kind']=='query' for b in row['batches']),
        'new_query_only_when_actual_search':row['retrieval_requests']==row['embedding_forward_batches']==int(searched),
        'tool_results_match_actual_top5':True}
    if searched:
        last=row['model_calls'][-1];tools=[m for m in last['official_messages'] if m['role']=='tool']
        expected_chunks=[{'chunk_id':c.chunk_id,'conversation_id':c.conversation_id,'rounds':[c.start_round,c.end_round],'text':c.text} for c in result.retrieved_chunks]
        checks['tool_results_match_actual_top5']=(len(expected_chunks)==5 and len(tools)==1 and journal.strict_json(tools[0]['content'])==expected_chunks
            and 'tools' not in last['native_request'] and [c.chunk_id for c in result.retrieved_chunks]==row['retrievals'][0]['returned_top5_ids'])
    else:
        checks['tool_results_match_actual_top5']=not result.retrieved_chunks and 'retrievals' not in row
    row['checks']=checks
    if not all(checks.values()): raise ValueError('Hybrid执行一致性检查失败')
    row['status']='answer_saved_semantic_review_pending'
    print('实际路径：'+row['observed_path']+'；原core卡片'+str(len(core))+'张',flush=True)
    print('回答：'+result.answer,flush=True)


def run_cases(official,selected,record,plan,tokenizer,directory,save,guard,embedding,source,post=base.local_post,embedder_factory=None):
    touched={r['case_id'] for r in record['cases']}
    for row in record['cases']:
        if row['status']=='started': row['status']='interrupted_not_retried'
    save()
    for case,spec in zip(selected,plan['cases']):
        if case.test_id in touched: continue
        row={**deepcopy(spec),'document_forward_batch_cap':0,'status':'started','model_calls':[],
            'retrieval_requests':0,'embedding_forward_batches':0,'batches':[],'semantic_review':'pending'}
        record['cases'].append(row);save()
        target=directory/case.test_id;target.mkdir(exist_ok=False)
        try:
            print('开始：'+case.test_id,flush=True)
            fixture=embedder_factory(official,case,row,target,source[case.test_id]) if embedder_factory else None
            run_one(official,case,row,record,plan,tokenizer,target,save,guard,embedding,source[case.test_id],post,fixture)
        except (Exception,base.StopPipeline) as exc:
            row.update(status='failed_or_incomplete',error={'type':type(exc).__name__,'message':str(exc)})
            print('本题失败保留，不重试：'+str(exc),flush=True)
        finally:
            row['artifacts_sha256']={p.name:base.sha(p) for p in target.iterdir() if p.name in ARTIFACTS and p.is_file()};save()
    calls=[c for row in record['cases'] for c in row['model_calls']]
    complete=sum(r['status']=='answer_saved_semantic_review_pending' for r in record['cases'])
    def count_tokens(key):
        values=[c.get('response',{}).get(key) for c in calls];known=[v for v in values if type(v) is int and v>=0]
        return {'reported_sum':sum(known),'unknown_calls':len(values)-len(known)}
    model,retrieval,forward=cross.counts(record)
    record['summary']={'answers_saved':complete,'cases_touched':len(record['cases']),'reserved_model_requests':model,
        'retrieval_requests':retrieval,'new_embedding_forward_batches':forward,
        'paths':[r.get('observed_path','incomplete') for r in record['cases']],
        'prompt_tokens':count_tokens('prompt_eval_count'),'output_tokens':count_tokens('eval_count'),
        'cached_input_tokens':None,'local_billed_money':None,'formal_task_pass_rate':None}
    record['status']='three_case_hybrid_saved_review_pending' if complete==3 else 'three_case_hybrid_incomplete_failures_preserved'
    save()

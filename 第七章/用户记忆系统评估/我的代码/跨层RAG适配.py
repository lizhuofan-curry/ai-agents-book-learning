"""同三案例纯RAG；复用固定切块/索引/原生工具适配，旧代码只读。"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import 本机记忆适配 as base
import RAG本机适配 as rag
import 批量JSONCards适配 as journal

PLAN = base.ROOT / '测试材料/跨层三案例RAG方案.json'


def verify_inputs():
    plan = journal.strict_json(PLAN.read_text())
    old, cache, origin = journal.verify_inputs()
    fixed = {'protocol_version': 'rag-cross-layer-smoke-v1', 'model': base.MODEL, 'num_ctx': 16384,
        'num_predict': {'search_request': 1024, 'rag_answer': 1024}, 'maximum_model_requests': 6,
        'maximum_retrieval_requests': 3, 'maximum_embedding_forward_batches': 12,
        'document_forward_batch_cap': 9, 'query_forward_batch_cap': 3, 'rounds_per_chunk': 6,
        'overlap': 2, 'top_k': 5, 'embedding_batch_size': 8, 'reranker': 'none',
        'normalization': 'casefold', 'temperature': 0, 'seed': 42, 'think': False,
        'automatic_retries': False, 'judge_requests': 0, 'engineering_headroom_tokens': 256,
        'local_tool_template_exact_equivalence_claimed': False, 'observed_prompt_token_drift_cap': 256}
    if any(type(plan.get(k)) is not type(v) or plan[k] != v for k, v in fixed.items()):
        raise ValueError('固定纯RAG三案例或协议预算变化')
    if ([{k: c[k] for k in old['cases'][0]} for c in plan['cases']] != old['cases']
            or [c['expected_chunks'] for c in plan['cases']] != [12, 24, 31]
            or [c['document_forward_batch_cap'] for c in plan['cases']] != [2, 3, 4]):
        raise ValueError('必须为先前Cards同三题和固定切块；不挑题/调参')
    parent = (base.ROOT / '本地运行结果').resolve()
    source = (parent / plan['cards_run']).resolve()
    if source.parent != parent or not source.name.startswith('cards_cross_layer_'):
        raise ValueError('原Cards来源目录越界')
    for filename, key in [('record.json', 'cards_record_sha256'), ('核对记录.json', 'cards_execution_review_sha256'),
            ('来源审阅.json', 'cards_semantic_review_sha256')]:
        if base.sha(source / filename) != plan[key]:
            raise ValueError('原Cards及审阅变化；不覆盖负面基线')
    embedding, hashes, info, model_dir, files = rag.verify_plan()
    code = ['10_同三案例RAG对照.py', '跨层RAG适配.py', '跨层RAG离线检查.py']
    provenance = {'upstream_and_cards_preparation': origin, 'plan_sha256': base.sha(PLAN),
        'implementation_sha256': {n: base.sha(Path(__file__).with_name(n)) for n in code},
        'bge_source_hashes': hashes, 'bge_files': files, 'bge_model': info}
    return plan, cache, provenance, embedding, model_dir


def new_record(plan, provenance):
    return {'status': 'prepared', 'plan': deepcopy(plan), 'provenance': deepcopy(provenance), 'cases': [],
        'network_audit': {'allowed_events': [], 'blocked_events': []}, 'judge_requests': 0,
        'external_network_requests': 0, 'semantic_review': 'pending source-level review',
        'full_sixty_or_three_arm_runner_completed': False}


def counts(record):
    return (sum(len(r['model_calls']) for r in record['cases']),
        sum(r['retrieval_requests'] for r in record['cases']),
        sum(r['embedding_forward_batches'] for r in record['cases']))


def check_resume(record, plan, provenance, directory):
    if record['plan'] != plan or record['provenance'] != provenance:
        raise ValueError('恢复时方案/实现/资料/模型资产变化')
    ids = [c['case_id'] for c in record['cases']]
    if ids != [c['case_id'] for c in plan['cases']][:len(ids)] or any(a > b for a, b in zip(counts(record), [6, 3, 12])):
        raise ValueError('断点顺序或预算异常')
    for row in record['cases']:
        if [c['stage'] for c in row['model_calls']] not in [[], ['search_request'], ['search_request', 'rag_answer']]:
            raise ValueError('RAG阶段乱序或重复')
        if row['retrieval_requests'] > 1 or row['embedding_forward_batches'] > row['document_forward_batch_cap'] + 1:
            raise ValueError('单题检索/嵌入预算异常')
        for name, expected in row.get('artifacts_sha256', {}).items():
            if name not in ARTIFACTS or base.sha(directory / row['case_id'] / name) != expected:
                raise ValueError('原RAG产物变化，拒绝覆盖')


ARTIFACTS = {'chunks.json', 'documents.npy', 'query.npy', 'retrieval.json', 'retrieved_top5.json', 'answer.txt'}


class VariableBGE:
    def __init__(self, official, row, record, save, directory, chunks, embedding):
        self.official, self.row, self.record, self.save, self.directory = official, row, record, save, directory
        self.chunks, self.embedding, self.calls = chunks, embedding, 0
        self.last_usage, self.last_latency_ms = official.Usage(), 0.0
        self.tokenizer = self.model = None

    def embed(self, texts):
        kind = 'documents' if self.calls == 0 else 'query'
        expected = [c.text for c in self.chunks] if kind == 'documents' else None
        if self.calls >= 2 or (expected is not None and list(texts) != expected) or (kind == 'query' and len(texts) != 1):
            raise base.StopPipeline('仅一次原文文档嵌入和一次模型实际查询；不补检索')
        self.calls += 1
        started = time.perf_counter()
        if self.model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            _, _, info, model_dir, files = rag.verify_plan()
            self.row['embedding_model'], self.row['embedding_files_verified'] = info, files
            torch.manual_seed(42)
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                trust_remote_code=False, use_safetensors=True).to('cpu').eval()
            if self.model.config.hidden_size != 512 or self.model.config.max_position_embeddings != 512:
                raise ValueError('固定BGE维度/长度变化')
        if counts(self.record)[2] >= 12:
            raise base.StopPipeline('全批嵌入前向上限12')
        encoder = rag.load(base.PROJECT / self.embedding['bge_encoder'], 'cross_layer_fixed_bge_encoder')
        instruction = self.row['embedding_model']['query_instruction']
        inputs = [t.casefold() for t in texts] if kind == 'documents' else [(instruction + texts[0]).casefold()]
        ids = [c.chunk_id for c in self.chunks] if kind == 'documents' else ['actual_model_search_query']
        self.row.setdefault('embedding_model_inputs', {})[kind] = inputs
        self.save()
        vectors = encoder.encode(inputs, ids, kind, self.tokenizer, self.model, self.row, self.save,
            8, self.row['document_forward_batch_cap'] + 1)
        import numpy as np
        np.save(self.directory / (kind + '.npy'), vectors, allow_pickle=False)
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        tokens = sum(self.row['token_lengths'][kind])
        self.last_usage = self.official.Usage(input_tokens=tokens, unpriced_tokens=tokens, unpriced_requests=1)
        self.row.setdefault('embedding_usage', {})[kind] = {'tokenizer_input_tokens': tokens,
            'latency_ms_including_first_load_if_any': self.last_latency_ms, 'local_billed_money': None}
        self.save()
        return vectors.tolist()


def guarded_post(post, tokenizer, row, record, save, guard):
    def send(payload):
        entry = row['model_calls'][-1]
        try:
            guard()
            if counts(record)[0] > 6:
                raise ValueError('全批模型请求预算6')
            ids = tokenizer.apply_chat_template(payload['messages'], tools=payload.get('tools'),
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            nout = payload['options']['num_predict']
            length = {'local_template_tokens': len(ids), 'output_reserve': nout,
                'engineering_headroom_tokens': 256, 'required_total': len(ids) + nout + 256,
                'num_ctx': 16384, 'exact_tool_template_equivalence_claimed': False}
            entry['length_check'] = length
            save()
            if length['required_total'] > 16384:
                raise ValueError('动态工具上下文超过预留上限，不截断、不发送')
            entry['http_dispatch_reserved'] = True
            save()
            status, raw = post(deepcopy(payload))
            entry.update(http_status=status, raw_http_body=raw)
            save()
            if status == 200:
                response = journal.strict_json(raw)
                entry['response'] = response
                save()
                pin, pout = response.get('prompt_eval_count'), response.get('eval_count')
                if any(type(v) is not int or v < 0 for v in [pin, pout]) or pout > nout:
                    raise ValueError('实际用量未知/类型异常/超过输出上限')
                entry['reported_input_minus_local_template'] = pin - len(ids)
                save()
                if abs(pin - len(ids)) > 256 or pin + nout > 16384:
                    raise ValueError('实际报告计数超过保留余量；停止本题后续阶段，不声称工具模板精确等价')
            return status, raw
        except Exception as exc:
            entry.update(status='failed_or_blocked', error={'type': type(exc).__name__, 'message': str(exc)})
            save()
            raise base.StopPipeline(str(exc)) from exc
    return send


def run_one(official, case, row, record, plan, tokenizer, directory, save, guard, post, embedding, embedder=None):
    chunks = official.conversation_chunks(case, 6, 2)
    if len(chunks) != row['expected_chunks'] or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError('实际切块与冻结预检不符')
    base.dump(directory / 'chunks.json', [asdict(c) for c in chunks])
    if embedder is None:
        embedder = VariableBGE(official, row, record, save, directory, chunks, embedding)
    class TrackedIndex(official.VectorMemoryIndex):
        def search(self, query, candidate_k=20):
            if row['retrieval_requests'] >= 1 or counts(record)[1] >= 3:
                raise base.StopPipeline('每题只允许一次检索，全批最多三次')
            row['retrieval_requests'] += 1
            save()
            ranked, usage, latency = super().search(query, candidate_k)
            result = {'query': query, 'candidates': [{'chunk_id': c.chunk_id, 'cosine': score} for c, score in ranked],
                'returned_top5_ids': [c.chunk_id for c, _ in ranked[:5]]}
            row['retrievals'] = [result]
            base.dump(directory / 'retrieval.json', result)
            save()
            print('实际Top5：' + ' → '.join(result['returned_top5_ids']), flush=True)
            return ranked, usage, latency
    index = TrackedIndex(chunks, embedder)
    chat = rag.RAGChat(official, plan, row, save, guarded_post(post, tokenizer, row, record, save, guard))
    result = official.MemoryAgent(chat).rag(case.user_question, index, official.NoReranker(),
        top_k=5, allow_followup_searches=False, max_search_rounds=1)
    row['answer'] = result.answer
    row['agent_result'] = {'steps': result.steps, 'tool_calls': result.tool_calls, 'trace': result.trace,
        'latency_ms_excluding_index_build': result.latency_ms, 'retrieved_chunks': [asdict(c) for c in result.retrieved_chunks]}
    base.dump(directory / 'retrieved_top5.json', row['agent_result']['retrieved_chunks'])
    (directory / 'answer.txt').write_text(result.answer, encoding='utf-8')
    first, second = row['model_calls']
    expected = [{'chunk_id': c.chunk_id, 'conversation_id': c.conversation_id,
        'rounds': [c.start_round, c.end_round], 'text': c.text} for c in result.retrieved_chunks]
    tools = [m for m in second['official_messages'] if m['role'] == 'tool']
    row['checks'] = {'two_completed_model_requests': len(row['model_calls']) == 2 and all(c['status']=='completed' for c in row['model_calls']),
        'question_only_independent_initial_session': first['official_messages'] == [{'role':'system','content':official.SYSTEM_RULES},{'role':'user','content':case.user_question}],
        'one_actual_model_search': result.steps == 2 and result.tool_calls == row['retrieval_requests'] == 1,
        'only_actual_top5_forwarded': len(expected)==5 and len(tools)==1 and journal.strict_json(tools[0]['content'])==expected,
        'native_mapping_no_final_tools': second['native_request']['messages']==rag.native_messages(second['official_messages']) and 'tools' not in second['native_request'],
        'top5_matches_saved_ranking': [c.chunk_id for c in result.retrieved_chunks]==row['retrievals'][0]['returned_top5_ids'],
        'answer_matches_original_response': row['answer']==second['response']['message']['content'],
        'local_and_reported_lengths_within_guard': all(c['length_check']['required_total']<=16384 and abs(c['reported_input_minus_local_template'])<=256 for c in row['model_calls'])}
    if not all(row['checks'].values()):
        raise ValueError('RAG执行一致性检查失败')
    row['status'] = 'answer_saved_semantic_review_pending'
    print('回答：' + result.answer, flush=True)


def run_cases(official, selected, record, plan, tokenizer, directory, save, guard, embedding, post=base.local_post, embedder_factory=None):
    touched = {r['case_id'] for r in record['cases']}
    for row in record['cases']:
        if row['status']=='started':
            row['status']='interrupted_not_retried'
    save()
    for case, spec in zip(selected, plan['cases']):
        if case.test_id in touched:
            continue
        row = {**deepcopy(spec), 'status':'started', 'model_calls':[], 'retrieval_requests':0,
            'embedding_forward_batches':0, 'batches':[], 'semantic_review':'pending'}
        record['cases'].append(row)
        save()
        target = directory / case.test_id
        target.mkdir(exist_ok=False)
        try:
            embedder = embedder_factory(official, case, row, target) if embedder_factory else None
            print('开始：' + case.test_id, flush=True)
            run_one(official, case, row, record, plan, tokenizer, target, save, guard, post, embedding, embedder)
        except (Exception, base.StopPipeline) as exc:
            row.update(status='failed_or_incomplete', error={'type':type(exc).__name__,'message':str(exc)})
            print('本题失败保留，不重试：' + str(exc), flush=True)
        finally:
            row['artifacts_sha256']={p.name:base.sha(p) for p in target.iterdir() if p.name in ARTIFACTS and p.is_file()}
            save()
    complete=sum(r['status']=='answer_saved_semantic_review_pending' for r in record['cases'])
    model, retrieval, forward=counts(record)
    record['summary']={'answers_saved':complete,'cases_touched':len(record['cases']), 'reserved_model_requests':model,
        'retrieval_requests':retrieval,'embedding_forward_batches':forward,'formal_task_pass_rate':None,
        'cached_input_tokens':None,'local_billed_money':None}
    record['status']='three_case_rag_saved_review_pending' if complete==3 else 'three_case_rag_incomplete_failures_preserved'
    save()

"""复用冻结构建产物的单题Hybrid；保留Cards/RAG旧实现与失败。"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import 本机记忆适配 as base
import RAG本机适配 as rag

ROOT = base.ROOT
PLAN = ROOT / '测试材料/Hybrid单案例方案.json'


def verify_inputs():
    import numpy as np
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    rag_plan, deps, info, _, files = rag.verify_plan()
    same = ['upstream_commit', 'case_id', 'case_source', 'case_sha256', 'model', 'endpoint',
        'num_ctx', 'temperature', 'seed', 'think', 'rounds_per_chunk', 'overlap', 'top_k', 'reranker']
    fixed = {'maximum_model_requests': 2, 'maximum_tool_calls': 1, 'maximum_new_embedding_forward_batches': 1,
        'num_predict': 1024, 'reuse_original_core_cards': True, 'reuse_frozen_document_vectors': True,
        'automatic_retries': False, 'judge_requests': 0}
    if any(plan.get(k) != rag_plan[k] for k in same) or any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('Hybrid单题、模型、预算或来源口径变化')
    source = {}
    parent = (ROOT / '本地运行结果').resolve()
    for name, prefix in [('cards', 'cards_local_'), ('rag', 'rag_local_')]:
        directory = (parent / plan[name + '_source_run']).resolve()
        if directory.parent != parent or not directory.name.startswith(prefix):
            raise ValueError('冻结来源越界')
        record = json.loads((directory / 'record.json').read_text())
        review = json.loads((directory / '核对记录.json').read_text())
        if (base.sha(directory / 'record.json') != plan[name + '_record_sha256']
                or base.sha(directory / '核对记录.json') != plan[name + '_review_sha256']
                or review['source_record_sha256'] != plan[name + '_record_sha256']
                or not all(record['checks'].values()) or not all(review['checks'].values())):
            raise ValueError('冻结原记录或核对旁证变化')
        for filename, digest in record['artifacts_sha256'].items():
            if base.sha(directory / filename) != digest:
                raise ValueError('冻结来源产物变化：' + filename)
        source[name] = {'directory': directory, 'record': record, 'review': review}
    cards = json.loads((source['cards']['directory'] / 'cards.json').read_text())['cards']
    if (source['cards']['record']['plan'] != base.load_plan() or source['rag']['record']['plan'] != rag_plan
            or source['rag']['record']['bge_source_hashes'] != deps
            or source['cards']['record']['adapter_sha256'] != base.sha(Path(base.__file__))
            or source['rag']['record']['adapter_sha256'] != base.sha(Path(rag.__file__))):
        raise ValueError('来源构建配置或旧实现已变化')
    chunks = json.loads((source['rag']['directory'] / 'chunks.json').read_text())
    vectors = np.load(source['rag']['directory'] / 'documents.npy', allow_pickle=False)
    if (cards != source['cards']['record']['cards'] or len(chunks) != 11 or vectors.shape != (11, 512)
            or vectors.dtype != np.float32 or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6)):
        raise ValueError('冻结卡片/原文/文档向量不符')
    if source['rag']['record']['embedding_files_verified'] != files:
        raise ValueError('冻结向量与当前嵌入模型文件不一致')
    source.update(cards_data=cards, chunks=chunks, vectors=vectors, rag_plan=rag_plan,
        bge_source_hashes=deps, embedding_model=info)
    return plan, source


class HybridChat:
    def __init__(self, official, plan, record, save, post=base.local_post):
        self.official, self.plan, self.record, self.save, self.post = official, plan, record, save, post

    def complete(self, messages, tools=None, tool_choice=None, json_object=False):
        calls = self.record['model_calls']
        if len(calls) >= 2:
            raise base.StopPipeline('Hybrid最多两次请求，不重试')
        first = not calls
        if tools != [self.official.SEARCH_TOOL] or tool_choice != ('auto' if first else 'none') or json_object:
            raise base.StopPipeline('Hybrid只允许首次自主选择、检索后直接回答')
        request = {'model': base.MODEL, 'messages': rag.native_messages(messages), 'stream': False, 'think': False,
            'options': {'temperature': 0, 'seed': 42, 'num_ctx': self.plan['num_ctx'], 'num_predict': 1024}}
        if first:
            request['tools'] = deepcopy(tools)
        entry = {'stage': 'hybrid_decision' if first else 'hybrid_answer', 'status': 'started',
            'official_messages': deepcopy(messages), 'official_tool_choice': tool_choice, 'native_request': request}
        calls.append(entry)
        self.save()
        print(f'第{len(calls)}次本机模型请求：{entry["stage"]}', flush=True)
        started = time.perf_counter()
        try:
            status, raw = self.post(deepcopy(request))
            entry.update(http_status=status, raw_http_body=raw, client_wall_ms=(time.perf_counter() - started) * 1000)
            self.save()
            if status != 200:
                raise ValueError('本机HTTP状态：' + str(status))
            response = json.loads(raw)
            entry['response'] = response
            self.save()
            msg = response.get('message', {})
            if (response.get('model') != base.MODEL or response.get('done') is not True
                    or response.get('done_reason') != 'stop' or msg.get('role') != 'assistant'):
                raise ValueError('模型标签/角色/结束状态异常')
            pin, pout = response.get('prompt_eval_count'), response.get('eval_count')
            if any(type(v) is not int or v < 0 for v in [pin, pout]):
                raise ValueError('模型用量未知，不补0')
            content, wire = msg.get('content') or '', msg.get('tool_calls') or []
            if not isinstance(content, str) or not isinstance(wire, list):
                raise ValueError('正文或工具调用格式异常')
            parsed, assistant = [], {'role': 'assistant', 'content': content}
            if wire:
                if not first or len(wire) != 1:
                    raise ValueError('只允许首次响应选择一个工具')
                f = wire[0].get('function', {})
                args = f.get('arguments')
                if (f.get('name') != 'search_memory' or not isinstance(args, dict) or set(args) != {'query'}
                        or not isinstance(args['query'], str) or not args['query'].strip() or len(args['query']) > 1000):
                    raise ValueError('工具名或查询参数异常')
                identity = 'local-hybrid-1-0'
                parsed = [self.official.ToolCall(identity, 'search_memory', deepcopy(args))]
                assistant['tool_calls'] = [{'id': identity, 'type': 'function', 'function': {
                    'name': 'search_memory', 'arguments': json.dumps(args, ensure_ascii=False)}}]
                print('模型选择：search_memory ' + json.dumps(args, ensure_ascii=False), flush=True)
            elif not content.strip():
                raise ValueError('无工具也无有效正文')
            entry.update(status='completed', reported_prompt_tokens=pin, reported_output_tokens=pout)
            return self.official.ChatTurn(content, parsed, self.official.Usage(input_tokens=pin,
                output_tokens=pout, unpriced_tokens=pin + pout, unpriced_requests=1), entry['client_wall_ms'], assistant)
        except Exception as exc:
            entry.update(status='failed_or_incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            self.save()


class CachedDocumentBGE:
    def __init__(self, official, source, record, save, directory):
        self.official, self.source, self.record, self.save, self.directory = official, source, record, save, directory
        self.calls, self.last_latency_ms, self.last_usage = 0, 0.0, official.Usage()

    def embed(self, texts):
        self.calls += 1
        if self.calls == 1:
            if texts != [c['text'] for c in self.source['chunks']]:
                raise ValueError('原文片段与冻结文档向量顺序不一致')
            started = time.perf_counter()
            result = self.source['vectors'].tolist()
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            self.last_usage = self.official.Usage()
            self.record['document_vectors_reused'] = True
            self.save()
            return result
        if self.calls == 2 and len(texts) == 1:
            live = rag.LocalBGE(self.official, self.source['rag_plan'], self.record, self.save,
                self.directory, [c['chunk_id'] for c in self.source['chunks']])
            live.calls = 1  # 文档已由缓存提供；这里只做本次实际查询，不能复用旧查询向量。
            result = live.embed(texts)
            self.last_latency_ms, self.last_usage = live.last_latency_ms, live.last_usage
            return result
        raise base.StopPipeline('最多复用一次文档向量并执行一次新查询嵌入')


def run_hybrid_path(official, case, plan, source, record, save, directory, post=base.local_post, embedder=None):
    chunks = official.conversation_chunks(case, 6, 2)
    if [asdict(c) for c in chunks] != source['chunks']:
        raise ValueError('原方法重建片段与缓存来源不一致')
    core = official.select_core_cards(source['cards_data'], case.test_id)
    if len(core) != 11:
        raise ValueError('原始模型卡片应筛出11张core；不修复或重新分层')
    base.dump(directory / 'core_cards.json', core)
    record['core_card_count'] = len(core)
    if embedder is None:
        embedder = CachedDocumentBGE(official, source, record, save, directory)

    class TrackedIndex(official.VectorMemoryIndex):
        def search(self, query, candidate_k=20):
            if record['retrieval_requests'] >= 1:
                raise base.StopPipeline('只允许一次真实检索')
            record['retrieval_requests'] = 1
            save()
            ranked, usage, latency = super().search(query, candidate_k)
            row = {'query': query, 'candidates': [{'chunk_id': c.chunk_id, 'cosine': score} for c, score in ranked],
                'returned_top5_ids': [c.chunk_id for c, _ in ranked[:5]]}
            record['retrieval'] = row
            base.dump(directory / 'retrieval.json', row)
            save()
            print('实际检索Top5：' + ' → '.join(row['returned_top5_ids']), flush=True)
            return ranked, usage, latency

    index = TrackedIndex(chunks, embedder)
    chat = HybridChat(official, plan, record, save, post)
    result = official.MemoryAgent(chat).rag(case.user_question, index, official.NoReranker(),
        hybrid_cards=core, top_k=5, allow_followup_searches=False, max_search_rounds=1)
    record['answer'] = result.answer
    record['agent_result'] = {'steps': result.steps, 'tool_calls': result.tool_calls,
        'trace': result.trace, 'latency_ms_excluding_cache_index_load': result.latency_ms,
        'retrieved_chunks': [asdict(c) for c in result.retrieved_chunks]}
    record['observed_path'] = 'core_cards_only' if result.tool_calls == 0 else 'core_cards_plus_search'
    (directory / 'answer.txt').write_text(result.answer, encoding='utf-8')
    base.dump(directory / 'retrieved_top5.json', record['agent_result']['retrieved_chunks'])
    first = record['model_calls'][0]
    expected_first = [{'role': 'system', 'content': official.SYSTEM_RULES + '\nResident core Advanced JSON Cards:\n' + json.dumps(core, ensure_ascii=False)},
        {'role': 'user', 'content': case.user_question}]
    checks = {
        'original_core_cards_selected_without_repair': json.loads((directory / 'core_cards.json').read_text()) == core,
        'independent_initial_context_is_core_cards_and_question': first['official_messages'] == expected_first,
        'model_choices_determine_observed_path': result.tool_calls in (0, 1) and result.steps == 1 + result.tool_calls
            and len(record['model_calls']) == 1 + result.tool_calls and record['retrieval_requests'] == result.tool_calls,
        'completed_within_two_request_budget': len(record['model_calls']) <= 2 and all(c['status'] == 'completed' for c in record['model_calls']),
        'answer_matches_last_raw_response': result.answer == record['model_calls'][-1]['response']['message']['content'],
        'retrieval_path_matches_actual_tool_output': True,
    }
    if result.tool_calls:
        second = record['model_calls'][1]
        tool_results = [m for m in second['official_messages'] if m['role'] == 'tool']
        expected = [{'chunk_id': c.chunk_id, 'conversation_id': c.conversation_id,
            'rounds': [c.start_round, c.end_round], 'text': c.text} for c in result.retrieved_chunks]
        checks['retrieval_path_matches_actual_tool_output'] = (len(expected) == 5 and len(tool_results) == 1
            and json.loads(tool_results[0]['content']) == expected and 'tools' not in second['native_request']
            and [c.chunk_id for c in result.retrieved_chunks] == record['retrieval']['returned_top5_ids'])
    else:
        checks['retrieval_path_matches_actual_tool_output'] = not result.retrieved_chunks and 'retrieval' not in record
    record['checks'] = checks
    if not all(checks.values()):
        raise ValueError('Hybrid流程核对失败')
    record['reported_incremental_usage'] = {'chat_prompt_tokens': sum(c['reported_prompt_tokens'] for c in record['model_calls']),
        'chat_output_tokens': sum(c['reported_output_tokens'] for c in record['model_calls']),
        'new_embedding_forward_batches': record.get('embedding_forward_batches', 0),
        'cached_input_tokens': None, 'local_billed_money': None}
    print('实际路径：' + record['observed_path'], flush=True)
    print('回答：\n' + result.answer, flush=True)
    return result

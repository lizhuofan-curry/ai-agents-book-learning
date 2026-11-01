"""原记忆Agent的RAG本机适配；旧JSONCards代码及结果保持不变。"""
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

import 本机记忆适配 as base

ROOT, PROJECT = base.ROOT, base.PROJECT
PLAN = ROOT / '测试材料/RAG单案例方案.json'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def augment_dependencies():
    # 复用已安装的rich/tenacity；root环境的torch/transformers仍优先，不安装新依赖。
    path = PROJECT / f'.venv-ch7-memory/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    if not path.is_dir():
        raise ValueError('缺少已准备的独立依赖目录')
    if str(path) not in sys.path:
        sys.path.append(str(path))


def verify_plan():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    original = base.load_plan()
    shared = ['upstream_commit', 'case_id', 'case_source', 'case_sha256', 'model', 'endpoint', 'num_ctx', 'temperature', 'seed', 'think']
    fixed = {'maximum_model_requests': 2, 'maximum_tool_calls': 1, 'maximum_embedding_forward_batches': 3,
        'embedding_batch_size': 8, 'expected_chunk_count': 11, 'rounds_per_chunk': 6, 'overlap': 2,
        'top_k': 5, 'reranker': 'none', 'num_predict': {'search_request': 1024, 'rag_answer': 1024},
        'embedding_model_id': 'BAAI/bge-small-zh-v1.5',
        'embedding_model_revision': '7999e1d3359715c523056ef9478215996d62a620',
        'model_input_normalization': 'casefold', 'automatic_retries': False, 'judge_requests': 0}
    if any(plan.get(k) != original[k] for k in shared) or any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('RAG单题、模型、切块、预算或参数方案变化')
    hashes = {}
    for name in ['bge_manifest', 'bge_verifier', 'bge_encoder']:
        path = (PROJECT / plan[name]).resolve()
        if not path.is_relative_to((PROJECT / '第三章/个人Wiki混合RAG').resolve()) or base.sha(path) != plan[name + '_sha256']:
            raise ValueError('既有BGE参考或实现变化：' + name)
        hashes[plan[name]] = base.sha(path)
    verifier = load(PROJECT / plan['bge_verifier'], 'rag_bge_file_guard')
    model_info, directory, files = verifier.verify_local_files()
    if (model_info['model_id'], model_info['revision']) != (plan['embedding_model_id'], plan['embedding_model_revision']):
        raise ValueError('固定BGE模型不符')
    return plan, hashes, model_info, directory, files


def native_messages(messages):
    """与已有CodingAgent适配相同：客户端维护ID，原生消息使用工具名。"""
    result, names = [], {}
    for message in messages:
        row = {'role': message['role'], 'content': message.get('content') or ''}
        if message.get('tool_calls'):
            row['tool_calls'] = []
            for call in message['tool_calls']:
                f = call['function']
                arguments = json.loads(f['arguments']) if isinstance(f['arguments'], str) else deepcopy(f['arguments'])
                names[call['id']] = f['name']
                row['tool_calls'].append({'function': {'name': f['name'], 'arguments': arguments}})
        if message['role'] == 'tool':
            row['tool_name'] = names[message['tool_call_id']]
        result.append(row)
    return result


class RAGChat:
    def __init__(self, official, plan, record, save, post=base.local_post):
        self.official, self.plan, self.record, self.save, self.post = official, plan, record, save, post

    def complete(self, messages, tools=None, tool_choice=None, json_object=False):
        calls = self.record['model_calls']
        if len(calls) >= 2:
            raise base.StopPipeline('已达两次模型请求预算，不重试')
        first = not calls
        stage = 'search_request' if first else 'rag_answer'
        expected_choice = {'type': 'function', 'function': {'name': 'search_memory'}} if first else 'none'
        if tools != [self.official.SEARCH_TOOL] or tool_choice != expected_choice or json_object:
            raise base.StopPipeline('仅支持一次search_memory后直接回答')
        request = {'model': base.MODEL, 'messages': native_messages(messages), 'stream': False, 'think': False,
            'options': {'temperature': 0, 'seed': 42, 'num_ctx': self.plan['num_ctx'],
                'num_predict': self.plan['num_predict'][stage]}}
        if first:
            request['tools'] = deepcopy(tools)
        entry = {'stage': stage, 'status': 'started', 'official_messages': deepcopy(messages),
            'official_tool_choice': deepcopy(tool_choice), 'native_request': request,
            'tool_choice_adaptation': 'Offer native tools then validate actual model selection; final stage omits tools. No synthetic retrieval call.'}
        calls.append(entry)
        self.save()
        print(f'第{len(calls)}次本机模型请求：{stage}', flush=True)
        started = time.perf_counter()
        try:
            status, raw = self.post(deepcopy(request))
            entry.update(http_status=status, raw_http_body=raw, client_wall_ms=(time.perf_counter() - started) * 1000)
            self.save()
            if status != 200:
                raise ValueError('本机模型HTTP状态：' + str(status))
            response = json.loads(raw)
            entry['response'] = response
            self.save()
            message = response.get('message', {})
            if (response.get('model') != base.MODEL or response.get('done') is not True
                    or response.get('done_reason') != 'stop' or message.get('role') != 'assistant'):
                raise ValueError('模型标签、结束状态或角色异常；保留失败不重试')
            pin, pout = response.get('prompt_eval_count'), response.get('eval_count')
            if any(type(x) is not int or x < 0 for x in [pin, pout]):
                raise ValueError('生成用量未知，不补0')
            raw_calls, parsed = message.get('tool_calls') or [], []
            content = message.get('content') or ''
            if not isinstance(content, str):
                raise ValueError('模型正文类型异常')
            assistant = {'role': 'assistant', 'content': content}
            if first:
                if not isinstance(raw_calls, list) or len(raw_calls) != 1:
                    raise ValueError('首次响应必须实际选择一次search_memory；不伪造检索、不补请求')
                function = raw_calls[0].get('function', {})
                args = function.get('arguments')
                if (function.get('name') != 'search_memory' or not isinstance(args, dict) or set(args) != {'query'}
                        or not isinstance(args['query'], str) or not args['query'].strip() or len(args['query']) > 1000):
                    raise ValueError('工具名、查询参数或长度不符')
                call_id = 'local-rag-1-0'
                parsed = [self.official.ToolCall(call_id, 'search_memory', deepcopy(args))]
                assistant['tool_calls'] = [{'id': call_id, 'type': 'function', 'function': {
                    'name': 'search_memory', 'arguments': json.dumps(args, ensure_ascii=False)}}]
                print('模型选择：search_memory ' + json.dumps(args, ensure_ascii=False), flush=True)
            elif raw_calls or not content.strip():
                raise ValueError('回答阶段出现额外工具调用或空回答')
            entry.update(status='completed', reported_prompt_tokens=pin, reported_output_tokens=pout)
            usage = self.official.Usage(input_tokens=pin, output_tokens=pout,
                unpriced_tokens=pin + pout, unpriced_requests=1)
            return self.official.ChatTurn(content, parsed, usage, entry['client_wall_ms'], assistant)
        except Exception as exc:
            entry.update(status='failed_or_incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            self.save()


class LocalBGE:
    def __init__(self, official, plan, record, save, directory, chunk_ids):
        self.official, self.plan, self.record, self.save = official, plan, record, save
        self.directory, self.chunk_ids, self.calls = directory, chunk_ids, 0
        self.last_usage, self.last_latency_ms = official.Usage(), 0.0
        self.tokenizer = self.model = None

    def embed(self, texts):
        if self.calls >= 2 or (self.calls == 0 and len(texts) != 11) or (self.calls == 1 and len(texts) != 1):
            raise base.StopPipeline('嵌入仅允许11个原文片段及1条实际查询')
        kind = 'documents' if self.calls == 0 else 'query'
        self.calls += 1
        started = time.perf_counter()
        if self.model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            _, _, info, model_dir, files = verify_plan()
            self.record['embedding_model'] = info
            self.record['embedding_files_verified'] = files
            torch.manual_seed(42)
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                trust_remote_code=False, use_safetensors=True).to('cpu').eval()
            if self.model.config.hidden_size != 512 or self.model.config.max_position_embeddings != 512:
                raise ValueError('嵌入模型维度/长度不符')
        encoder = load(PROJECT / self.plan['bge_encoder'], 'rag_reused_bge_encoder')
        instruction = self.record['embedding_model']['query_instruction']
        inputs = [x.casefold() for x in texts] if kind == 'documents' else [(instruction + texts[0]).casefold()]
        ids = self.chunk_ids if kind == 'documents' else ['actual_model_search_query']
        self.record.setdefault('embedding_model_inputs', {})[kind] = inputs
        self.save()
        vectors = encoder.encode(inputs, ids, kind, self.tokenizer, self.model, self.record, self.save, 8, 3)
        import numpy as np
        np.save(self.directory / (kind + '.npy'), vectors, allow_pickle=False)
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        tokens = sum(self.record['token_lengths'][kind])
        self.last_usage = self.official.Usage(input_tokens=tokens, unpriced_tokens=tokens, unpriced_requests=1)
        self.record.setdefault('embedding_usage', {})[kind] = {'tokenizer_input_tokens': tokens,
            'latency_ms_including_first_load_if_any': self.last_latency_ms, 'local_billed_money': None}
        self.save()
        return vectors.tolist()


def run_rag_path(official, case, plan, record, save, directory, post=base.local_post, embedder=None):
    chunks = official.conversation_chunks(case, rounds_per_chunk=6, overlap=2)
    if len(chunks) != 11 or len({c.chunk_id for c in chunks}) != 11:
        raise ValueError('固定案例应为11个原文片段')
    base.dump(directory / 'chunks.json', [asdict(c) for c in chunks])
    if embedder is None:
        embedder = LocalBGE(official, plan, record, save, directory, [c.chunk_id for c in chunks])

    class TrackedIndex(official.VectorMemoryIndex):
        def search(self, query, candidate_k=20):
            if record.get('retrieval_requests', 0) >= 1:
                raise base.StopPipeline('只允许一次实际检索')
            record['retrieval_requests'] = 1
            save()
            ranked, usage, latency = super().search(query, candidate_k)
            row = {'query': query, 'candidates': [{'chunk_id': c.chunk_id, 'cosine': score} for c, score in ranked],
                'returned_top5_ids': [c.chunk_id for c, _ in ranked[:5]]}
            record.setdefault('retrievals', []).append(row)
            base.dump(directory / 'retrieval.json', row)
            save()
            print('实际检索Top5：' + ' → '.join(row['returned_top5_ids']), flush=True)
            return ranked, usage, latency

    index = TrackedIndex(chunks, embedder)
    chat = RAGChat(official, plan, record, save, post)
    result = official.MemoryAgent(chat).rag(case.user_question, index, official.NoReranker(),
        top_k=5, allow_followup_searches=False, max_search_rounds=1)
    record['answer'] = result.answer
    record['agent_result'] = {'steps': result.steps, 'tool_calls': result.tool_calls,
        'trace': result.trace, 'latency_ms_excluding_index_build': result.latency_ms,
        'retrieved_chunks': [asdict(c) for c in result.retrieved_chunks]}
    (directory / 'answer.txt').write_text(result.answer, encoding='utf-8')
    base.dump(directory / 'retrieved_top5.json', record['agent_result']['retrieved_chunks'])
    first, second = record['model_calls']
    source_tool_results = [m for m in second['official_messages'] if m['role'] == 'tool']
    expected = [{'chunk_id': c.chunk_id, 'conversation_id': c.conversation_id,
        'rounds': [c.start_round, c.end_round], 'text': c.text} for c in result.retrieved_chunks]
    checks = {
        'exactly_two_completed_model_requests': len(record['model_calls']) == 2 and all(c['status'] == 'completed' for c in record['model_calls']),
        'model_actually_requested_one_search': len(first['response']['message'].get('tool_calls', [])) == 1,
        'two_agent_steps_one_actual_tool': result.steps == 2 and result.tool_calls == 1 and record['retrieval_requests'] == 1,
        'question_only_initial_session': first['official_messages'] == [{'role': 'system', 'content': official.SYSTEM_RULES}, {'role': 'user', 'content': case.user_question}],
        'only_actual_top5_source_chunks_forwarded': len(expected) == 5 and len(source_tool_results) == 1
            and json.loads(source_tool_results[0]['content']) == expected,
        'native_tool_result_mapping_matches': second['native_request']['messages'] == native_messages(second['official_messages']),
        'no_tools_offered_for_final_answer': 'tools' not in second['native_request'],
        'top5_matches_saved_cosine_ranking': [c.chunk_id for c in result.retrieved_chunks] == record['retrievals'][0]['returned_top5_ids'],
        'saved_answer_matches_model_response': result.answer == second['response']['message']['content'],
    }
    record['checks'] = checks
    if not all(checks.values()):
        raise ValueError('RAG流程检查失败')
    record['reported_usage'] = {'chat_prompt_tokens': sum(c['reported_prompt_tokens'] for c in record['model_calls']),
        'chat_output_tokens': sum(c['reported_output_tokens'] for c in record['model_calls']),
        'embedding_input_tokens': {k: v['tokenizer_input_tokens'] for k, v in record.get('embedding_usage', {}).items()},
        'cached_input_tokens': None, 'local_billed_money': None,
        'scope': 'Chat and BGE input counts kept separate; no currency or missing-cache substitution.'}
    print('回答：\n' + result.answer, flush=True)
    return result

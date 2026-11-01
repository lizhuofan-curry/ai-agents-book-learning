"""只读60案例，核对固定切块/分词长度及理论预算；0次模型或网络请求。"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import 本机记忆适配 as base
import RAG本机适配 as rag

ROOT = base.ROOT
PLAN = ROOT / '测试材料/60案例离线预检方案.json'
CHINA = timezone(timedelta(hours=8))


class PromptCaptured(BaseException):
    pass


def card_prompt(official, case):
    class Capture:
        def complete(self, messages, **kwargs):
            self.messages = messages
            raise PromptCaptured()
    chat = Capture()
    try:
        official.CardBuilder(chat).build(case)
    except PromptCaptured:
        pass
    if len(chat.messages) != 1 or chat.messages[0]['role'] != 'user':
        raise ValueError('原提取器输入结构变化')
    return chat.messages[0]['content']


def load_inputs():
    plan = json.loads(PLAN.read_text())
    fixed = {'upstream_commit': 'dbc046eb896ac4e39aa19c7774c8bf49583b89a6', 'expected_cases': 60,
        'expected_cases_per_layer': 20, 'rounds_per_chunk': 6, 'overlap': 2, 'embedding_batch_size': 8,
        'embedding_max_tokens': 512, 'model_input_normalization': 'casefold', 'qwen_model': 'qwen3.5:9b',
        'qwen_num_ctx': 16384, 'writer_output_limit': 4096, 'answer_and_search_output_limit': 1024,
        'proposed_local_generation_cap_per_case': 6, 'proposed_judge_cap_per_case': 3,
        'all_real_model_and_network_requests': 0}
    if any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('离线预检范围、分词配置或预算草案变化')
    guard = rag.load(Path(__file__).with_name('01_官方案例与评分规则离线检查.py'), 'sixty_case_guard')
    hashes = guard.verify_reference()
    _, bge_sources, info, directory, files = rag.verify_plan()
    return plan, hashes, bge_sources, info, directory, files


def scan(official, cases, tokenizer, plan, source_files):
    import yaml
    by_id = {}
    for path in source_files:
        data = yaml.safe_load(path.read_text())
        if data['test_id'] in by_id:
            raise ValueError('案例文件ID重复')
        by_id[data['test_id']] = path
    if len(cases) != 60 or len({c.test_id for c in cases}) != 60 or len(by_id) != 60:
        raise ValueError('须为60个独立官方案例，不把路径/配置数当案例数')
    rows = []
    for case in sorted(cases, key=lambda c: (c.category, c.test_id)):
        chunks = official.conversation_chunks(case, 6, 2)
        inputs = [c.text.casefold() for c in chunks]
        ids = tokenizer(inputs, padding=False, truncation=False)['input_ids']
        lengths = [len(x) for x in ids]
        if not lengths or len(lengths) != len(chunks):
            raise ValueError('案例没有完整原文片段：' + case.test_id)
        unk = tokenizer.unk_token_id
        over = [{'chunk_id': c.chunk_id, 'tokens': n, 'rounds': [c.start_round, c.end_round]}
            for c, n in zip(chunks, lengths) if n > 512]
        prompt = card_prompt(official, case)
        path = by_id[case.test_id]
        rows.append({'case_id': case.test_id, 'layer': case.category,
            'source_path': str(path.relative_to(ROOT / '官方资料/upstream')), 'source_sha256': base.sha(path),
            'conversations': len(case.conversation_histories),
            'messages': sum(len(h.messages) for h in case.conversation_histories),
            'card_prompt_characters': len(prompt), 'card_prompt_utf8_bytes': len(prompt.encode()),
            'card_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'qwen_prompt_tokens': None, 'qwen_context_fits': None,
            'chunks': len(chunks), 'chunk_token_lengths': lengths,
            'minimum_chunk_tokens': min(lengths), 'maximum_chunk_tokens': max(lengths),
            'unknown_token_count': sum(sum(t == unk for t in x) for x in ids) if unk is not None else None,
            'overlength_chunks': over, 'fits_bge_without_truncation': not over,
            'hypothetical_document_forward_batches': math.ceil(len(chunks) / 8)})
    counts = dict(sorted(Counter(r['layer'] for r in rows).items()))
    if counts != {'layer1': 20, 'layer2': 20, 'layer3': 20}:
        raise ValueError('须为三层各20题')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='只核对资料/模型清单，不分词、不保存结果')
    args = parser.parse_args()
    os.environ.update(PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1', MAX_RETRIES='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    attempts = []
    def audit(event, values):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            attempts.append(event)
            raise RuntimeError('60案例离线预检禁止网络连接')
    sys.addaudithook(audit)
    plan, hashes, bge_sources, model_info, model_dir, files = load_inputs()
    print('第七章60案例离线预检：原始案例、固定切块、BGE分词长度和理论预算。')
    print('0次生成、裁判、嵌入前向或网络请求；不截断、不更换切块、不启动批量实验。')
    if args.preview:
        print('SIXTY_PREFLIGHT_PREVIEW：固定资料与模型文件已核对，未分词或调用模型。')
        return 0
    directory = ROOT / '本地运行结果' / ('sixty_preflight_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(), 'plan': plan,
        'script_sha256': base.sha(Path(__file__)), 'plan_sha256': base.sha(PLAN),
        'official_reference_hashes': hashes, 'bge_source_hashes': bge_sources, 'embedding_files_verified': files,
        'generation_requests': 0, 'judge_requests': 0, 'embedding_forward_batches': 0, 'network_requests': 0,
        'scope': plan['scope']}
    def save():
        base.dump(directory / 'record.json', record)
    save()
    print('记录路径：' + str(directory / 'record.json'), flush=True)
    try:
        rag.augment_dependencies()
        official = base.load_official()
        cases = official.UserMemoryEvaluationFramework().test_suite.test_cases
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        source_files = sorted((ROOT / '官方资料/upstream/chapter3/user-memory-evaluation/test_cases').glob('layer*/*.yaml'))
        rows = scan(official, cases, tokenizer, plan, source_files)
        blocked = [r for r in rows if not r['fits_bge_without_truncation']]
        record['cases'] = rows
        record['summary'] = {'cases': len(rows), 'chunks': sum(r['chunks'] for r in rows),
            'bge_overlength_cases': len(blocked), 'bge_overlength_chunks': sum(len(r['overlength_chunks']) for r in rows),
            'maximum_bge_chunk_tokens': max(r['maximum_chunk_tokens'] for r in rows),
            'maximum_card_prompt_characters': max(r['card_prompt_characters'] for r in rows),
            'qwen_prompt_token_counts': 'unknown; BGE lengths and character counts are not Qwen token counts'}
        record['theoretical_from_scratch_budget_only'] = {'local_generation_cap': len(rows) * 6,
            'cloud_judge_cap': len(rows) * 3, 'total_model_request_cap': len(rows) * 9,
            'document_forward_batches_if_all_chunks_fit': sum(r['hypothetical_document_forward_batches'] for r in rows),
            'query_forward_batches_cap_if_all_paths_complete': len(rows) * 2,
            'scope': 'Proposed bounded local-adaptation protocol, not upstream retry behavior, an implemented batch runner, current consumption or monetary estimate. Existing pilot reuse not decided here.'}
        record['readiness'] = {'bge_chunks_fit_fixed_configuration': not blocked,
            'qwen_token_lengths_verified': False, 'generic_batch_runner_prepared': False,
            'batch_execution_started': False}
        record['checks'] = {'all_sixty_cases_scanned': len(rows) == 60,
            'three_layers_twenty_cases_each': dict(Counter(r['layer'] for r in rows)) == {'layer1': 20, 'layer2': 20, 'layer3': 20},
            'overlength_inputs_preserved_without_truncation': all(len(r['chunk_token_lengths']) == r['chunks'] for r in rows),
            'unknown_qwen_counts_not_filled_zero': all(r['qwen_prompt_tokens'] is None and r['qwen_context_fits'] is None for r in rows),
            'original_references_and_weights_unchanged': load_inputs()[1:3] == (hashes, bge_sources),
            'no_network_attempts': not attempts}
        if not all(record['checks'].values()):
            raise ValueError('预检资料一致性检查未通过')
        record['status'] = 'preflight_saved_execution_not_started'
        for layer in ['layer1', 'layer2', 'layer3']:
            selected = [r for r in rows if r['layer'] == layer]
            print(layer + '：20题，超BGE长度案例=' + str(sum(not r['fits_bge_without_truncation'] for r in selected))
                + '，最大片段tokens=' + str(max(r['maximum_chunk_tokens'] for r in selected)))
        print('从零运行的本机协议草案上限：本机生成360次、云端裁判180次；这里只估算次数，不启动请求。')
        print('Qwen实际输入token长度尚未核实；现有单题脚本不能直接当通用60题入口。')
        print('SIXTY_PREFLIGHT_SAVED：离线预检已保存；长度问题、批量协议和预算仍须处理，不代表60题评估完成。')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete_preflight', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        record['network_audit'] = {'blocked_events': attempts}
        save()


if __name__ == '__main__':
    raise SystemExit(main())

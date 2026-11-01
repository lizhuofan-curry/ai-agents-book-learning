"""同案例Hybrid：原核心卡片常驻，可自主检索；复用冻结文档向量。"""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import 本机记忆适配 as base
import RAG本机适配 as rag
import Hybrid本机适配 as hybrid

ROOT, PROJECT = base.ROOT, base.PROJECT
CHINA = timezone(timedelta(hours=8))


def references():
    return rag.load(Path(__file__).with_name('01_官方案例与评分规则离线检查.py'), 'hybrid_reference_guard').verify_reference()


def code_hashes():
    return {p.name: base.sha(p) for p in [Path(__file__), Path(base.__file__), Path(rag.__file__), Path(hybrid.__file__)]}


def worker(directory):
    path = directory / 'record.json'
    record = json.loads(path.read_text())
    save = lambda: base.dump(path, record)
    network = {'allowed_loopback_events': [], 'blocked_events': []}

    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            allowed = (event == 'socket.connect' and args[1] == ('127.0.0.1', 11434)) or (
                event == 'socket.getaddrinfo' and args[0] == '127.0.0.1' and args[1] == 11434)
            if not allowed:
                network['blocked_events'].append(event)
                raise base.StopPipeline('Hybrid只允许固定Ollama回环地址')
            network['allowed_loopback_events'].append(event)

    sys.addaudithook(audit)
    try:
        plan, source = hybrid.verify_inputs()
        if (plan != record['plan'] or references() != record['reference_hashes']
                or code_hashes() != record['implementation_sha256'] or base.sha(hybrid.PLAN) != record['input_plan_sha256']):
            raise ValueError('运行前来源或实现变化')
        rag.augment_dependencies()
        record['versions'] = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'numpy', 'safetensors', 'rich', 'tenacity']}
        official = base.load_official()
        case = official.UserMemoryEvaluationFramework().get_test_case(plan['case_id'])
        record['status'] = 'running'
        save()
        hybrid.run_hybrid_path(official, case, plan, source, record, save, directory)
        searched = record['retrieval_requests'] == 1
        record['checks'].update(
            original_sources_and_vector_cache_unchanged=hybrid.verify_inputs()[0] == plan and references() == record['reference_hashes'],
            no_external_network_attempts=not network['blocked_events'],
            new_embedding_batches_match_actual_search=record['embedding_forward_batches'] == int(searched)
                and len(record['batches']) == int(searched) and all(b['status'] == 'completed' for b in record['batches']),
            no_query_input_truncation=(not searched) or (max(record['token_lengths']['query']) <= 512),
            document_vectors_reused_without_new_build=record.get('document_vectors_reused') is True)
        if not all(record['checks'].values()):
            raise ValueError('Hybrid来源或实际路径检查失败')
        record['status'] = 'hybrid_answer_saved_source_review_pending'
        print('检查：' + json.dumps(record['checks'], ensure_ascii=False), flush=True)
        print('HYBRID_MEMORY_SAVED：路径与回答已保存；原卡片问题保留，内容待来源核对，尚无裁判分数。', flush=True)
        return 0
    except (Exception, base.StopPipeline, KeyboardInterrupt) as exc:
        record.update(status='failed_or_incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('HYBRID_MEMORY_INCOMPLETE：失败原样保存，不重试。' + str(exc), flush=True)
        return 1
    finally:
        record['network_audit'] = network
        record['artifacts_sha256'] = {p.name: base.sha(p) for p in directory.iterdir() if p.is_file()
            and p.name in {'core_cards.json', 'query.npy', 'retrieval.json', 'retrieved_top5.json', 'answer.txt'}}
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        directory = args.worker.resolve()
        if directory.parent != (ROOT / '本地运行结果').resolve() or not directory.is_dir():
            raise ValueError('内部worker目录越界')
        return worker(directory)
    plan, source = hybrid.verify_inputs()
    hashes = references()
    print('第七章记忆评估第四步：原11张核心卡片＋可选原文检索，独立Hybrid会话。')
    print('最多2次本机qwen3.5:9b请求、1次search_memory、1批新查询嵌入；可能只回答而不检索。')
    print('复用原卡片与冻结文档向量，不重提取、不修复旧卡、不重建文档向量；0次裁判或外部网络请求。')
    print('每次输出最多1024、上下文16384、temp0/seed42/think=false；无重试或下载。')
    if args.preview:
        print('HYBRID_MEMORY_PREVIEW：官方资料、原卡片/向量/审阅和固定方案已核对；0次生成或嵌入前向。')
        return 0
    directory = ROOT / '本地运行结果' / ('hybrid_local_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    build_call = source['cards']['record']['model_calls'][0]
    inherited = {'cards_build': {'source_run': plan['cards_source_run'], 'model_requests': 1,
        'reported_prompt_tokens': build_call['reported_prompt_tokens'], 'reported_output_tokens': build_call['reported_output_tokens']},
        'document_index': {'source_run': plan['rag_source_run'], 'embedding_forward_batches': 2,
            'tokenizer_input_tokens': source['rag']['record']['embedding_usage']['documents']['tokenizer_input_tokens']},
        'scope': 'Construction work already performed in source runs; separate from new incremental use, not free/zero full-path cost.'}
    record = {'status': 'prepared', 'created_at': datetime.now(CHINA).isoformat(), 'plan': plan,
        'input_plan_sha256': base.sha(hybrid.PLAN), 'implementation_sha256': code_hashes(),
        'reference_hashes': hashes, 'inherited_construction_usage': inherited,
        'known_source_content_issues_preserved': ['Cards contain incorrect overdraft choice and metadata qualifications; no source card was repaired.',
            'Prior pure-RAG Top5 missed the requested account; this run does not reuse its query or answer.'],
        'model_calls': [], 'retrieval_requests': 0, 'embedding_forward_batches': 0, 'batches': [],
        'maximum_model_requests': 2, 'maximum_new_embedding_forward_batches': 1,
        'judge_requests': 0, 'external_network_requests': 0, 'generation_model_weight_digest': None,
        'semantic_review': 'pending', 'scope': plan['scope']}
    base.dump(directory / 'record.json', record)
    print('记录路径：' + str(directory / 'record.json'), flush=True)
    env = {k: v for k, v in os.environ.items() if not k.endswith('_API_KEY') and k not in {'PYTHONPATH', 'PYTHONHOME'}}
    env.update(PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', MAX_RETRIES='1', REQUEST_TIMEOUT='30',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    with (directory / 'terminal_output.txt').open('w', encoding='utf-8') as output:
        child = subprocess.Popen([str(PROJECT / '.venv/bin/python'), str(Path(__file__).resolve()), '--worker', str(directory)],
            env=env, cwd=directory, text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in child.stdout:
            print(line, end='', flush=True)
            output.write(line)
            output.flush()
        return child.wait()


if __name__ == '__main__':
    raise SystemExit(main())

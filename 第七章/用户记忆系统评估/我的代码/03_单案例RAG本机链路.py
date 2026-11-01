"""同一官方单案例：真实模型检索工具调用、本地BGE和原MemoryAgent.rag。"""
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

ROOT, PROJECT = base.ROOT, base.PROJECT
CHINA = timezone(timedelta(hours=8))


def references():
    module = rag.load(Path(__file__).with_name('01_官方案例与评分规则离线检查.py'), 'rag_official_guard')
    return module.verify_reference()


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
                raise base.StopPipeline('RAG只允许固定Ollama回环地址')
            network['allowed_loopback_events'].append(event)

    sys.addaudithook(audit)
    try:
        plan, deps, _, _, _ = rag.verify_plan()
        if (plan != record['plan'] or references() != record['reference_hashes']
                or deps != record['bge_source_hashes'] or base.sha(Path(__file__)) != record['script_sha256']
                or base.sha(Path(rag.__file__)) != record['adapter_sha256']
                or base.sha(Path(base.__file__)) != record['base_adapter_sha256']
                or base.sha(rag.PLAN) != record['input_plan_sha256']):
            raise ValueError('方案、脚本、适配或来源变化')
        rag.augment_dependencies()
        record['versions'] = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'numpy', 'safetensors', 'rich', 'tenacity']}
        official = base.load_official()
        case = official.UserMemoryEvaluationFramework().get_test_case(plan['case_id'])
        record['status'] = 'running'
        save()
        rag.run_rag_path(official, case, plan, record, save, directory)
        record['checks'].update(all_reference_files_unchanged=references() == record['reference_hashes'],
            bge_sources_and_weights_unchanged=rag.verify_plan()[1] == record['bge_source_hashes'],
            three_completed_embedding_batches=record['embedding_forward_batches'] == 3 and len(record['batches']) == 3
                and all(b['status'] == 'completed' for b in record['batches']),
            no_embedding_input_truncation=max(n for lengths in record['token_lengths'].values() for n in lengths) <= 512,
            no_external_network_attempts=not network['blocked_events'])
        if not all(record['checks'].values()):
            raise ValueError('来源、嵌入或网络检查失败')
        record['status'] = 'rag_answer_saved_source_review_pending'
        print('检查：' + json.dumps(record['checks'], ensure_ascii=False), flush=True)
        print('RAG_MEMORY_SAVED：真实检索与回答已保存；证据相关性和回答忠实性待核对，尚无裁判分数。', flush=True)
        return 0
    except (Exception, base.StopPipeline, KeyboardInterrupt) as exc:
        record.update(status='failed_or_incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('RAG_MEMORY_INCOMPLETE：失败原样保存，不重试。' + str(exc), flush=True)
        return 1
    finally:
        record['network_audit'] = network
        record['artifacts_sha256'] = {p.name: base.sha(p) for p in directory.iterdir() if p.is_file()
            and p.name in {'chunks.json', 'documents.npy', 'query.npy', 'retrieval.json', 'retrieved_top5.json', 'answer.txt'}}
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
    plan, deps, info, _, files = rag.verify_plan()
    hashes = references()
    print('第七章记忆评估第三步：同一案例，原始对话RAG；不读取JSONCards产物。')
    print('最多2次本机qwen3.5:9b请求、1次search_memory、3批CPU嵌入前向；0次裁判或外部网络请求。')
    print('官方6轮/重叠2轮切成11块，BGE-small-zh-v1.5 casefold；Top5，无重排或追加检索。')
    print('生成每次最多1024、上下文16384、temp0/seed42/think=false；不重试、不下载。')
    if args.preview:
        print('RAG_MEMORY_PREVIEW：73份官方资料、7份模型文件和固定输入方案已核对；0次生成/嵌入前向。')
        return 0
    directory = ROOT / '本地运行结果' / ('rag_local_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'prepared', 'created_at': datetime.now(CHINA).isoformat(), 'plan': plan,
        'script_sha256': base.sha(Path(__file__)), 'adapter_sha256': base.sha(Path(rag.__file__)),
        'base_adapter_sha256': base.sha(Path(base.__file__)),
        'input_plan_sha256': base.sha(rag.PLAN), 'reference_hashes': hashes, 'bge_source_hashes': deps,
        'embedding_model': info, 'embedding_files_verified': files, 'model_calls': [],
        'embedding_forward_batches': 0, 'batches': [], 'maximum_model_requests': 2,
        'maximum_embedding_forward_batches': 3, 'retrieval_requests': 0,
        'judge_requests': 0, 'external_network_requests': 0,
        'generation_model_weight_digest': None, 'semantic_review': 'pending', 'scope': plan['scope']}
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

"""三层各1题的Cards小批；最多6次本机请求，默认真实运行，无云端评分。"""
import argparse
from datetime import datetime
import fcntl
import importlib.metadata
import os
from pathlib import Path
import sys

import 本机记忆适配 as base
import RAG本机适配 as rag
import 批量JSONCards适配 as batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preview', action='store_true', help='仅离线核对来源，无请求')
    modes.add_argument('--self-test', action='store_true', help='离线假HTTP检查，无真实模型或网络请求')
    modes.add_argument('--resume', type=Path, help='仅恢复本协议原目录，跳过所有已开始题，绝不重发')
    args = parser.parse_args()
    os.environ.update(PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1', MAX_RETRIES='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    # 本程序不使用凭据/云端SDK；不把继承的Key交给官方模块。
    for key in list(os.environ):
        if key.endswith('_API_KEY'):
            os.environ.pop(key)
    network = {'allowed_events': [], 'blocked_events': []}
    def audit(event, values):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            allow = not (args.preview or args.self_test) and (
                (event == 'socket.connect' and values[1] == ('127.0.0.1', 11434))
                or (event == 'socket.getaddrinfo' and values[0] == '127.0.0.1' and values[1] == 11434))
            network['allowed_events' if allow else 'blocked_events'].append(event)
            if not allow:
                raise base.StopPipeline('只允许本机固定Ollama；离线模式禁止所有网络连接')
    sys.addaudithook(audit)
    plan, cache, provenance = batch.verify_inputs()
    print('第七章跨层Cards小批：三层各1条新案例；每题提取后独立回答。', flush=True)
    print('默认最多6次本机qwen3.5:9b请求；0次裁判、检索、嵌入或外部网络请求。', flush=True)
    print('实际消息逐次检查长度；不截断、不重试、不扩跑60题；恢复只继续未开始题。', flush=True)
    if args.preview:
        print('CARDS_BATCH_PREVIEW：固定来源与分词资产已核对，0次真实请求。')
        return 0
    rag.augment_dependencies()
    official = base.load_official()
    framework = official.UserMemoryEvaluationFramework()
    selected = [framework.get_test_case(x['case_id']) for x in plan['cases']]
    if any(c is None or c.category != s['layer'] for c, s in zip(selected, plan['cases'])):
        raise ValueError('三个固定案例缺失或层次变化')
    if args.self_test:
        tests = rag.load(Path(__file__).with_name('跨层Cards离线检查.py'), 'cards_batch_self_test')
        return tests.run(official, selected, plan, provenance, network)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(cache), local_files_only=True, trust_remote_code=False)
    root = (base.ROOT / '本地运行结果').resolve()
    root.mkdir(exist_ok=True)
    # 跨创建/恢复共用排他锁，防止两个终端同时消费同一批预算。
    with (root / '.cards_cross_layer.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('已有跨层Cards进程运行，不并发重发')
        if args.resume:
            directory = args.resume.resolve()
            if directory.parent != root or not directory.name.startswith('cards_cross_layer_'):
                raise ValueError('恢复目录越界或不属于本协议')
            record = batch.load_checkpoint(directory / 'record.json')
            batch.check_resume(record, plan, provenance, directory)
            record.setdefault('resume_events', []).append({'at': datetime.now(batch.CHINA).isoformat(),
                'policy': 'Skip every previously started case; continue untouched cases only.'})
        else:
            directory = root / ('cards_cross_layer_' + datetime.now(batch.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
            directory.mkdir(exist_ok=False)
            record = batch.fresh_record(plan, provenance)
        record['runtime_versions'] = {n: importlib.metadata.version(n) for n in ['transformers', 'tokenizers', 'rich', 'tenacity']}
        previous_network = record['network_audit']
        def save():
            record['network_audit'] = {k: previous_network[k] + network[k] for k in network}
            batch.persist(directory / 'record.json', record)
        save()
        print('记录路径：' + str(directory / 'record.json'), flush=True)
        def guard():
            if batch.verify_inputs()[2] != provenance:
                raise ValueError('发送前来源/清单/实现变化')
        try:
            batch.run_cases(official, selected, record, plan, tokenizer, directory, save, guard)
            guard()
            if network['blocked_events']:
                raise ValueError('存在被阻断的外部连接尝试')
            print('小批摘要：' + str(record['summary']), flush=True)
            print('CARDS_BATCH_SAVED：成功和失败已保存；卡片忠实性与回答质量待逐题核对，尚无评分。', flush=True)
            return 0 if record['summary']['answers_saved'] == 3 else 1
        except (Exception, base.StopPipeline, KeyboardInterrupt) as exc:
            record.update(status='batch_interrupted_or_blocked', error={'type': type(exc).__name__, 'message': str(exc)})
            print('CARDS_BATCH_INCOMPLETE：保留已占预算和原响应，不重试。' + str(exc), flush=True)
            return 1
        finally:
            save()
            print('记录路径：' + str(directory / 'record.json'), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

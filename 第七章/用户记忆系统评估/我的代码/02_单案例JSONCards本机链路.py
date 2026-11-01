"""单个官方合成案例：最多两次本机请求构建卡片并独立回答，不调用裁判。"""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import 本机记忆适配 as local

ROOT, PROJECT = local.ROOT, local.PROJECT
CHINA = timezone(timedelta(hours=8))


def references():
    source = Path(__file__).with_name('01_官方案例与评分规则离线检查.py')
    spec = importlib.util.spec_from_file_location('memory_reference_guard', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_reference()


def worker(directory):
    record_path = directory / 'record.json'
    record = json.loads(record_path.read_text())

    def save():
        local.dump(record_path, record)

    network = {'allowed_loopback_events': [], 'blocked_events': []}

    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            allowed = (event == 'socket.connect' and args[1] == ('127.0.0.1', 11434)) or (
                event == 'socket.getaddrinfo' and args[0] == '127.0.0.1' and args[1] == 11434)
            if not allowed:
                network['blocked_events'].append(event)
                raise local.StopPipeline('仅允许固定Ollama回环地址')
            network['allowed_loopback_events'].append(event)

    sys.addaudithook(audit)
    try:
        plan = local.load_plan()
        if plan != record['plan'] or references() != record['reference_hashes']:
            raise ValueError('运行前来源或方案变化')
        official = local.load_official()
        framework = official.UserMemoryEvaluationFramework()
        case = framework.get_test_case(plan['case_id'])
        if case is None or case.category != 'layer1':
            raise ValueError('缺少固定单案例')
        record['status'] = 'running'
        save()
        local.run_cards_path(official, case, plan, record, save, directory)
        record['checks']['all_reference_files_unchanged'] = references() == record['reference_hashes']
        record['checks']['no_external_network_attempts'] = not network['blocked_events']
        if not all(record['checks'].values()):
            raise ValueError('来源或网络检查失败')
        record['status'] = 'cards_and_answer_saved_semantic_review_pending'
        print('检查：' + json.dumps(record['checks'], ensure_ascii=False), flush=True)
        print('CARD_MEMORY_SAVED：仅完成JSONCards路径；卡片忠实性与回答质量待核对，尚无裁判分数。', flush=True)
        return 0
    except (Exception, local.StopPipeline, KeyboardInterrupt) as exc:
        record.update(status='failed_or_incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('CARD_MEMORY_INCOMPLETE：失败原样保存，不自动重试。' + str(exc), flush=True)
        return 1
    finally:
        record['network_audit'] = network
        record['artifacts_sha256'] = {p.name: local.sha(p) for p in directory.iterdir()
            if p.is_file() and p.name in {'cards.json', 'answer.txt'}}
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='只核对来源/方案/环境，不发送模型请求')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        directory = args.worker.resolve()
        if directory.parent != (ROOT / '本地运行结果').resolve() or not directory.is_dir():
            raise ValueError('内部worker目录越界')
        return worker(directory)
    plan, hashes = local.load_plan(), references()
    python = PROJECT / '.venv-ch7-memory/bin/python'
    if not python.is_file():
        raise ValueError('缺少已准备的独立环境；不自动安装或回退')
    print('第七章记忆评估第二步：单案例Advanced JSON Cards，提取后独立回答。')
    print('最多2次本机qwen3.5:9b请求：提取1次、回答1次；0次嵌入、检索、裁判或外部网络请求。')
    print('上下文16384；提取输出最多4096、回答1024；无自动重试、云端回退或模型下载。')
    if args.preview:
        print('CARD_MEMORY_PREVIEW：固定73份参考与单案例方案已核对；尚未发送请求。')
        return 0
    directory = ROOT / '本地运行结果' / ('cards_local_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'prepared', 'created_at': datetime.now(CHINA).isoformat(), 'plan': plan,
        'script_sha256': local.sha(Path(__file__)), 'adapter_sha256': local.sha(Path(local.__file__)),
        'input_plan_sha256': local.sha(local.PLAN), 'reference_hashes': hashes,
        'model_calls': [], 'maximum_model_requests': 2, 'new_embedding_forward_batches': 0,
        'retrieval_requests': 0, 'judge_requests': 0, 'external_network_requests': 0,
        'model_weight_digest': None, 'model_identity_scope': 'Ollama response model tag checked; weight digest not measured.',
        'semantic_review': 'pending', 'scope': plan['scope']}
    local.dump(directory / 'record.json', record)
    print('记录路径：' + str(directory / 'record.json'), flush=True)
    env = {k: v for k, v in os.environ.items() if not k.endswith('_API_KEY') and k not in {'PYTHONPATH', 'PYTHONHOME'}}
    env.update(PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1', MAX_RETRIES='1', REQUEST_TIMEOUT='30', PYTHONUTF8='1')
    try:
        with (directory / 'terminal_output.txt').open('w', encoding='utf-8') as output:
            child = subprocess.Popen([str(python), str(Path(__file__).resolve()), '--worker', str(directory)],
                env=env, cwd=directory, text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in child.stdout:
                print(line, end='', flush=True)
                output.write(line)
                output.flush()
            return child.wait()
    finally:
        print('记录路径：' + str(directory / 'record.json'), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

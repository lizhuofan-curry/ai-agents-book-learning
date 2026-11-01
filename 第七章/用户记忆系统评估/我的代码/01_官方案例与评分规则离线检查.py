"""读取官方60案例，演示原评分解析器；0次模型、嵌入、检索或网络请求。"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
REFERENCE = ROOT / '官方资料'
SHARED = REFERENCE / 'upstream/chapter3/user-memory-evaluation'
COMMIT = 'dbc046eb896ac4e39aa19c7774c8bf49583b89a6'
DIMENSIONS = ('precision', 'recall', 'reasoning', 'proactivity')
CHINA = timezone(timedelta(hours=8))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_reference():
    manifest = json.loads((REFERENCE / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['commit'] != COMMIT or len(manifest['files']) != 73:
        raise ValueError('固定参考提交或73份文件清单不符')
    hashes = {}
    for item in manifest['files']:
        path = (REFERENCE / 'upstream' / item['path']).resolve()
        if not path.is_relative_to((REFERENCE / 'upstream').resolve()):
            raise ValueError('参考路径越界')
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if len(raw) != item['bytes'] or blob != item['git_blob_sha1'] or sha(path) != item['sha256']:
            raise ValueError('固定参考变化：' + item['path'])
        hashes[item['path']] = item['sha256']
    return hashes


def grading_sheet(scores, hallucination=False):
    """人工指定的数字示例，仅演示公式；这些分数未经模型或人工语义裁判。"""
    grades = {1: 'fail', 2: 'pass', 3: 'good', 4: 'excellent'}
    return {
        'dimensions': {name: {'score': score, 'grade': grades[score],
            'reasoning': 'Synthetic numeric example, not an actual answer judgment.',
            'evidence': [], 'boundary_case': None}
            for name, score in zip(DIMENSIONS, scores)},
        'hallucination': {'detected': hallucination, 'claims': [], 'evidence': [],
            'reasoning': 'Veto flag is manually set for arithmetic demonstration.'},
        'overall_reasoning': '教学用预设评分单；不构成答案质量、记忆效果或裁判校准结果。',
        'required_info_found': {},
    }


def offline_worker(directory):
    attempts = []

    def no_network(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            attempts.append(event)
            raise RuntimeError('评分规则离线检查禁止网络请求')

    sys.addaudithook(no_network)
    try:
        sys.path.insert(0, str(SHARED))
        import dotenv
        if 'PYTHON_DOTENV_DISABLED' not in Path(dotenv.__file__).with_name('main.py').read_text(encoding='utf-8'):
            raise ValueError('dotenv不支持禁用读取开关')
        from framework import UserMemoryEvaluationFramework
        from evaluator import LLMEvaluator

        framework = UserMemoryEvaluationFramework()
        cases = framework.test_suite.test_cases
        layers = dict(sorted(Counter(case.category for case in cases).items()))
        if len(cases) != 60 or len({c.test_id for c in cases}) != 60 or layers != {'layer1': 20, 'layer2': 20, 'layer3': 20}:
            raise ValueError('官方案例须为三层各20题、共60个独立ID')
        case = framework.get_test_case('layer1_01_bank_account')
        if case is None:
            raise ValueError('缺少固定单题入口')
        # 原附带离线测试也使用此方式：跳过__init__和SDK客户端构造。
        evaluator = object.__new__(LLMEvaluator)
        prompt = evaluator._build_evaluation_prompt(case, '[OFFLINE PLACEHOLDER: no generated answer]', None)
        (directory / 'judge_prompt_preview.txt').write_text(prompt, encoding='utf-8')
        definitions = [
            ('core_good', [4, 3, 3, 2], False),
            ('recall_partial', [4, 2, 3, 2], False),
            ('hallucination_veto', [4, 3, 3, 2], True),
        ]
        examples = []
        for name, scores, veto in definitions:
            sheet = grading_sheet(scores, veto)
            result = evaluator._parse_evaluation_response(json.dumps(sheet), name)
            expected_reward = 0.0 if veto else round(float(sum(Decimal(s - 1) / 3 for s in scores) / 4), 6)
            expected_passed = not veto and all(s >= 3 for s in scores[:3])
            if result.reward != expected_reward or result.passed != expected_passed or result.veto_applied != veto:
                raise ValueError('原解析器与独立规则核算不一致：' + name)
            examples.append({'name': name, 'scores': dict(zip(DIMENSIONS, scores)),
                'synthetic_judge_payload': sheet, 'official_parsed_result': result.model_dump(mode='json'),
                'independent_expected_reward': expected_reward, 'independent_expected_passed': expected_passed})
        checks = {
            'sixty_distinct_official_cases_loaded': True,
            'three_layers_twenty_cases_each': True,
            'sample_prompt_contains_authoritative_source': 'AUTHORITATIVE CONVERSATION SOURCE' in prompt,
            'sample_prompt_contains_scale_and_veto': all(s in prompt for s in ['4 / excellent', '1 / fail', 'hallucination (VETO)']),
            'original_parser_matches_independent_arithmetic': True,
            'partial_recall_not_task_success': examples[1]['official_parsed_result']['passed'] is False,
            'hallucination_forces_zero_reward': examples[2]['official_parsed_result']['reward'] == 0,
        }
        if not all(checks.values()):
            raise ValueError('离线规则检查失败')
        save(directory / 'offline_rules.json', {
            'case_count': len(cases), 'layer_counts': layers,
            'case_ids': sorted(c.test_id for c in cases),
            'sample_case': {'test_id': case.test_id, 'question': case.user_question,
                'evaluation_criteria': case.evaluation_criteria, 'expected_behavior': case.expected_behavior,
                'conversation_count': len(case.conversation_histories)},
            'numeric_examples_only': examples, 'checks': checks,
            'scope': 'Original loader, prompt builder and parser only. No memory construction, retrieval, answer generation or semantic grading.'})
    finally:
        save(directory / 'network_audit.json', {'blocked_network_events': attempts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='只核对参考及环境位置，不保存运行')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        directory = args.worker.resolve()
        if directory.parent != (ROOT / '本地运行结果').resolve() or not directory.is_dir():
            raise ValueError('内部worker目录不合法')
        offline_worker(directory)
        return
    hashes = verify_reference()
    python = PROJECT / '.venv-ch7-memory/bin/python'
    if not python.is_file():
        raise ValueError('缺少已准备的独立.venv-ch7-memory环境；不自动安装或回退')
    print('第七章记忆评估第一步：官方60案例、四档评分与幻觉否决。')
    print('0次模型、嵌入、检索或网络请求；仅使用预设数字评分单演示原解析器。')
    if args.preview:
        print('MEMORY_RUBRIC_PREVIEW：73份固定参考哈希一致；未运行评分规则或真实记忆系统。')
        return
    directory = ROOT / '本地运行结果' / ('rubric_offline_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'prepared', 'created_at': datetime.now(CHINA).isoformat(),
        'script_sha256': sha(Path(__file__)), 'manifest_sha256': sha(REFERENCE / 'manifest.json'),
        'reference_hashes': hashes, 'new_model_requests': 0, 'new_embedding_forward_batches': 0,
        'retrieval_requests': 0, 'network_requests': 0,
        'scope': 'Offline official loader/prompt/parser and synthetic numeric grade demonstrations; not 60-case answer evaluation or three-system comparison.'}
    save(directory / 'record.json', record)
    print('记录路径：' + str(directory / 'record.json'), flush=True)
    env = {k: v for k, v in os.environ.items() if not k.endswith('_API_KEY') and k not in {'PYTHONPATH', 'PYTHONHOME'}}
    env.update(PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1', MAX_RETRIES='1', REQUEST_TIMEOUT='30', PYTHONUTF8='1')
    try:
        result = subprocess.run([str(python), str(Path(__file__).resolve()), '--worker', str(directory)],
            env=env, cwd=directory, text=True, encoding='utf-8', capture_output=True, timeout=45)
        record['worker_returncode'] = result.returncode
        (directory / 'worker_stdout.txt').write_text(result.stdout, encoding='utf-8')
        (directory / 'worker_stderr.txt').write_text(result.stderr, encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError('离线worker失败，详见worker_stderr.txt')
        report = json.loads((directory / 'offline_rules.json').read_text(encoding='utf-8'))
        audit = json.loads((directory / 'network_audit.json').read_text(encoding='utf-8'))
        checks = dict(report['checks'], all_reference_files_unchanged=verify_reference() == hashes,
            no_audited_network_attempts=audit == {'blocked_network_events': []})
        record.update(status='offline_rule_demo_checked_user_review_pending', report=report,
            network_audit=audit, checks=checks, artifacts_sha256={p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'record.json'})
        if not all(checks.values()):
            raise ValueError('本次离线检查未全部通过')
        print('案例：60题；layer1/layer2/layer3各20题。')
        print('示例问题：' + report['sample_case']['question'])
        print('四档：1=fail、2=pass、3=good、4=excellent。')
        print('整题通过：precision、recall、reasoning均≥3，且无幻觉否决；proactivity参与reward。')
        for item in report['numeric_examples_only']:
            out = item['official_parsed_result']
            print(f"  {item['name']} {list(item['scores'].values())} → reward={out['reward']:.6f}，passed={out['passed']}，veto={out['veto_applied']}")
        print('核对：' + json.dumps(checks, ensure_ascii=False))
        print('MEMORY_RUBRIC_OFFLINE_SAVED：仅案例读取和数字规则演示；尚未运行真实记忆系统或校准裁判。')
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        save(directory / 'record.json', record)


if __name__ == '__main__':
    main()

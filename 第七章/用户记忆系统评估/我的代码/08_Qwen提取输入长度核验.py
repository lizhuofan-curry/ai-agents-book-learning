"""只用本机Qwen分词文件与聊天模板核对提取输入；0次模型或网络请求。"""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import 本机记忆适配 as base
import RAG本机适配 as rag

ROOT, PROJECT = base.ROOT, base.PROJECT
PLAN = ROOT / '测试材料/Qwen提取输入长度方案.json'
CHINA = timezone(timedelta(hours=8))


def verify_inputs():
    plan = json.loads(PLAN.read_text())
    if (plan['model'] != 'qwen3.5:9b' or plan['num_ctx'] != 16384 or plan['reserved_output_tokens'] != 4096
            or plan['transport_headroom_tokens'] != 256 or plan['enable_thinking'] is not False
            or plan['maximum_real_requests'] != 0):
        raise ValueError('固定模型、长度口径或零请求预算变化')
    manifest_path = (Path.home() / plan['manifest_relative_to_home']).resolve()
    if not manifest_path.is_relative_to((Path.home() / '.ollama/models/manifests').resolve()):
        raise ValueError('本机模型清单路径越界')
    if base.sha(manifest_path) != plan['manifest_sha256']:
        raise ValueError('当前qwen3.5:9b清单变化，不静默替换分词文件')
    manifest = json.loads(manifest_path.read_text())
    cache = (PROJECT / plan['local_cache_path']).resolve()
    if not cache.is_relative_to((PROJECT / '.uv-cache/huggingface/local_tokenizers').resolve()):
        raise ValueError('分词缓存目录越界')
    assets = {}
    for asset in plan['tokenizer_assets']:
        if asset['name'] not in {'tokenizer.json', 'tokenizer_config.json'}:
            raise ValueError('只复用两份分词文件，不读权重张量')
        item = next(x for x in manifest['layers'] if x.get('name') == asset['name'])
        blob = Path.home() / '.ollama/models/blobs' / item['digest'].replace(':', '-')
        if (base.sha(blob) != asset['sha256'] or item['digest'] != 'sha256:' + asset['sha256']
                or blob.stat().st_size != asset['bytes']):
            raise ValueError('本机原分词资产与固定清单不符')
        cached = cache / asset['name']
        if not cached.is_file() or base.sha(cached) != asset['sha256']:
            raise ValueError('已准备的分词缓存缺失或变化，不自动下载/修复')
        assets[asset['name']] = asset['sha256']
    if set(assets) != {'tokenizer.json', 'tokenizer_config.json'}:
        raise ValueError('两份分词资产不完整')
    parent = (ROOT / '本地运行结果').resolve()
    pre = (parent / plan['preflight_run']).resolve()
    if pre.parent != parent or not pre.name.startswith('sixty_preflight_'):
        raise ValueError('预检来源越界')
    r = json.loads((pre / 'record.json').read_text())
    review = json.loads((pre / '核对记录.json').read_text())
    source07 = Path(__file__).with_name('07_六十案例离线预检.py')
    if (base.sha(pre / 'record.json') != plan['preflight_record_sha256']
            or base.sha(pre / '核对记录.json') != plan['preflight_review_sha256']
            or review['source_record_sha256'] != plan['preflight_record_sha256']
            or base.sha(source07) != plan['source_07_sha256']
            or r['script_sha256'] != plan['source_07_sha256']
            or not all(r['checks'].values()) or not all(review['checks'].values())):
        raise ValueError('固定本人60题预检或实现变化')
    return plan, cache, r, manifest, assets


def calibrate(tokenizer, plan):
    rows = []
    parent = (ROOT / '本地运行结果').resolve()
    for item in plan['calibration_runs']:
        path = (parent / item['run'] / 'record.json').resolve()
        if path.parent.parent != parent or base.sha(path) != item['record_sha256']:
            raise ValueError('已保存请求的校准来源变化或越界')
        record = json.loads(path.read_text())
        for call in record['model_calls']:
            request = call.get('native_request', call.get('request'))
            with_tools = bool(request.get('tools') or any(m.get('tool_calls') or m['role'] == 'tool' for m in request['messages']))
            ids = tokenizer.apply_chat_template(request['messages'], tools=request.get('tools'),
                add_generation_prompt=True, tokenize=True, enable_thinking=False)
            reported = call['response']['prompt_eval_count']
            rows.append({'source_run': item['run'], 'stage': call['stage'], 'contains_tools_or_tool_history': with_tools,
                'local_template_tokens': len(ids), 'reported_prompt_tokens': reported,
                'difference': len(ids) - reported})
    plain = [r for r in rows if not r['contains_tools_or_tool_history']]
    if len(plain) != 2 or any(r['difference'] != 0 for r in plain):
        raise ValueError('无工具模板与两条历史请求不一致，暂停60题提取长度判读')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    os.environ.update(PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1', MAX_RETRIES='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    attempts = []
    def audit(event, values):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            attempts.append(event)
            raise RuntimeError('Qwen长度核验禁止网络连接')
    sys.addaudithook(audit)
    plan, cache, preflight, manifest, assets = verify_inputs()
    print('第七章Qwen提取长度核验：本机已安装分词资产＋原60题提取提示。')
    print('0次生成、裁判、嵌入前向或网络请求；不下载、不加载模型权重、不启动批量。')
    if args.preview:
        print('QWEN_LENGTH_PREVIEW：模型清单、分词文件和本人预检来源已核对，尚未分词。')
        return 0
    directory = ROOT / '本地运行结果' / ('qwen_length_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(), 'plan': plan,
        'script_sha256': base.sha(Path(__file__)), 'input_plan_sha256': base.sha(PLAN),
        'installed_model_manifest_sha256': plan['manifest_sha256'], 'model_runner': manifest.get('runner'),
        'tokenizer_assets_sha256': assets, 'generation_requests': 0, 'judge_requests': 0,
        'embedding_forward_batches': 0, 'network_requests': 0,
        'scope': plan['scope']}
    save = lambda: base.dump(directory / 'record.json', record)
    save()
    print('记录路径：' + str(directory / 'record.json'), flush=True)
    try:
        rag.augment_dependencies()
        source07 = rag.load(Path(__file__).with_name('07_六十案例离线预检.py'), 'qwen_length_source07')
        refs = rag.load(Path(__file__).with_name('01_官方案例与评分规则离线检查.py'), 'qwen_length_ref_guard').verify_reference()
        official = base.load_official()
        cases = official.UserMemoryEvaluationFramework().test_suite.test_cases
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(cache), local_files_only=True, trust_remote_code=False)
        record['runtime_versions'] = {k: importlib.metadata.version(k) for k in ['transformers', 'tokenizers']}
        record['tokenizer_class'] = tokenizer.__class__.__name__
        record['calibration_against_saved_requests'] = calibrate(tokenizer, plan)
        prior = {r['case_id']: r for r in preflight['cases']}
        rows = []
        for case in sorted(cases, key=lambda c: (c.category, c.test_id)):
            prompt = source07.card_prompt(official, case)
            import hashlib
            if hashlib.sha256(prompt.encode()).hexdigest() != prior[case.test_id]['card_prompt_sha256']:
                raise ValueError('原始提取提示变化：' + case.test_id)
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                add_generation_prompt=True, tokenize=True, enable_thinking=False)
            required = len(ids) + plan['reserved_output_tokens'] + plan['transport_headroom_tokens']
            rows.append({'case_id': case.test_id, 'layer': case.category, 'local_card_prompt_tokens': len(ids),
                'output_reserve': 4096, 'engineering_headroom_not_measured_tokens': 256,
                'required_with_reserve_and_headroom': required, 'num_ctx': 16384,
                'fits_source_only_card_prompt': required <= 16384})
        record['cases'] = rows
        record['official_reference_hashes'] = refs
        record['summary'] = {'cases': len(rows), 'maximum_card_prompt_tokens': max(r['local_card_prompt_tokens'] for r in rows),
            'maximum_required_with_reserves': max(r['required_with_reserve_and_headroom'] for r in rows),
            'over_limit_cases': [r['case_id'] for r in rows if not r['fits_source_only_card_prompt']]}
        record['checks'] = {'sixty_source_prompts_reconstructed': len(rows) == len(prior) == 60,
            'two_plain_historical_requests_match_exactly': True,
            'installed_manifest_tokenizer_and_source_unchanged': verify_inputs()[0] == plan,
            'no_network_attempts': not attempts}
        record['readiness'] = {'source_only_card_prompts_fit': all(r['fits_source_only_card_prompt'] for r in rows),
            'tool_history_template_exact_equivalence_claimed': False,
            'generated_cards_and_runtime_retrieval_context_checked': False,
            'generic_batch_runner_prepared': False, 'batch_execution_started': False}
        if not all(record['checks'].values()):
            raise ValueError('分词或来源一致性检查未通过')
        record['status'] = 'qwen_source_prompt_lengths_saved_no_inference'
        print('60题提取提示最大本地token数：' + str(record['summary']['maximum_card_prompt_tokens']))
        print('加4096输出预留及256工程余量后最大：' + str(record['summary']['maximum_required_with_reserves']) + ' / 16384')
        print('超过此上限的案例：' + str(len(record['summary']['over_limit_cases'])))
        print('两条无工具历史请求计数匹配；含工具的历史请求仍有差异，不声称全链路模板精确等价。')
        print('QWEN_SOURCE_LENGTH_SAVED：仅提取输入长度核验；生成卡片/检索后的动态上下文及通用批量入口仍待处理。')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete_qwen_length_check', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        record['network_audit'] = {'blocked_events': attempts}
        save()


if __name__ == '__main__':
    raise SystemExit(main())

"""官方agent-cost-analysis四场景离线CLI＋本人RAG用量旁证；0次新模型请求。"""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
REFERENCE = ROOT/'官方资料'
OFFICIAL = REFERENCE/'upstream/chapter7/agent-cost-analysis'
PLAN = ROOT/'测试材料/离线输入方案.json'
KEYS = ['naive', 'kv', 'compress', 'both']
CHINA = timezone(timedelta(hours=8))
BOOTSTRAP = r'''
import json,runpy,sys
from pathlib import Path
reference,audit_path,program=sys.argv[1:4]
arguments=sys.argv[4:]
attempts=[]
def no_network(event,args):
    if event in {'socket.connect','socket.getaddrinfo','socket.sendto'}:
        attempts.append(event)
        raise RuntimeError('离线成本复算不允许网络连接')
sys.addaudithook(no_network)
sys.path.insert(0,reference)
sys.argv=[program,*arguments]
try:
    runpy.run_path(program,run_name='__main__')
finally:
    Path(audit_path).write_text(json.dumps({'blocked_network_events':attempts})+'\n',encoding='utf-8')
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def verify_reference():
    manifest_raw = (REFERENCE/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest['commit'] != 'dbc046eb896ac4e39aa19c7774c8bf49583b89a6' or len(manifest['files']) != 13:
        raise ValueError('固定参考提交或文件数量不符')
    hashes = {}
    for item in manifest['files']:
        path = (REFERENCE/'upstream'/item['path']).resolve()
        if not path.is_relative_to((REFERENCE/'upstream').resolve()):
            raise ValueError('参考文件路径越界')
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if sha(raw) != item['sha256'] or blob != item['git_blob_sha1'] or len(raw) != item['bytes']:
            raise ValueError('固定参考变化：'+item['path'])
        hashes[str(path)] = sha(raw)
    # config.py在导入时调用load_dotenv，需确认当前安装版本支持禁用开关。
    import dotenv
    if 'PYTHON_DOTENV_DISABLED' not in Path(dotenv.__file__).with_name('main.py').read_text(encoding='utf-8'):
        raise ValueError('当前dotenv版本不支持离线入口的禁用开关')
    return hashes, sha(manifest_raw)


def span_cost(span, prices):
    values = [span.get(key) for key in ['prompt_tokens', 'cached_tokens', 'completion_tokens']]
    if any(type(value) is not int or value < 0 for value in values) or values[1] > values[0]:
        raise ValueError('样例输入/缓存/输出计数不完整或不合法，不将未知值补0计费')
    prompt, cached, output = values
    per_m = Decimal(1_000_000)
    return ((Decimal(prompt-cached)*Decimal(str(prices['input']))
             + Decimal(cached)*Decimal(str(prices['cached']))
             + Decimal(output)*Decimal(str(prices['output'])))/per_m)


def verify_official_output(raw_sample, output, prices):
    if output['model'] != 'gpt-4o-mini' or output['pricing'] != prices:
        raise ValueError('官方输出的价格情景或模型标签不符')
    if [s['key'] for s in output['scenarios']] != KEYS:
        raise ValueError('没有按固定顺序复算全部四场景')
    sample = {s['key']: s for s in raw_sample['scenarios']}
    rows = []
    for scenario in output['scenarios']:
        source = sample[scenario['key']]
        original_spans, actual_spans = source['spans'], scenario['spans']
        if len(original_spans) != 8 or len(actual_spans) != 8:
            raise ValueError('每场景应有8条已保存span')
        costs = [span_cost(span, prices) for span in original_spans]
        for original, actual in zip(original_spans, actual_spans):
            if any(original[key] != actual[key] for key in ['step','tool','prompt_tokens','cached_tokens','completion_tokens','latency_s','tool_ctx_tokens']):
                raise ValueError('官方CLI改变了原用量或步骤顺序')
        total = float(sum(costs, Decimal(0)))
        prompt = sum(s['prompt_tokens'] for s in original_spans)
        cached = sum(s['cached_tokens'] for s in original_spans)
        completion = sum(s['completion_tokens'] for s in original_spans)
        comp = scenario['component_costs']
        expected = {'uncached_input_cost': float(Decimal(prompt-cached)*Decimal(str(prices['input']))/Decimal(1_000_000)),
                    'cached_input_cost': float(Decimal(cached)*Decimal(str(prices['cached']))/Decimal(1_000_000)),
                    'output_cost': float(Decimal(completion)*Decimal(str(prices['output']))/Decimal(1_000_000))}
        if not math.isclose(scenario['total_cost'], total, rel_tol=0, abs_tol=1e-12):
            raise ValueError('总成本与独立Decimal复算不一致')
        if any(not math.isclose(comp[key], value, rel_tol=0, abs_tol=1e-12) for key, value in expected.items()):
            raise ValueError('成本构成与独立复算不一致')
        if (comp['uncached_input_tokens'], comp['cached_input_tokens'], comp['output_tokens']) != (prompt-cached,cached,completion):
            raise ValueError('缓存输入被重复计入普通输入')
        sorted_costs = sorted(float(c) for c in costs)
        distribution = scenario['cost_distribution']
        if distribution['n'] != 8 or any(not math.isclose(distribution[key], sorted_costs[math.ceil(q*8)-1], rel_tol=0, abs_tol=1e-12)
                                               for key,q in [('p50',.5),('p95',.95),('p99',.99)]):
            raise ValueError('nearest-rank分位数复算不一致')
        worst_index = max(range(8), key=lambda i: costs[i])
        rows.append({'key': scenario['key'], 'prompt_tokens': prompt, 'cached_tokens': cached,
            'completion_tokens': completion, 'recomputed_scenario_usd': total,
            'most_expensive_step': original_spans[worst_index]['step'],
            'most_expensive_tool': original_spans[worst_index]['tool'],
            'component_costs': comp, 'per_step_cost_distribution': distribution})
    return rows


def load_personal(plan):
    paths = {}
    for kind in ['source_record', 'source_review']:
        path = (PROJECT/plan[kind]).resolve()
        if not path.is_relative_to((PROJECT/'第三章/个人Wiki混合RAG/本地运行结果').resolve()):
            raise ValueError('本人轨迹来源越界')
        raw = path.read_bytes()
        if sha(raw) != plan[kind+'_sha256']:
            raise ValueError('本人轨迹或既有审阅已变化，不自动换新批次')
        paths[str(path)] = sha(raw)
        if kind == 'source_record':
            record = json.loads(raw)
        else:
            review = json.loads(raw)
    rows = []
    if len(record['model_calls']) != 3 or len(review['question_reviews']) != 3:
        raise ValueError('须为本人三题独立请求及对应审阅')
    for call, verdict in zip(record['model_calls'], review['question_reviews']):
        if call['query_id'] != verdict['query_id'] or json.loads(call['raw_http_body']) != call['response']:
            raise ValueError('本人轨迹响应或题号不一致')
        pin, out = call.get('prompt_tokens_reported'), call.get('output_tokens_reported')
        duration = call['response'].get('total_duration')
        if any(type(v) is not int or v < 0 for v in [pin,out]):
            raise ValueError('本人token计数未知，不补0')
        seconds = duration/1_000_000_000 if type(duration) is int and duration >= 0 else None
        rows.append({'query_id': call['query_id'], 'reported_prompt_tokens': pin,
            'reported_output_tokens': out, 'reported_server_total_duration_s': seconds,
            'model_status': call['answer']['status'], 'review_outcome': verdict['review_outcome'],
            'cached_input_tokens': None, 'local_billed_money': None})
    return {'rows': rows, 'total_reported_prompt_tokens': sum(r['reported_prompt_tokens'] for r in rows),
        'total_reported_output_tokens': sum(r['reported_output_tokens'] for r in rows),
        'server_duration_sum_s': sum(r['reported_server_total_duration_s'] for r in rows) if all(r['reported_server_total_duration_s'] is not None for r in rows) else None,
        'cache_split': 'unknown_not_assumed_zero', 'billed_money': 'not_measured',
        'scope': 'Only three prior answer-generation calls, not all RAG embedding/retrieval/human-review cost. Server duration is not full task wall time; no energy or current cloud pricing inferred.'}, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--official-only', action='store_true', help='只复算官方样例，不读取本机私有RAG轨迹')
    for name in ['input','cached','output']:
        parser.add_argument('--price-'+name, type=float)
    args = parser.parse_args()
    hashes, manifest_sha = verify_reference()
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    prices = dict(plan['historical_sample_price_per_million_usd'])
    for name in prices:
        value = getattr(args, 'price_'+name)
        if value is not None:
            prices[name] = value
    if any(not math.isfinite(value) or value < 0 for value in prices.values()):
        raise ValueError('情景价格须为有限非负数')
    personal, personal_hashes = (None, {}) if args.official_only else load_personal(plan)
    hashes.update(personal_hashes)
    print('第七章第一步：官方四场景离线成本复算；0次新模型、工具业务或网络请求。')
    print('作者样例和本人RAG记录分开；教材价格仅为计算假设，不是实时价格或本机账单。')
    if args.preview:
        print('COST_OFFLINE_PREVIEW_ONLY：13份参考和所选输入SHA通过；不运行官方CLI，无新结果目录。')
        return 0
    folder = ROOT/'本地运行结果'/('cost_offline_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    path = folder/'record.json'
    output, audit = folder/'official_output.json', folder/'network_audit.json'
    arguments = ['--offline','--scenario','all','--model','gpt-4o-mini',
        '--price-input',str(prices['input']),'--price-cached',str(prices['cached']),
        '--price-output',str(prices['output']),'--output',str(output)]
    command = [sys.executable,'-B','-c',BOOTSTRAP,str(OFFICIAL),str(audit),str(OFFICIAL/'demo.py'),*arguments]
    record = {'status':'running','created_at':datetime.now(CHINA).isoformat(),
        'script_sha256':sha(Path(__file__).read_bytes()),'manifest_sha256':manifest_sha,
        'input_plan_sha256':sha(PLAN.read_bytes()),'source_hashes':hashes,
        'official_cli_arguments':arguments,'offline_bootstrap':BOOTSTRAP,
        'sample_provenance':'pinned upstream sample_trace; author describes recorded usage; not newly measured calls or independently audited upstream run',
        'pricing_per_million_usd':prices,'pricing_scope':'historical or overridden scenario assumption, not current vendor prices or local billed cost',
        'personal_generation_usage':personal,'new_model_requests':0,'new_embedding_forward_batches':0,
        'network_requests':0,'dotenv_loading_disabled':True,'provider_key_environment_variables_removed':True,
        'scope':'Official offline CLI replay plus independent arithmetic; no new cache/compression/model-quality experiment or cloud billing.'}
    save_json(path, record)
    print('本地记录路径：'+str(path), flush=True)
    try:
        env = dict(os.environ, PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
        for key in ['OPENAI_API_KEY','OPENROUTER_API_KEY','ANTHROPIC_API_KEY','COST_DEMO_MODEL']:
            env.pop(key, None)
        child = subprocess.run(command, cwd=OFFICIAL, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, encoding='utf-8', timeout=60)
        record['official_cli'] = {'returncode':child.returncode,'stdout':child.stdout,'stderr':child.stderr}
        (folder/'official_stdout.txt').write_text(child.stdout, encoding='utf-8')
        if child.stderr:
            (folder/'official_stderr.txt').write_text(child.stderr, encoding='utf-8')
        print(child.stdout, end='', flush=True)
        if child.returncode:
            raise RuntimeError('官方离线CLI退出非0；stdout/stderr已保存，不转在线模式')
        result = json.loads(output.read_text(encoding='utf-8'))
        sample = json.loads((OFFICIAL/'sample_trace.json').read_text(encoding='utf-8'))
        rows = verify_official_output(sample, result, prices)
        record['official_recomputed_scenarios'] = rows
        record['official_output_sha256'] = sha(output.read_bytes())
        network_audit = json.loads(audit.read_text(encoding='utf-8'))
        record['network_audit'] = network_audit
        if personal:
            print('\n本人既有3次RAG生成用量（不是上面的退款样例）：')
            for row in personal['rows']:
                seconds = '-' if row['reported_server_total_duration_s'] is None else f'{row["reported_server_total_duration_s"]:.3f}s'
                print(f'  {row["query_id"]} 输入={row["reported_prompt_tokens"]} 输出={row["reported_output_tokens"]} 服务端耗时={seconds} 原状态={row["model_status"]}')
            print(f'  合计输入={personal["total_reported_prompt_tokens"]}，输出={personal["total_reported_output_tokens"]}；缓存拆分未知，本机金额未测。')
            print('  token/耗时不是答案正确性的证明；过度拒答、引用问题和合理拒答仍按既有审阅保留。')
        record['checks'] = {
            'official_offline_cli_exit_zero':child.returncode == 0,
            'four_scenarios_eight_spans_each_recomputed':len(rows) == 4,
            'pricing_components_and_quantiles_independently_verified':True,
            'no_audited_network_attempts':network_audit['blocked_network_events'] == [],
            'all_reference_and_personal_sources_unchanged':all(sha(Path(name).read_bytes()) == expected for name, expected in hashes.items()),
        }
        passed = all(record['checks'].values())
        record['status'] = 'official_offline_cost_replay_checked_review_pending' if passed else 'failed_checks'
        print('\n复算检查：'+json.dumps(record['checks'], ensure_ascii=False))
        print('COST_OFFLINE_SAVED：仅既有轨迹与价格假设复算；不代表本人新测32次请求或得到Ollama账单。')
        return 0 if passed else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete',error={'type':type(exc).__name__,'message':str(exc)})
        print('本步未完成，不重试：'+str(exc))
        return 1
    finally:
        save_json(path, record)
        print('本地记录路径：'+str(path), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

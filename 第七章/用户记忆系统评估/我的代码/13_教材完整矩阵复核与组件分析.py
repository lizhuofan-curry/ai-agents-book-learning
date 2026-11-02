"""复核固定作者1440条结果并重算组件统计；0次模型/API请求，不冒充个人重跑。"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
MANIFEST=ROOT/'测试材料/教材7-11完整结果参考清单.json'
CHINA=timezone(timedelta(hours=8))
METRICS=['retrieval_hit_at_5','retrieval_recall_at_5','retrieval_mrr',
    'fixed_query_hit_at_5','fixed_query_recall_at_5','fixed_query_mrr','reward']


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def group(rows,keys):
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for identity,values in sorted(groups.items()):
        result.append({**dict(zip(keys,identity)),'n':len(values),'successes':sum(x['success'] for x in values),
            'success_rate':statistics.fmean(x['success'] for x in values),
            'average_reward':statistics.fmean(x['reward'] for x in values),
            'fixed_query_recall_at_5':statistics.fmean(x['fixed_query_recall_at_5'] for x in values),
            'retrieval_recall_at_5':statistics.fmean(x['retrieval_recall_at_5'] for x in values),
            'latency_ms':statistics.fmean(x['latency_ms'] for x in values),
            'hallucination_vetoes':sum(x['hallucination_veto'] for x in values)})
    return result


def main():
    os.environ['PYTHON_DOTENV_DISABLED']='1'
    for key in list(os.environ):
        if key.endswith('_API_KEY'): os.environ.pop(key)
    attempts=[]
    def audit(event,args):
        if event in {'socket.connect','socket.getaddrinfo','socket.sendto'}:
            attempts.append(event);raise RuntimeError('作者数据复核禁止网络请求')
    sys.addaudithook(audit)
    manifest=json.loads(MANIFEST.read_text());source=(ROOT/manifest['directory']).resolve()
    if source.parent!=(ROOT/'本地运行结果').resolve(): raise ValueError('参考目录越界')
    for item in manifest['files']:
        path=source/item['name']
        if path.parent!=source or sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:
            raise ValueError('固定作者结果或核验源码变化')
    data=json.loads((source/'full_7_11_60_case_matrix.json').read_text());rows=data['records']
    plan=json.loads((ROOT/'测试材料/完整7-11矩阵与3元预算方案.json').read_text())
    expected_cases={x['case_id'] for x in plan['cases']};cfg=plan['matrix']
    combos=set(itertools.product(cfg['embeddings'],cfg['rerankers'],cfg['main_models']))
    expected={(cid,*cell) for cid in expected_cases for cell in combos}
    actual=[(x['test_id'],x['embedding'],x['reranker'],x['main_model']) for x in rows]
    checks={'pinned_files_verified':True,'exact_60_by_24_unique_cells':len(rows)==len(set(actual))==1440 and set(actual)==expected,
        'three_layers_20_cases_each':dict(Counter(cid.split('_')[0] for cid in expected_cases))=={'layer1':20,'layer2':20,'layer3':20},
        'all_1440_records_ok_no_errors':all(x['status']=='ok' and not x.get('error') and x['experiment']=='7-11' and x['system']=='rag' for x in rows),
        'source_api_labels_preserved_not_new_execution':all(x['evidence_mode']=='real_api' for x in rows),
        'numeric_metrics_finite_in_unit_interval':all(type(x.get(k)) in [int,float] and math.isfinite(x[k]) and 0<=x[k]<=1 for x in rows for k in METRICS),
        'zero_unpriced_recorded_usage':all(x[k]==0 for x in rows for k in ['unpriced_tokens','unpriced_requests','fixed_query_unpriced_tokens','fixed_query_unpriced_requests']),
        'index_and_fixed_query_accounting_present':all(type(x.get(k)) in [int,float] and math.isfinite(x[k]) and x[k]>=0 for x in rows for k in ['embedding_index_latency_ms','fixed_query_retrieval_latency_ms']),
        'all_rubric_scores_and_veto_reward_consistent':True}
    for row in rows:
        scores=[row['rubric_dimensions'][k] for k in ['precision','recall','reasoning','proactivity']]
        valid=all(type(v) is int and 1<=v<=4 for v in scores)
        reward=0 if row['hallucination_veto'] else statistics.fmean((v-1)/3 for v in scores)
        success=not row['hallucination_veto'] and min(scores[:3])>=3
        checks['all_rubric_scores_and_veto_reward_consistent'] &= valid and type(row['success']) is bool and row['success']==success and abs(row['reward']-reward)<=1e-6
    cells=group(rows,['embedding','reranker','main_model']);by_layer=group(rows,['layer','embedding','reranker','main_model'])
    original_cells={(x['embedding'],x['reranker'],x['main_model']):x for x in data['interaction_analysis']['factorial_cells']}
    checks['all_24_cell_aggregates_recomputed']=all(all(abs(cell[k]-original_cells[(cell['embedding'],cell['reranker'],cell['main_model'])][k])<1e-10 for k in ['success_rate','average_reward','fixed_query_recall_at_5']) for cell in cells)
    original_layer={(x['layer'],x['embedding'],x['reranker'],x['main_model']):x for x in data['summary']}
    checks['all_72_layer_aggregates_recomputed']=all(all(abs(cell[k]-original_layer[(cell['layer'],cell['embedding'],cell['reranker'],cell['main_model'])][k])<1e-10 for k in ['success_rate','average_reward','fixed_query_recall_at_5']) for cell in by_layer)
    costs=defaultdict(Decimal)
    for row in rows:
        for field in ['cost_by_currency','fixed_query_retrieval_cost_by_currency']:
            for currency,value in row[field].items():
                if currency not in ['CNY','USD'] or not math.isfinite(value) or value<0: raise ValueError('币种或成本数值异常')
                costs[currency]+=Decimal(str(value))
    checks['native_currency_attributed_cost_totals_match']=all(abs(float(v)-data['pricing_coverage']['total_cost_by_currency'][k])<1e-9 for k,v in costs.items())
    readiness=json.loads((source/'full_matrix_backend_readiness_20260731.json').read_text())
    checks['nine_author_readiness_slots_not_personal_probes']=len(readiness['probes'])==9 and all(x['status']=='ok' for x in readiness['probes'])
    # 固定问题基线在两个主模型间共享；逐Agent改写查询不作相同查询对照。
    fixed=defaultdict(list)
    for row in rows: fixed[(row['test_id'],row['embedding'],row['reranker'])].append(row)
    checks['fixed_query_metrics_equal_between_two_main_models']=all(len(v)==2 and all(v[0][k]==v[1][k] for k in ['fixed_query_hit_at_5','fixed_query_recall_at_5','fixed_query_mrr']) for v in fixed.values())
    baseline={(x['embedding'],x['main_model']):x for x in cells if x['reranker']=='none'}
    deltas=[]
    for cell in cells:
        if cell['reranker']=='none': continue
        ref=baseline[(cell['embedding'],cell['main_model'])]
        deltas.append({'embedding':cell['embedding'],'reranker':cell['reranker'],'main_model':cell['main_model'],
            'success_rate_delta':cell['success_rate']-ref['success_rate'],
            'fixed_query_recall_delta':cell['fixed_query_recall_at_5']-ref['fixed_query_recall_at_5'],
            'latency_ms_delta':cell['latency_ms']-ref['latency_ms']})
    checks['sixteen_conditional_rerank_deltas']=len(deltas)==16
    checks['no_network_attempts']=not attempts
    if not all(checks.values()): raise ValueError('独立矩阵检查失败：'+json.dumps(checks))
    directory=ROOT/'本地运行结果'/('official_matrix_review_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(exist_ok=False)
    # 已逐行读取并锁定SHA的原核验器，只读输入、使用标准库；保存原始stdout，包括其USD标签局限。
    result=subprocess.run([sys.executable,'-I',str(source/'verify_full_matrix_20260731.py'),str(source/'full_7_11_60_case_matrix.json')],
        capture_output=True,text=True,timeout=30,env={k:v for k,v in os.environ.items() if not k.endswith('_API_KEY')})
    (directory/'official_verifier_stdout.txt').write_text(result.stdout,encoding='utf-8')
    (directory/'official_verifier_stderr.txt').write_text(result.stderr,encoding='utf-8')
    checks['original_verifier_passed']=result.returncode==0 and 'ALL CHECKS PASSED' in result.stdout
    warnings=[{'issue':'raw_generation_and_judge_calls_not_rerun','detail':'Author real_api tags do not independently establish that this user issued1440 new model requests.'},
        {'issue':'retrieval_gold_ids_not_present_per_record','detail':'Metric ranges and grouped means verified; per-record recall/MRR cannot be independently rebuilt without original relevant-ID sets.'},
        {'issue':'reference_rubric_source_conflicts','detail':'Known travel/vehicle source conflicts remain; rubric arithmetic verification does not calibrate its factual judgment.'},
        {'issue':'attributed_cost_not_invoice','detail':'Totals include per-cell allocated shared construction/fixed-query costs and exclude gold selection/task judging per author scope_note; no currency conversion or claim of exact campaign bill.'},
        {'issue':'official_verifier_usd_label_incomplete','detail':'Its total main-model cost print only sums cost_usd, omitting CNY; use both native-currency totals.'}]
    extra_steps=[{'case_id':x['test_id'],'embedding':x['embedding'],'reranker':x['reranker'],'main_model':x['main_model'],'steps':x['steps'],'tool_calls':x['tool_calls']} for x in rows if x['steps']>cfg['max_search_rounds']+1]
    if extra_steps: warnings.append({'issue':'observed_steps_exceed_simple_3_round_bound','records':extra_steps,'detail':'max_search_rounds/tool_choice is not a hard guard. Do not claim all author trajectories obey proposed one-tool/4-call cap.'})
    report={'status':'author_matrix_review_complete_with_stated_limits','evidence_owner':'textbook_authors',
        'source_commit':manifest['source_commit'],'manifest_sha256':sha(MANIFEST),'script_sha256':sha(Path(__file__)),
        'source_matrix_sha256':sha(source/'full_7_11_60_case_matrix.json'),'checks':checks,'warnings':warnings,
        'author_records_verified':1440,'author_cases_verified':60,'author_cells_verified':24,
        'source_generation_requests_this_run':0,'judge_requests_this_run':0,'embedding_forward_batches_this_run':0,
        'new_paid_api_requests':0,'personal_full_matrix_trajectories_executed':0,
        'user_budget_cny':'3.00','credential_presence_only_snapshot':manifest['personal_environment_snapshot'],
        'factorial_cells':cells,'layer_cells':by_layer,'reranker_deltas':deltas,
        'marginals':{k:group(rows,[k]) for k in ['embedding','reranker','main_model']},
        'attributed_historical_cost_by_currency':{k:str(v) for k,v in costs.items()},
        'statistics_scope':'Descriptive within fixed synthetic60-case author dataset; repeated cases across cells not independent1440 samples. No new significance or universal component ranking claim.',
        'network_audit':{'blocked_events':attempts}}
    dump(directory/'record.json',report)
    for name,values in [('组件24单元.csv',cells),('分层72单元.csv',by_layer),('重排边际增益16组.csv',deltas)]:
        with (directory/name).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    print('7-11教材完整结果复核：60题×24配置＝1440条作者记录，独立检查'+str(len(checks))+'项。')
    print('原核验器：'+('ALL CHECKS PASSED' if checks['original_verifier_passed'] else 'FAILED'))
    print('已重算24个组件组合、72个分层组合、16组重排边际增益，导出CSV。')
    print('本次新增模型/API/嵌入请求：0；个人完整矩阵重跑：0条。')
    print('AUTHOR_MATRIX_REVIEW_SAVED：教材证据复核与分析完成；原始裁判/Gold局限及超轮数记录已保留，不冒充个人完整复现。')
    print('记录路径：'+str(directory/'record.json'))
    return 0 if all(checks.values()) else 1


if __name__=='__main__': raise SystemExit(main())

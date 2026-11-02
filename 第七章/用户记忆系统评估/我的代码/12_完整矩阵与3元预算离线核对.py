"""只核对完整7-11计划与3元预算；不探测后端、不请求模型、不读凭据。"""
import argparse
from datetime import datetime
from decimal import Decimal
import itertools
import json
import os
from pathlib import Path
import sys
import unittest

import 本机记忆适配 as base
import RAG本机适配 as rag
import 批量JSONCards适配 as journal

PLAN=base.ROOT/'测试材料/完整7-11矩阵与3元预算方案.json'
PRICES=base.ROOT/'测试材料/7-11公开价格核对快照.json'


def workload(cases):
    """所有单元走完、单工具、最多3轮、零重试的提案；不等于原Runner天然硬上限。"""
    cells=cases*4*3*2
    rows={
        'retrieval_gold_chat':(cases,cases),
        'document_embedding_dispatches':(cases*4,cases*4),
        'fixed_question_embedding_dispatches':(cases*4*3,cases*4*3),
        'trajectory_query_embedding_dispatches':(cells,cells*3),
        'main_chat':(cells*2,cells*4),
        'task_judge_chat':(cells,cells),
        'fixed_question_llm_rerank_chat':(cases*4*2,cases*4*2),
        'trajectory_llm_rerank_chat':(cases*4*2*2,cases*4*2*2*3)}
    return {'cases':cases,'trajectory_cells':cells,
        'logical_dispatches_successful_completion_scenario':{k:{'minimum':a,'maximum':b} for k,(a,b) in rows.items()},
        'total_minimum_scenario':sum(a for a,b in rows.values()),
        'total_maximum_scenario':sum(b for a,b in rows.values()),
        'scope':'Proposed bounded single-dispatch adapters, one tool per turn, max3 search rounds total, one doc embedding request per case/backend. Excludes readiness probes; failures can stop early. Provider batching/retries must not silently multiply dispatches.'}


def budget_gate(estimate,budget,prices_verified=False,fx_verified=False,runner_guard=False,backends_ready=False,calibrated=False):
    reasons=[]
    if not isinstance(estimate,Decimal) or not estimate.is_finite() or estimate<0:
        reasons.append('complete_conservative_cost_unknown')
    elif estimate>budget:
        reasons.append('conservative_cost_exceeds_budget')
    if not prices_verified: reasons.append('exact_model_route_prices_unverified')
    if not fx_verified: reasons.append('usd_cny_conversion_and_fees_unverified')
    if not runner_guard: reasons.append('bounded_money_and_retry_guard_not_implemented')
    if not backends_ready: reasons.append('official_matrix_backends_not_live_verified')
    if not calibrated: reasons.append('rubric_and_retrieval_gold_calibration_pending')
    return {'budget_assurance_met':not reasons,'blockers':reasons}


def historical_chat_cap(cases,config,proposal):
    # 用冻结配置的旧单价展示“按完整token预留”的场景，绝不是最低成本/当前报价。
    max_in=Decimal(proposal['max_chat_input_tokens_per_dispatch'])
    limits=proposal['max_output_tokens']
    caps={'kimi':{'input':cases*(48+24+28+1)*max_in,
        'output':cases*(48*limits['main']+24*limits['task_judge']+28*limits['rerank']+limits['gold'])},
        'doubao':{'input':cases*(48+28)*max_in,
        'output':cases*(48*limits['main']+28*limits['rerank'])}}
    costs={}
    for name,tokens in caps.items():
        price=config['chat_models'][name]['pricing']
        if price['currency']!='CNY': raise ValueError('历史示例不隐式换汇')
        costs[name]=str((tokens['input']*Decimal(str(price['input_per_million']))
            +Decimal(tokens['output'])*Decimal(str(price['output_per_million'])))/Decimal(1000000))
    return {'basis':'Frozen July2026 config list prices; all allowed input/output caps fully used, no cache savings. Not a current quote, expected bill, actual spend or minimum required cost.',
        'cny_chat_only_by_model':costs,'cny_chat_only_total':str(sum(Decimal(x) for x in costs.values())),
        'embedding_usd_cost':None,'fx':None,'readiness_or_other_fees_included':False}


def verify_inputs():
    plan=journal.strict_json(PLAN.read_text());prices=journal.strict_json(PRICES.read_text())
    if (plan['protocol_version']!='full-711-budget-draft-v1' or plan['user_api_budget_cny']!='3.00'
            or plan['paid_execution_enabled'] is not False or plan['maximum_model_api_or_embedding_calls_this_script']!=0):
        raise ValueError('本步固定完整矩阵计划、总预算3元及零付费执行')
    guard=rag.load(Path(__file__).with_name('01_官方案例与评分规则离线检查.py'),'matrix_budget_reference_guard')
    refs=guard.verify_reference()
    upstream=base.ROOT/'官方资料/upstream/chapter7/user-memory-system-evaluation'
    for filename,key in [('default_config.yaml','original_config_sha256'),('experiment.py','original_experiment_sha256')]:
        if base.sha(upstream/filename)!=plan[key]: raise ValueError('固定官方配置或源码变化')
    parent=(base.ROOT/'本地运行结果').resolve();directory=(parent/plan['hybrid_run']).resolve()
    if directory.parent!=parent or not directory.name.startswith('hybrid_cross_layer_'):
        raise ValueError('Hybrid来源越界')
    for filename,key in [('record.json','hybrid_record_sha256'),('核对记录.json','hybrid_execution_review_sha256'),('来源审阅.json','hybrid_semantic_review_sha256')]:
        if base.sha(directory/filename)!=plan[key]: raise ValueError('本人Hybrid或审阅变化')
    for row in plan['cases']:
        path=(base.ROOT/'官方资料/upstream'/row['source_path']).resolve()
        if not path.is_relative_to((base.ROOT/'官方资料/upstream').resolve()) or base.sha(path)!=row['source_sha256']:
            raise ValueError('案例来源变化或越界')
    return plan,prices,refs,upstream


def self_test():
    class Tests(unittest.TestCase):
        def test_counts_and_fixed_work_shared_across_main_models(self):
            one=workload(1);self.assertEqual((one['trajectory_cells'],one['total_minimum_scenario'],one['total_maximum_scenario']),(24,137,265))
            all_cases=workload(60);self.assertEqual((all_cases['trajectory_cells'],all_cases['total_minimum_scenario'],all_cases['total_maximum_scenario']),(1440,8220,15900))
            self.assertEqual(all_cases['logical_dispatches_successful_completion_scenario']['document_embedding_dispatches']['maximum'],240)
        def test_unknown_or_invalid_cost_never_passes(self):
            for x in [None,3.0,Decimal('NaN'),Decimal('Infinity'),Decimal('-1')]:
                self.assertFalse(budget_gate(x,Decimal('3'),True,True,True,True,True)['budget_assurance_met'])
        def test_budget_overrun_and_missing_guard_block(self):
            self.assertFalse(budget_gate(Decimal('3.01'),Decimal('3'),True,True,True,True,True)['budget_assurance_met'])
            self.assertFalse(budget_gate(Decimal('1'),Decimal('3'),True,True,False,True,True)['budget_assurance_met'])
        def test_under_budget_still_requires_fx_prices_readiness_calibration(self):
            for flags in [(False,True,True,True,True),(True,False,True,True,True),(True,True,True,False,True),(True,True,True,True,False)]:
                self.assertFalse(budget_gate(Decimal('1'),Decimal('3'),*flags)['budget_assurance_met'])
            self.assertTrue(budget_gate(Decimal('3'),Decimal('3'),True,True,True,True,True)['budget_assurance_met'])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    os.environ.update(PYTHON_DOTENV_DISABLED='1',PYTHONDONTWRITEBYTECODE='1')
    for name in list(os.environ):
        if name.endswith('_API_KEY'): os.environ.pop(name)
    attempts=[]
    def audit(event,values):
        if event in {'socket.connect','socket.getaddrinfo','socket.sendto'}:
            attempts.append(event);raise base.StopPipeline('矩阵预算核对禁止全部网络连接')
    sys.addaudithook(audit)
    if args.self_test: return self_test()
    rag.augment_dependencies()
    plan,prices,refs,upstream=verify_inputs()
    import yaml
    config=yaml.safe_load((upstream/'default_config.yaml').read_text());official=base.load_official()
    cases=official.UserMemoryEvaluationFramework().test_suite.test_cases
    actual_ids=sorted(c.test_id for c in cases);planned_ids=[c['case_id'] for c in plan['cases']]
    matrix=plan['matrix'];combinations=list(itertools.product(matrix['embeddings'],matrix['rerankers'],matrix['main_models']))
    keys=[f'{cid}|{e}|{rr}|{m}' for cid in planned_ids for e,rr,m in combinations]
    stages=plan['stage_case_ids'];stage_ids=[cid for ids in stages.values() for cid in ids]
    known_readiness=sorted(official.required_readiness_components(config))
    full=workload(len(cases));proposal=plan['bounded_execution_proposal_not_implemented']
    expected_proposal={'max_search_rounds_total':3,'max_tool_calls_per_response':1,'max_main_calls_per_cell':4,
        'sdk_retries':0,'transport_retries':0,'json_repair_retries':0,'max_gold_calls_per_case':1,
        'max_task_judge_calls_per_cell':1,'max_reranker_calls_per_search':1,
        'max_chat_input_tokens_per_dispatch':12288,'max_output_tokens':{'main':1024,'rerank':2048,'task_judge':2048,'gold':4096},
        'cache_savings_assumed':False,'unknown_price_fx_or_usage_blocks_dispatch':True,'failed_or_unknown_dispatch_consumes_budget':True}
    checks={'sixty_distinct_official_ids':len(actual_ids)==len(set(actual_ids))==60 and sorted(planned_ids)==actual_ids,
        'exact_original_matrix_configuration':matrix==config['experiment_7_11'] and [len(matrix[k]) for k in ['embeddings','rerankers','main_models']]==[4,3,2],
        'all_1440_planned_cell_keys_unique':len(keys)==len(set(keys))==1440,
        'stage_partition_3_12_45_no_overlap':[len(v) for v in stages.values()]==[3,12,45] and sorted(stage_ids)==actual_ids and len(set(stage_ids))==60,
        'bounded_proposal_fixed_and_not_original_default_retry_claim':proposal==expected_proposal,
        'unknown_total_money_preserved':plan['monetary_total_estimate_cny'] is None and prices['is_complete_current_pricing'] is False,
        'nine_component_slots_defined_without_api_probe':len(known_readiness)==9,
        'no_network_attempts':not attempts}
    if not all(checks.values()): raise ValueError('矩阵计划完整性核对失败：'+json.dumps(checks))
    record={'status':'offline_full_matrix_plan_saved_paid_execution_not_authorized_or_ready',
        'created_at':datetime.now(journal.CHINA).isoformat(),'script_sha256':base.sha(Path(__file__)),
        'plan_sha256':base.sha(PLAN),'price_snapshot_sha256':base.sha(PRICES),'official_reference_hashes':refs,
        'planned_combinations':[{'embedding':e,'reranker':rr,'main_model':m} for e,rr,m in combinations],
        'planned_cell_key_sha256':journal.digest(keys),'planned_trajectories':1440,'executed_711_trajectories_this_run':0,
        'budget_cny':'3.00','full_workload_proposal':full,
        'stages_planned_not_executed':{name:workload(len(ids)) for name,ids in stages.items()},
        'readiness_slots':[{'component':c,'name':n,'status':'not_probed_by_this_run'} for c,n in known_readiness],
        'proposed_separate_readiness_dispatch_cap_not_authorization':8,
        'historical_chat_token_cap_illustration':historical_chat_cap(60,config,proposal),
        'complete_conservative_current_cost_cny':None,
        'budget_gate':budget_gate(None,Decimal('3.00')),
        'paid_execution_enabled':False,'bounded_online_runner_prepared':False,'credentials_inspected':False,
        'generation_requests':0,'judge_requests':0,'embedding_forward_batches':0,'api_readiness_probes':0,
        'network_audit':{'blocked_events':attempts},'checks':checks}
    directory=base.ROOT/'本地运行结果'/('full_matrix_budget_'+datetime.now(journal.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(exist_ok=False);base.dump(directory/'record.json',record)
    print('7-11完整矩阵预算：4嵌入×3重排×2主模型×60题＝1440条计划轨迹。')
    print('0次生成、裁判、嵌入前向、API探针或网络请求；不读取凭据，不启动任何批次。')
    print('用户API总预算：3.00元人民币；完整现价/汇率/费用预留尚未核准，禁止付费执行。')
    print('零重试/单工具/最多3轮检索提案：全矩阵成功走完约8220—15900次逻辑分发；不是1440次API。')
    print('这是待实现受限入口的情景预算，不是现有官方SDK/重试行为的硬上限。')
    print('拟定3/12/45题分期＝72/288/1080条轨迹，全部仍为未执行计划；先完成预算与后端/评分校准准入。')
    print('FULL_MATRIX_BUDGET_SAVED：完整计划已保存，3元预算下未获执行放行，不算正式7-11结果。')
    print('记录路径：'+str(directory/'record.json'))
    return 0


if __name__=='__main__': raise SystemExit(main())

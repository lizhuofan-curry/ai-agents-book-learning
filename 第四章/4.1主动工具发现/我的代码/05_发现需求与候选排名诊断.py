"""事后离线诊断：复算已保存need，并用人工需求检查下载能力候选。"""
import argparse
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = Path(__file__).with_name('02_全量与按需工具选择对照.py')
spec = importlib.util.spec_from_file_location('tool_choice_index', SPEC_PATH)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
SOURCE = ROOT/'运行结果/tool_choice_compare_20261025_203942_696397/record.json'
SOURCE_SHA = 'ebc3b6313fc09c60d45654736640482f7e04d67b2483048edf98363b7778bcec'
INDEX_SCRIPT_SHA = '621d06beda6a3dd950a90ef7f34b8dcec80ac82b014f99e3ddac3fc46ea75983'
MANUAL_NEED = '下载论文PDF文件'
TARGET = 'download_file'


def rank_all(index, need, excluded):
    if not isinstance(need, str) or not need.strip() or len(need) > 500:
        raise ValueError('需求长度错误')
    query = index.vector(compare.terms(need))
    ranking = []
    for name, vector in index.vectors.items():
        if name in excluded:
            continue
        score = sum(value*vector.get(term, 0) for term, value in query.items())
        if score > 0:
            ranking.append({'name': name, 'score': score})
    ranking.sort(key=lambda item: (-item['score'], item['name']))
    if ranking[:compare.TOP_K] != index.search(need, excluded):
        raise ValueError('诊断排名与原检索器不一致')
    return ranking


def view(index, need, excluded, origin, stage):
    ranking = rank_all(index, need, excluded)
    target = next((i+1 for i, item in enumerate(ranking) if item['name'] == TARGET), None)
    score = next((item['score'] for item in ranking if item['name'] == TARGET), 0.0)
    return {'need': need, 'origin': origin, 'stage': stage,
            'excluded_available_names': sorted(excluded), 'top5': ranking[:5], 'full_positive_ranking': ranking,
            'diagnostic_target': TARGET, 'target_rank_among_positive_scores': target, 'target_score': score,
            'target_in_top_k': {str(k): target is not None and target <= k for k in (5, 10, 20)},
            'note': 'known-target post-hoc diagnostic; no model decision or new execution'}


def prepare():
    if compare.sdk.sha(SOURCE) != SOURCE_SHA or compare.sdk.sha(SPEC_PATH) != INDEX_SCRIPT_SHA:
        raise ValueError('冻结原始对照或检索代码变化')
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    full, _, manifest = compare.prepare()
    if manifest != source['manifest']:
        raise ValueError('工具来源清单不一致')
    index = compare.ToolIndex(full)
    arm = source['arms']['discovery']
    available = set(arm['initial_available_names'])
    queries = []
    first_excluded = set(available)
    for action in arm['actions']:
        if action['name'] != 'discover_tools':
            continue
        need = action['arguments']['need']
        replay = index.search(need, excluded=available)
        saved = action['ranking']
        if [r['name'] for r in replay] != [r['name'] for r in saved] or any(
                not math.isclose(a['score'], b['score'], rel_tol=0, abs_tol=1e-12) for a, b in zip(replay, saved)):
            raise ValueError('已保存模型需求的候选排名复算不一致')
        queries.append(view(index, need, available, 'saved_model_need', 'actual_discovery_call'))
        candidates = action['result']['candidates']
        if [t['function']['name'] for t in candidates] != [r['name'] for r in replay]:
            raise ValueError('保存候选与检索结果不一致')
        available.update(t['function']['name'] for t in candidates)
    if len(queries) != 1:
        raise ValueError('本次诊断预期仅有一次真实发现动作')
    # 以下需求均非新增模型输出，只作假设检查。
    queries.append(view(index, source['task'], first_excluded, 'task_text_hypothetical_query', 'initial_visibility'))
    queries.append(view(index, MANUAL_NEED, available, 'manual_posthoc_download_need', 'after_saved_discovery_visibility'))
    actions = arm['actions']
    trace = {'actual_action_sequence': [a['name'] for a in actions],
             'actual_discovery_calls': len([a for a in actions if a['name'] == 'discover_tools']),
             'remaining_actions_after_search': compare.MAX_ACTIONS-2,
             'remaining_actions_after_unnecessary_paper_lookup': compare.MAX_ACTIONS-3,
             'discovery_plus_download_requires_actions': 2,
             'budget_note': 'equal budgets do not remove discovery overhead; extra get-paper action consumes possible discover+download path'}
    return queries, trace, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    queries, trace, manifest = prepare()
    print('4-1第三步：离线复算发现需求与工具排名；模型、网络、嵌入及业务调用均0。')
    print('保持原126工具及TF-IDF参数，使用已知download_file目标作事后诊断。')
    print('保存的模型need、原任务假设查询、人工下载需求明确分开；不算模型自行改进。')
    if args.preview:
        print('TOOL_RANK_DIAGNOSIS_PREVIEW_ONLY：来源与原候选复算一致，未创建正式结果。')
        return 0
    for query in queries:
        print('\n需求：', query['need'], '；来源：', query['origin'])
        print('Top5：', [(r['name'], round(r['score'], 6)) for r in query['top5']])
        print('download_file排名：', query['target_rank_among_positive_scores'], '；得分：', round(query['target_score'], 6))
        print('不同Top-K是否能看到：', query['target_in_top_k'])
    print('\n实际动作与预算：', json.dumps(trace, ensure_ascii=False))
    directory = ROOT/'运行结果'/('discovery_rank_diagnosis_'+datetime.now(compare.sdk.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'offline_discovery_ranking_saved_review_pending',
              'created_at': datetime.now(compare.sdk.CHINA).isoformat(),
              'script_sha256': compare.sdk.sha(Path(__file__)), 'index_script_sha256': INDEX_SCRIPT_SHA,
              'source_record_sha256': SOURCE_SHA, 'manifest': manifest, 'queries': queries, 'trace': trace,
              'model_requests': 0, 'network_requests': 0, 'business_operations': 0,
              'scope': 'post-hoc index diagnosis with known target and manual queries; not new model evidence or causal performance improvement'}
    (directory/'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('TOOL_RANK_DIAGNOSIS_SAVED：需求来源、完整正分排名及预算记录已保存，待核对。')
    print('记录路径：', directory/'record.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

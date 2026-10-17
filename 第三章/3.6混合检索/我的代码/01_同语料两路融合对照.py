"""3-6第一步：复用已核对的两路排名，比较官方RRF与加权融合；无模型、无网络。"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MANIFEST = ROOT / '官方资料/manifest.json'
UPSTREAM = ROOT / '官方资料/upstream'
FIXTURE = ROOT / '测试材料/融合输入来源.json'
COMMIT = 'dbc046eb896ac4e39aa19c7774c8bf49583b89a6'
CHINA = timezone(timedelta(hours=8))
K = 5
RRF_K = 60
WEIGHTS = {'dense': 0.5, 'sparse': 0.5}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_materials():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['commit'] != COMMIT:
        raise ValueError('官方版本发生变化')
    for item in manifest['files']:
        path = (UPSTREAM / item['path']).resolve()
        if not path.is_relative_to(UPSTREAM.resolve()) or sha(path) != item['sha256']:
            raise ValueError(f"官方文件校验失败：{item['path']}")
    fixture = json.loads(FIXTURE.read_text(encoding='utf-8'))
    records = {}
    for name, ref in fixture['sources'].items():
        path = (PROJECT / ref['record_relative_to_project']).resolve()
        allowed = (PROJECT / ref['allowed_results_relative_to_project']).resolve()
        if not path.is_relative_to(allowed) or not allowed.is_relative_to(PROJECT):
            raise ValueError('输入来源不在对应实验的运行结果目录')
        audit_path = path.parent / '核对记录.json'
        if sha(path) != ref['record_sha256'] or sha(audit_path) != ref['audit_sha256']:
            raise ValueError(f'{name}原始记录或核对记录发生变化')
        record = json.loads(path.read_text(encoding='utf-8'))
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if record['status'] != ref['expected_status'] or audit['status'] != ref['expected_audit_status']:
            raise ValueError('输入尚未完成实测核对')
        if audit['record_sha256'] != ref['record_sha256'] or record['upstream_commit'] != COMMIT:
            raise ValueError('来源核对关联或官方版本不一致')
        records[name] = record
    dense, sparse = records['dense'], records['sparse']
    if dense['corpus'] != sparse['corpus'] or dense['labels'] != sparse['labels']:
        raise ValueError('两路文档、查询或标注不同')
    if sparse['source_3_4']['record_sha256'] != fixture['sources']['dense']['record_sha256']:
        raise ValueError('BM25对照中引用的稠密结果与本次来源不同')
    if len(dense['corpus']) != 12 or len(dense['labels']) != 5:
        raise ValueError('本步固定12篇文档和5条查询')
    return fixture, dense, sparse


def load_fusion():
    # 官方fusion.py只有Python标准库导入；不加载HTTP服务或自动下载重排器。
    path = UPSTREAM / 'chapter3/retrieval-pipeline/fusion.py'
    spec = importlib.util.spec_from_file_location('pinned_hybrid_fusion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_ranked(ranked, allowed):
    ids = [doc for doc, _ in ranked]
    scores = [score for _, score in ranked]
    if len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError('候选ID重复或不属于语料')
    if any(not math.isfinite(score) for score in scores):
        raise ValueError('候选分数不是有限数')
    if any(a < b for a, b in zip(scores, scores[1:])):
        raise ValueError('原始候选没有按分数降序排列')


def metrics(ids, relevant):
    top = ids[:K]
    hits = [doc for doc in top if doc in relevant]
    first = next((i for i, doc in enumerate(top, 1) if doc in relevant), None)
    return {'recall_at_5': len(hits) / len(relevant), 'precision_at_5_fixed_slots': len(hits) / K,
            'rr_at_5': 1 / first if first else 0.0, 'hits': hits, 'returned_at_5': len(top)}


def macro(rows):
    return {out: sum(row[key] for row in rows) / len(rows)
            for out, key in [('recall_at_5', 'recall_at_5'),
                             ('precision_at_5_fixed_slots', 'precision_at_5_fixed_slots'),
                             ('mrr_at_5', 'rr_at_5')]}


def fuse_one(module, query, ranked):
    """只接收查询和两路候选；相关性标注不参与融合。保留并集全部候选。"""
    rankings = {}
    details = {}
    for method in ('rrf', 'weighted'):
        # RRF两路各权重1；加权分数融合两路各0.5。均固定，不按测试标签调参。
        weights = None if method == 'rrf' else WEIGHTS
        fused = module.fuse(ranked, method=method, k=RRF_K, weights=weights)
        rankings[method] = [{'rank': i, 'doc_id': doc, 'score': score}
                            for i, (doc, score) in enumerate(fused, 1)]
        per_source = {}
        for source, items in ranked.items():
            if method == 'rrf':
                per_source[source] = {doc: 1 / (RRF_K + rank) for rank, (doc, _) in enumerate(items, 1)}
            else:
                per_source[source] = {doc: WEIGHTS[source] * value
                                      for doc, value in module.min_max_normalize(dict(items)).items()}
        details[method] = []
        for doc, score in fused:
            contributions = {source: values.get(doc, 0.0) for source, values in per_source.items()}
            if not math.isclose(score, sum(contributions.values()), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('融合贡献重算不符')
            details[method].append({'doc_id': doc, 'contributions': contributions, 'score': score})
    return {'query': query, 'fused_rankings': rankings, 'fusion_contributions': details}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='只核验材料，不融合、不创建结果目录')
    args = parser.parse_args(argv)
    fixture, dense, sparse = load_materials()
    print('3-6两路融合：沿用12篇文档、5条查询；每路最多取前5，最终展示Top-5。')
    print('复用已核对的MiniLM和BM25排名；官方RRF(k=60)与min-max后等权融合各做一次。')
    print('生成式模型请求0，嵌入前向0，网络请求0；不安装依赖或下载模型。')
    print('本步只做候选融合；后续再做神经重排，不预设混合一定优于单路。')
    if args.preview:
        print('FUSION_PREVIEW_ONLY：官方资料及两份已核对来源验证通过；未运行正式融合。')
        return 0
    directory = ROOT / '运行结果' / ('hybrid_fusion_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'record.json'
    record = {'stage': 'chapter3_hybrid_fusion', 'status': 'running',
              'created_at': datetime.now(CHINA).isoformat(), 'script_sha256': sha(__file__),
              'upstream_commit': COMMIT, 'official_manifest_sha256': sha(MANIFEST),
              'fixture_sha256': sha(FIXTURE), 'sources': fixture['sources'],
              'corpus': dense['corpus'], 'labels': dense['labels'],
              'protocol': {'per_source_top_k': K, 'output_top_k': K, 'rrf_k': RRF_K,
                           'rrf_weights': {'dense': 1.0, 'sparse': 1.0}, 'weighted_weights': WEIGHTS,
                           'weighted_normalization': 'per-source min-max on returned top-5; equal scores -> 1',
                           'missing_source_contribution': 0, 'ties': 'upstream stable sort; dense source inserted first',
                           'candidates': 'union of both source top-5; preserve all union candidates for later reranking',
                           'precision_denominator': 5, 'rr_scope': 'top 5',
                           'source_code_extra_query_included': False},
              'model_requests': 0, 'embedding_forward_batches': 0, 'network_requests': 0,
              'reranking_performed': False,
              'scope': '同语料离线候选融合学习；非官方默认语料、非神经重排或完整服务部署', 'queries': []}
    dump(path, record)
    try:
        module = load_fusion()
        allowed = {d['doc_id'] for d in dense['corpus']}
        dense_rows = {q['query']: q for q in dense['exact']['queries']}
        sparse_rows = {q['query']: q for q in sparse['queries']}
        if set(dense_rows) != set(dense['labels']) or set(sparse_rows) != set(dense['labels']):
            raise ValueError('来源查询集合与标注不同')
        for query in dense['labels']:
            ranked = {
                'dense': [(r['doc_id'], r['cosine_similarity']) for r in dense_rows[query]['full_ranking'][:K]],
                'sparse': [(r['doc_id'], r['score']) for r in sparse_rows[query]['full_matched_ranking'][:K]]}
            for items in ranked.values():
                validate_ranked(items, allowed)
            result = fuse_one(module, query, ranked)
            union = list(dict.fromkeys(doc for items in ranked.values() for doc, _ in items))
            if any({r['doc_id'] for r in rows} != set(union) for rows in result['fused_rankings'].values()):
                raise ValueError('融合后候选集合与两路并集不一致')
            result['source_rankings'] = {
                source: [{'rank': i, 'doc_id': doc, 'score': score} for i, (doc, score) in enumerate(items, 1)]
                for source, items in ranked.items()}
            relevant = dense['labels'][query]
            result.update(relevant_doc_ids=relevant, candidate_union=union,
                          candidate_recall=len(set(union) & set(relevant)) / len(relevant))
            all_rankings = {**result['source_rankings'], **result['fused_rankings']}
            result['metrics'] = {name: metrics([r['doc_id'] for r in rows], relevant)
                                 for name, rows in all_rankings.items()}
            result['rank_changes'] = [
                {'doc_id': doc, **{name: next((r['rank'] for r in rows if r['doc_id'] == doc), None)
                                  for name, rows in all_rankings.items()}} for doc in union]
            record['queries'].append(result)
            dump(path, record)
            print(f'\n查询：{query}；标注相关：{relevant}')
            print(f'  候选并集{len(union)}篇，标注覆盖={result["candidate_recall"]:.0%}')
            for name, rows in all_rankings.items():
                print(f'  {name:8}：{[r["doc_id"] for r in rows[:K]]}；RR@5={result["metrics"][name]["rr_at_5"]:.3f}')
            if query == 'training neural networks':
                print('  逐篇排名变化（—表示该路未召回）：')
                for row in result['rank_changes']:
                    print('   ', row['doc_id'], '  '.join(f'{name}={row[name] if row[name] is not None else "—"}'
                                                        for name in all_rankings))
                print('  RRF逐路贡献：', result['fusion_contributions']['rrf'])
        record['macro'] = {name: macro([q['metrics'][name] for q in record['queries']])
                           for name in ('dense', 'sparse', 'rrf', 'weighted')}
        record['candidate_recall_macro'] = sum(q['candidate_recall'] for q in record['queries']) / len(record['queries'])
        record['status'] = 'hybrid_fusion_saved'
        dump(path, record)
        print('\n共同口径宏平均（Precision固定除以5、MRR仅看前5）：', record['macro'])
        print('HYBRID_FUSION_SAVED：仅融合步骤完成；排名、贡献与指标待核对，神经重排尚未运行。')
        print('记录路径：', path)
        return 0
    except Exception as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        dump(path, record)
        print('本次未完成：', record['error'])
        print('记录路径：', path)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

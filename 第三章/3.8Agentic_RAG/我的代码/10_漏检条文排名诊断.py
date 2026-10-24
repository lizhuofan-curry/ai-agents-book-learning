"""已知目标的离线BM25诊断：完整排名、词项贡献、Top-K与人工简短查询。"""
from collections import Counter
from datetime import datetime
import argparse
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'运行结果/gap_prompt_20261024_200015_288841/record.json'
AUDIT = SOURCE.with_name('核对记录.json')
TARGET = '第六十五条'
SHORT_QUERY = '累犯 过失犯罪'
DEPTHS = (5, 10, 15, 20, 50)


def helper():
    spec = importlib.util.spec_from_file_location('diagnostic_retrieval', ROOT/'我的代码/01_单次与分解检索对照.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    h = helper()
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    if h.sha(SOURCE) != audit['record_sha256']:
        raise ValueError('待诊断记录与已核对指纹不符')
    if source['arm']['status'] != 'answered_review_pending' or len(source['arm']['searches']) != 3:
        raise ValueError('本入口固定诊断已完成的三次检索')
    if (h.sha(h.MANIFEST) != source['manifest_sha256']
            or h.sha(Path(h.__file__)) != source['retrieval_script_sha256']):
        raise ValueError('检索实现或来源改变')
    retriever, _, _ = h.load_sources()
    targets = [c for c in retriever.chunks if h.leading(c['text']) == TARGET]
    if len(targets) != 1:
        raise ValueError('本单文档中必须存在唯一目标条文')
    return h, source, retriever, targets[0]


def contributions(retriever, query, chunk, tokenize):
    """从片段正文独立计算TF/查询频次；DF和长度统计来自冻结索引。"""
    tf = Counter(tokenize(chunk['text']))
    qtf = Counter(tokenize(query))
    length = sum(tf.values())
    rows = []
    for term, query_count in qtf.items():
        frequency = tf[term]
        df = retriever._df.get(term, 0)
        idf = math.log(1+(len(retriever.chunks)-df+0.5)/(df+0.5)) if df else 0.0
        denominator = frequency + retriever.k1*(1-retriever.b+retriever.b*length/(retriever._avg_len or 1))
        value = query_count*idf*frequency*(retriever.k1+1)/denominator if frequency else 0.0
        rows.append({'term': term, 'query_frequency': query_count, 'tf': frequency,
                     'df': df, 'idf': idf, 'contribution': value})
    return {'chunk_id': chunk['chunk_id'], 'text': chunk['text'], 'token_length': length,
            'score': sum(row['contribution'] for row in rows),
            'terms': sorted(rows, key=lambda row: (-row['contribution'], row['term']))}


def diagnose(retriever, query, target, h):
    # 官方search仅返回正分条目，按(score, 原始片段索引)降序排列。
    hits = retriever.search(query, len(retriever.chunks))
    full = [{'rank': i+1, 'chunk_id': hit['chunk_id'], 'article': h.leading(hit['text']),
             'score': hit['score']} for i, hit in enumerate(hits)]
    target_row = next((r for r in full if r['chunk_id'] == target['chunk_id']), None)
    target_detail = contributions(retriever, query, target, h.tokenize)
    if target_row:
        if not math.isclose(target_row['score'], target_detail['score'], rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError('目标BM25贡献之和不一致')
    elif abs(target_detail['score']) > 1e-10:
        raise ValueError('非零目标未在完整正分排名出现')
    details = []
    for hit in hits[:5]:
        detail = contributions(retriever, query, hit, h.tokenize)
        if not math.isclose(detail['score'], hit['score'], rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError('候选BM25贡献之和不一致')
        details.append(detail)
    rank = target_row['rank'] if target_row else None
    return {'query': query, 'target_chunk_id': target['chunk_id'], 'target_rank': rank,
            'positive_score_candidates': len(hits), 'full_positive_ranking': full,
            'target_detail': target_detail, 'top5_details': details,
            'top_k_target_hit': {str(k): rank is not None and rank <= k for k in DEPTHS},
            'rank_note': 'Exact frozen search order; score ties use descending original chunk index. Zero-score targets have no rank.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    h, source, retriever, target = prepare()
    print('3-8离线排名诊断：目标', TARGET, '，片段ID', target['chunk_id'])
    print('固定506块；诊断三条已保存模型查询及人工简短查询；模型/网络请求均0。')
    print('本步使用已知目标作事后诊断，不是Agent自主改进，也不生成法律回答。')
    if args.preview:
        print('RANK_DIAGNOSIS_PREVIEW_ONLY：来源与目标存在性校验通过，未执行查询或创建结果目录。')
        return 0
    directory = ROOT/'运行结果'/('rank_diagnosis_'+datetime.now(h.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(h.CHINA).isoformat(),
        'script_sha256': h.sha(Path(__file__)), 'source_record_sha256': h.sha(SOURCE),
        'source_audit_sha256': h.sha(AUDIT), 'manifest_sha256': h.sha(h.MANIFEST),
        'retrieval_script_sha256': h.sha(Path(h.__file__)),
        'source_relative_path': SOURCE.relative_to(ROOT).as_posix(),
        'target_article': TARGET, 'target': target, 'depths': DEPTHS,
        'model_requests': 0, 'network_requests': 0,
        'scope': 'post-hoc known-target lexical ranking diagnosis; manually selected short query; not agent improvement',
        'results': []}
    h.dump(path, record)
    try:
        queries = [(f'model_query_{i+1}', s['query'], s['hits'])
                   for i, s in enumerate(source['arm']['searches'])]
        queries.append(('manual_short_query', SHORT_QUERY, None))
        for origin, query, saved_hits in queries:
            row = diagnose(retriever, query, target, h)
            if saved_hits is not None and retriever.search(query, 5) != saved_hits:
                raise ValueError('保存的模型查询结果无法复现')
            row['origin'] = origin
            record['results'].append(row)
            h.dump(path, record)
            print('\n查询：', query, '来源：', origin)
            print('目标排名：', row['target_rank'] if row['target_rank'] is not None else '无正分排名',
                  '；BM25：', round(row['target_detail']['score'], 6))
            print('扩大Top-K能否命中：', row['top_k_target_hit'])
            print('目标得分贡献前8项：', [(t['term'], round(t['contribution'], 4))
                  for t in row['target_detail']['terms'] if t['contribution'] > 0][:8])
            for hit in row['top5_details'][:1]:
                print('第一名片段：', hit['chunk_id'], h.leading(hit['text']))
                print('第一名得分贡献前8项：', [(t['term'], round(t['contribution'], 4))
                      for t in hit['terms'] if t['contribution'] > 0][:8])
        if h.sha(SOURCE) != record['source_record_sha256'] or h.sha(h.MANIFEST) != record['manifest_sha256']:
            raise ValueError('运行期间源记录或清单变化')
        h.load_sources()
        record['status'] = 'rank_diagnosis_saved_review_pending'
        h.dump(path, record)
        print('\nRANK_DIAGNOSIS_SAVED：排名、完整词项贡献与Top-K对照已保存，待核对。')
        print('记录路径：', path)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        h.dump(path, record)
        print('本步未完成：', record['error'])
        print('记录路径：', path)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

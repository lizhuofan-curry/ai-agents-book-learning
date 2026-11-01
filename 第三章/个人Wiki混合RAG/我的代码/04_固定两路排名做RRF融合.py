"""复用本人BM25和casefold语义排名，调用固定官方RRF函数；不重新检索或编码。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'测试材料/融合输入.json'
REFERENCE = ROOT.parent/'3.6混合检索/官方资料'
CHINA = timezone(timedelta(hours=8))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def load_rrf():
    manifest = json.loads((REFERENCE/'manifest.json').read_text(encoding='utf-8'))
    upstream = 'chapter3/retrieval-pipeline/fusion.py'
    item = next(f for f in manifest['files'] if f['path'] == upstream)
    path = REFERENCE/'upstream'/upstream
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if (manifest['commit'] != 'dbc046eb896ac4e39aa19c7774c8bf49583b89a6'
            or sha(raw) != item['sha256'] or blob != item['git_blob_sha1']):
        raise ValueError('固定官方RRF源码与清单不符，不使用变化后的代码')
    spec = importlib.util.spec_from_file_location('pinned_wiki_rrf', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reciprocal_rank_fusion, {'commit': manifest['commit'], 'path': upstream,
        'sha256': sha(raw), 'git_blob_sha1': blob, 'url': item['url']}


def validate_hits(hits, chunks):
    if not isinstance(hits, list) or len(hits) > 5:
        raise ValueError('每路最多取保存的Top5')
    ids = [h['chunk_id'] for h in hits]
    if len(set(ids)) != len(ids):
        raise ValueError('输入排名有重复片段')
    previous = math.inf
    for hit in hits:
        score = hit['score']
        if type(score) not in {int, float} or not math.isfinite(score) or score > previous:
            raise ValueError('输入分数须有限，且保存顺序为降序；不替源记录重排')
        previous = score
        if hit['chunk_id'] not in chunks:
            raise ValueError('排名引用未知片段')
        chunk = chunks[hit['chunk_id']]
        if any(hit[key] != chunk[key] for key in ['document_id', 'path', 'text', 'source_line_start', 'source_line_end']):
            raise ValueError('排名片段正文、来源或跨度与冻结语料不一致')


def fuse_rankings(dense, sparse, rrf):
    # 顺序固定dense优先，沿用官方稳定排序处理同分。
    ordered = {'dense': dense, 'sparse': sparse}
    sources = {name: [(h['chunk_id'], h['score']) for h in hits] for name, hits in ordered.items()}
    ranks = {name: {h['chunk_id']: rank for rank, h in enumerate(hits, 1)} for name, hits in ordered.items()}
    fused = rrf(sources, k=60, weights={'dense': 1.0, 'sparse': 1.0})
    rows = []
    for chunk_id, score in fused:
        dense_rank, sparse_rank = ranks['dense'].get(chunk_id), ranks['sparse'].get(chunk_id)
        contributions = {'dense': 1/(60+dense_rank) if dense_rank else 0.0,
                         'sparse': 1/(60+sparse_rank) if sparse_rank else 0.0}
        if not math.isclose(score, sum(contributions.values()), rel_tol=0, abs_tol=1e-15):
            raise ValueError('官方RRF结果与独立名次贡献复算不一致')
        rows.append({'chunk_id': chunk_id, 'rrf_score': score,
                     'dense_rank': dense_rank, 'sparse_rank': sparse_rank, 'contributions': contributions})
    if {row['chunk_id'] for row in rows} != set(ranks['dense']) | set(ranks['sparse']):
        raise ValueError('候选并集发生丢失或补造')
    return rows


def load_inputs():
    plan = json.loads(INPUT.read_text(encoding='utf-8'))
    if (plan['method'], plan['rrf_k'], plan['candidate_top_k_each'], plan['output_top_k'], plan['source_order']) != ('rrf', 60, 5, 5, ['dense', 'sparse']):
        raise ValueError('本步使用固定RRF60、各路Top5、dense先插入，不调参数')
    if plan['source_weights'] != {'dense': 1.0, 'sparse': 1.0} or plan['relevance_labels'] is not None:
        raise ValueError('本步固定等权，无完整金标，不在融合时调整权重或填写指标')
    parent = (ROOT/'本地运行结果').resolve()
    loaded, originals = {}, {}
    for kind in ['dense', 'sparse']:
        folder = (parent/plan[kind+'_run']).resolve()
        expected_prefix = 'wiki_dense_casefold_' if kind == 'dense' else 'wiki_sparse_'
        if folder.parent != parent or not folder.name.startswith(expected_prefix):
            raise ValueError('输入来源批次越界或类型不符')
        path = folder/'record.json'
        raw = path.read_bytes()
        if sha(raw) != plan[kind+'_record_sha256']:
            raise ValueError('指定本人源记录已变化，不自动改用其他批次')
        record = json.loads(raw)
        if not all(value is True for value in record['checks'].values()) or not record['checks']:
            raise ValueError('源记录技术检查未通过')
        loaded[kind], originals[str(path)] = record, sha(raw)
    dense, sparse = loaded['dense'], loaded['sparse']
    if (dense['model_input_normalization'] != 'casefold' or dense['unknown_token_counts']['queries'] != [0, 0, 0]
            or dense['source_record_sha256'] != plan['sparse_record_sha256']
            or dense['source_corpus_sha256'] != plan['source_corpus_sha256']):
        raise ValueError('稠密来源不是同语料casefold对照，或查询仍含UNK')
    sparse_folder = parent/plan['sparse_run']
    corpus_path = sparse_folder/'corpus.json'
    raw = corpus_path.read_bytes()
    if sha(raw) != plan['source_corpus_sha256'] or sha(raw) != sparse['corpus_sha256']:
        raise ValueError('冻结语料哈希不符')
    corpus = json.loads(raw)
    chunks = {c['id']: c for c in corpus['chunks']}
    if len(chunks) != 65 or len(corpus['documents']) != 12:
        raise ValueError('固定12页65块不符')
    if len(dense['results']) != 3 or len(sparse['results']) != 3:
        raise ValueError('须为同三条查询')
    for dense_row, sparse_row in zip(dense['results'], sparse['results']):
        if any(dense_row[k] != sparse_row[k] for k in ['query_id', 'query', 'retrieval_query']):
            raise ValueError('两路查询ID、原句或检索文本不一致')
        validate_hits(dense_row['hits'], chunks)
        validate_hits(sparse_row['hits'], chunks)
    originals[str(corpus_path)] = sha(raw)
    dense_folder = parent/plan['dense_run']
    for name, expected in dense['vector_files_sha256'].items():
        if name not in {'documents.npy', 'queries.npy'} or sha((dense_folder/name).read_bytes()) != expected:
            raise ValueError('稠密向量文件被改动')
        originals[str(dense_folder/name)] = expected
    return plan, loaded, chunks, originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    plan, loaded, chunks, originals = load_inputs()
    rrf, reference = load_rrf()
    print('个人Wiki第三步：本人BM25与casefold语义Top5，固定官方RRF60等权融合。')
    print('0次嵌入、回答生成、MCP或网络请求；只复用旧排名，不改原文、向量或源记录。')
    if args.preview:
        print('WIKI_FUSION_PREVIEW_ONLY：两路来源、片段与官方源码核对通过；未融合，无新结果目录。')
        return 0
    folder = ROOT/'本地运行结果'/('wiki_fusion_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    path = folder/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
        'script_sha256': sha(Path(__file__).read_bytes()), 'input_plan_sha256': sha(INPUT.read_bytes()),
        'plan': plan, 'official_rrf_reference': reference, 'source_file_hashes': originals,
        'embedding_forward_batches': 0, 'generation_model_requests': 0, 'network_requests': 0,
        'mcp_requests': 0, 'automatic_retries': 0, 'results': [], 'relevance_labels': None, 'metrics': None,
        'scope': 'Fixed ranked-list fusion and source consistency only; not semantic correctness, full relevance evaluation or answer generation.'}
    dump(path, record)
    print('本地记录路径：'+str(path), flush=True)
    try:
        for dense_row, sparse_row in zip(loaded['dense']['results'], loaded['sparse']['results']):
            all_rows = fuse_rankings(dense_row['hits'], sparse_row['hits'], rrf)
            for row in all_rows:
                chunk = chunks[row['chunk_id']]
                row.update({key: chunk[key] for key in ['document_id', 'path', 'text', 'source_line_start', 'source_line_end']})
            result = {'query_id': dense_row['query_id'], 'query': dense_row['query'],
                      'retrieval_query': dense_row['retrieval_query'],
                      'dense_top5_ids': [h['chunk_id'] for h in dense_row['hits']],
                      'sparse_top5_ids': [h['chunk_id'] for h in sparse_row['hits']],
                      'candidate_union_count': len(all_rows), 'candidate_union_ranked': all_rows,
                      'hits': all_rows[:5], 'relevance_review': 'pending'}
            record['results'].append(result)
            dump(path, record)
            print('\n'+result['query_id']+'：'+result['query'])
            print('  BM25：'+' → '.join(result['sparse_top5_ids']))
            print('  语义：'+' → '.join(result['dense_top5_ids']))
            print(f'  候选并集：{len(all_rows)}块；融合Top5：')
            for rank, hit in enumerate(result['hits'], 1):
                print(f'  {rank}. {hit["chunk_id"]} RRF={hit["rrf_score"]:.6f}，语义名次={hit["dense_rank"]}，关键词名次={hit["sparse_rank"]}')
                print('     '+hit['path']+f' 第{hit["source_line_start"]}—{hit["source_line_end"]}行')
                print('     '+' '.join(hit['text'].split())[:140])
        record['checks'] = {
            'all_original_source_files_unchanged': all(sha(Path(name).read_bytes()) == value for name, value in originals.items()),
            'three_queries_fused': len(record['results']) == 3,
            'top5_only_from_frozen_union': all(len(row['hits']) == 5 and set(h['chunk_id'] for h in row['hits']) <= set(row['dense_top5_ids']) | set(row['sparse_top5_ids']) for row in record['results']),
            'every_score_matches_rank_contributions': all(math.isclose(h['rrf_score'], sum(h['contributions'].values()), rel_tol=0, abs_tol=1e-15) for row in record['results'] for h in row['candidate_union_ranked']),
        }
        passed = all(record['checks'].values())
        record['status'] = 'fusion_saved_relevance_review_pending' if passed else 'failed_checks'
        print('\n技术检查：'+json.dumps(record['checks'], ensure_ascii=False))
        print('WIKI_FUSION_SAVED：只完成固定排名融合；是否更相关及回答是否充分仍待核对。')
        return 0 if passed else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('融合未完成，不重试：'+str(exc))
        return 1
    finally:
        dump(path, record)
        print('本地记录路径：'+str(path), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

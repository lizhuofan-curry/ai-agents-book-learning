"""只用本地MiniLM权重做迁移健康检查，不写入正式实验目录。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.uv-cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='可选：保存健康检查结果')
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    dense = load('migration_dense', '第三章/3.4稠密向量检索/我的代码/03_真实文本嵌入与检索.py')
    helper, corpus, labels, definitions = dense.load_sources()
    model_path, _ = dense.load_model_files()
    encoder = definitions['DenseEncoder'](str(model_path), pooling='mean', device='cpu', max_length=256)
    docs = encoder.encode([d['text'] for d in corpus])
    queries = encoder.encode(list(labels))
    saved_path = ROOT / '第三章/3.4稠密向量检索/运行结果/text_embedding_20261017_213507_432933'
    saved_docs = np.load(saved_path / 'documents.npy', allow_pickle=False)
    saved_queries = np.load(saved_path / 'queries.npy', allow_pickle=False)
    ids = [d['doc_id'] for d in corpus]
    current_rankings = [[ids[i] for i in np.argsort(-(docs @ q))] for q in queries]
    old_rankings = [[ids[i] for i in np.argsort(-(saved_docs @ q))] for q in saved_queries]
    max_difference = max(float(np.abs(docs - saved_docs).max()), float(np.abs(queries - saved_queries).max()))
    if current_rankings != old_rankings or max_difference > 1e-4:
        raise ValueError('迁移后嵌入与历史排名或数值差异超出阈值')

    _, indexes = helper.load_official()
    index_checks = {}
    for name, index in [('annoy', indexes.AnnoyIndex(384, n_trees=10)),
                        ('hnsw', indexes.HNSWIndex(384))]:
        for doc_id, vector in zip(ids, docs):
            index.add_item(doc_id, vector)
        index.rebuild_index()
        found, _ = index.search(queries[0], 5)
        self_hits, _ = index.search(docs[0], 1)
        if (len(found) != 5 or len(set(found)) != 5 or not set(found) <= set(ids)
                or self_hits != [ids[0]]):
            raise ValueError(name + '本机索引返回无效或未找回自身向量')
        index_checks[name] = {'top_5': found, 'self_query_passed': True,
                             'exact_top_5_overlap': len(set(found) & set(current_rankings[0][:5])) / 5,
                             'note': '近似索引跨平台重建不要求与精确排序完全相同'}

    rerank = load('migration_rerank', '第三章/3.6混合检索/我的代码/03_固定候选神经重排.py')
    _, fusion, _, _ = rerank.load_materials()
    _, tokenizer, model = rerank.load_local_model()
    corpus_by_id = {d['doc_id']: d['text'] for d in fusion['corpus']}
    saved = json.loads((ROOT / '第三章/3.6混合检索/运行结果/neural_rerank_20261018_001724_653459/record.json').read_text())
    ranks_match, differences, pairs = True, [], 0
    for row in saved['queries']:
        candidates = row['candidate_ids_in_model_order']
        encoded = tokenizer([[row['query'], corpus_by_id[x]] for x in candidates],
                            padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.inference_mode():
            scores = model(**encoded).logits.view(-1).tolist()
        differences.extend(abs(a-b) for a,b in zip(scores, row['raw_logits']))
        ranking = rerank.rank_scores(candidates, scores)
        ranks_match &= [x['doc_id'] for x in ranking] == [x['doc_id'] for x in row['reranked_full_ranking']]
        pairs += len(candidates)
    if not ranks_match or max(differences) > 1e-3:
        raise ValueError('迁移后重排与历史排名或数值差异超出阈值')
    result = {'status': 'local_model_health_passed', 'python': sys.version.split()[0],
              'torch': torch.__version__, 'device': 'cpu', 'network_requests': 0,
              'dense': {'document_count': len(corpus), 'query_count': len(labels),
                        'dimension': 384, 'all_rankings_match': True,
                        'max_embedding_absolute_difference': max_difference},
              'native_indexes': index_checks,
              'rerank': {'pairs': pairs, 'query_count': len(saved['queries']),
                         'all_rankings_match': ranks_match,
                         'max_logit_absolute_difference': max(differences)},
              'scope': '迁移后本地权重及原生索引健康检查，非新增正式实验或速度基准'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

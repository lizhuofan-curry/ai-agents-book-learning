"""用固定中文BGE在本人已冻结的Wiki片段上运行CPU语义检索。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_PLAN = ROOT/'测试材料/语义检索输入.json'
CHINA = timezone(timedelta(hours=8))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def load_snapshot():
    plan = json.loads(INPUT_PLAN.read_text(encoding='utf-8'))
    parent = (ROOT/'本地运行结果').resolve()
    folder = (parent/plan['source_run']).resolve()
    if folder.parent != parent or not folder.name.startswith('wiki_sparse_'):
        raise ValueError('固定来源批次越界')
    record_raw, corpus_raw = (folder/'record.json').read_bytes(), (folder/'corpus.json').read_bytes()
    if sha(record_raw) != plan['source_record_sha256'] or sha(corpus_raw) != plan['source_corpus_sha256']:
        raise ValueError('指定的本人来源记录或冻结语料已变化，不自动换用最新批次')
    record, corpus = json.loads(record_raw), json.loads(corpus_raw)
    if record['status'] != 'sparse_baseline_saved_relevance_review_pending' or not all(v is True for v in record['checks'].values()):
        raise ValueError('来源关键词批次技术检查未通过')
    docs = {d['id']: d for d in corpus['documents']}
    chunks, queries = corpus['chunks'], record['results']
    if (len(docs) != 12 or len(chunks) != 65 or len(queries) != 3
            or len({c['id'] for c in chunks}) != 65
            or [row['retrieval_query'] for row in queries] != plan['retrieval_texts']):
        raise ValueError('固定语料、片段或查询结构不符')
    if (plan['batch_size'], plan['maximum_forward_batches'], plan['top_k'], plan['device'], plan['seed']) != (8, 10, 5, 'cpu', 42):
        raise ValueError('本步固定batch8、前向10批、Top5、CPU和seed42，修改方案后需另行核对')
    for chunk in chunks:
        doc = docs[chunk['document_id']]
        if (doc['text'][chunk['source_char_start']:chunk['source_char_end']] != chunk['text']
                or sha(chunk['text'].encode()) != chunk['text_sha256'] or chunk['path'] != doc['path']):
            raise ValueError('冻结片段与原文跨度不一致')
    sparse = load(Path(__file__).with_name('01_固定Wiki语料与关键词检索.py'), 'frozen_sparse')
    if sha(Path(sparse.__file__).read_bytes()) != record['script_sha256']:
        raise ValueError('原关键词实现已变化，先核对版本，不替换旧排名')
    expected = sparse.rank_bm25(chunks, record['plan']['queries'], 1.2, .75, 5)
    if expected != queries:
        raise ValueError('保存的关键词排名与冻结输入复算不一致')
    return plan, folder, record, corpus


def validate_vectors(vectors, count, dimension=512):
    import numpy as np
    if (vectors.shape != (count, dimension) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6)):
        raise ValueError('向量形状、精度、有限性或L2范数异常')


def rank_dense(chunks, queries, documents, query_vectors, top_k):
    if documents.ndim != 2 or query_vectors.ndim != 2 or documents.shape[1] != query_vectors.shape[1]:
        raise ValueError('语料与查询向量维度不符')
    validate_vectors(documents, len(chunks), documents.shape[1])
    validate_vectors(query_vectors, len(queries), documents.shape[1])
    scores = query_vectors @ documents.T
    results = []
    for row, row_scores in zip(queries, scores):
        indices = sorted(range(len(chunks)), key=lambda i: (-float(row_scores[i]), chunks[i]['id']))[:top_k]
        hits = []
        for index in indices:
            chunk = chunks[index]
            hits.append({'chunk_id': chunk['id'], 'document_id': chunk['document_id'],
                'path': chunk['path'], 'source_line_start': chunk['source_line_start'],
                'source_line_end': chunk['source_line_end'], 'score': float(row_scores[index]),
                'text': chunk['text']})
        results.append({'query_id': row['query_id'], 'query': row['query'],
                        'retrieval_query': row['retrieval_query'], 'hits': hits,
                        'relevance_review': 'pending'})
    return results


def model_inputs(chunks, queries, instruction, casefold=False):
    documents = [chunk['text'] for chunk in chunks]
    question_texts = [instruction+row['retrieval_query'] for row in queries]
    if casefold:
        documents = [text.casefold() for text in documents]
        question_texts = [text.casefold() for text in question_texts]
    return documents, question_texts


def encode(texts, ids, kind, tokenizer, model, record, save, batch_size, budget):
    import numpy as np
    import torch
    input_ids = tokenizer(texts, padding=False, truncation=False)['input_ids']
    lengths = [len(tokens) for tokens in input_ids]
    if not lengths or max(lengths) > 512:
        raise ValueError('输入为空或超过512 tokens；不自动截断或改变既定片段')
    record.setdefault('token_lengths', {})[kind] = lengths
    unknown = getattr(tokenizer, 'unk_token_id', None)
    record.setdefault('unknown_token_counts', {})[kind] = [
        sum(token == unknown for token in tokens) if unknown is not None else None for tokens in input_ids]
    vectors = []
    for start in range(0, len(texts), batch_size):
        if record['embedding_forward_batches'] >= budget:
            raise RuntimeError('已达嵌入前向预算')
        selected = texts[start:start+batch_size]
        batch = {'kind': kind, 'ids': ids[start:start+batch_size], 'status': 'started',
                 'input_sha256': [sha(text.encode()) for text in selected]}
        record['batches'].append(batch)
        save()
        encoded = tokenizer(selected, padding=True, truncation=False, return_tensors='pt')
        # 调用前计数，若失败则该批保留started，不伪称completed。
        record['embedding_forward_batches'] += 1
        save()
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state[:, 0]
            normalized = torch.nn.functional.normalize(output.float(), p=2, dim=1)
        values = normalized.cpu().numpy().astype(np.float32, copy=False)
        validate_vectors(values, len(selected))
        vectors.append(values)
        batch['status'] = 'completed'
        save()
        print(f'本机嵌入前向第{record["embedding_forward_batches"]}批：{kind} {len(selected)}条。', flush=True)
    return np.concatenate(vectors, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='助手准备检查：仅两条合成句、1批前向，不运行正式排名')
    parser.add_argument('--casefold', action='store_true', help='独立对照：仅将文档与查询的模型输入统一casefold；原冻结文本不改')
    args = parser.parse_args()
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_IMPLICIT_TOKEN', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    plan, source_folder, source, corpus = load_snapshot()
    download = load(Path(__file__).with_name('02_下载固定中文嵌入模型.py'), 'bge_files')
    model_info, model_directory, model_files = download.verify_local_files()
    if (model_info['model_id'], model_info['revision']) != (plan['model_id'], plan['model_revision']):
        raise ValueError('模型与固定输入方案不符')
    print('个人Wiki第二步：固定中文BGE模型，复用本人65个片段与3条检索文本。')
    print('CPU执行，最多10批嵌入前向；0次回答生成、MCP或网络请求，无下载或重试。')
    print('模型输入大小写处理：'+('casefold；原文本与上一批结果保持不变。' if args.casefold else '原样；保留此选项用于复现未规范化的对照。'))
    if args.preview:
        print('WIKI_DENSE_PREVIEW_ONLY：来源/片段/模型文件已核对，不加载模型、不执行前向、无新结果目录。')
        return 0
    prefix = 'bge_smoke_' if args.smoke else ('wiki_dense_casefold_' if args.casefold else 'wiki_dense_')
    folder = ROOT/'本地运行结果'/(prefix+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    path = folder/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
        'script_sha256': sha(Path(__file__).read_bytes()), 'input_plan_sha256': sha(INPUT_PLAN.read_bytes()),
        'source_record': str(source_folder/'record.json'), 'source_record_sha256': plan['source_record_sha256'],
        'source_corpus_sha256': plan['source_corpus_sha256'], 'model': model_info, 'model_files_verified': model_files,
        'model_manifest_sha256': sha(download.MANIFEST.read_bytes()), 'device': 'cpu',
        'seed': plan['seed'], 'batch_size': plan['batch_size'], 'embedding_forward_batches': 0,
        'maximum_embedding_forward_batches': 1 if args.smoke else plan['maximum_forward_batches'],
        'generation_model_requests': 0, 'network_requests': 0, 'mcp_requests': 0,
        'automatic_retries': 0, 'batches': [], 'live_wiki_reread': False, 'test_only_smoke': args.smoke,
        'model_input_normalization': 'casefold' if args.casefold else 'none',
        'scope': 'Local CLS/L2 embeddings and cosine ranking only; no fused ranking, answer generation or relevance metrics.'}

    def save():
        dump(path, record)
    save()
    print('本地记录路径：'+str(path), flush=True)
    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
        record['versions'] = {name: importlib.metadata.version(name) for name in ['numpy', 'torch', 'transformers', 'safetensors']}
        torch.manual_seed(plan['seed'])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_directory), local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(str(model_directory), local_files_only=True,
                                          trust_remote_code=False, use_safetensors=True).to('cpu').eval()
        if model.config.hidden_size != 512 or model.config.max_position_embeddings != 512:
            raise ValueError('模型维度或上下文长度与清单不符')
        if args.smoke:
            synthetic = ['Agent通过工具读取资料。', 'Transformer处理序列。']
            if args.casefold:
                synthetic = [text.casefold() for text in synthetic]
            values = encode(synthetic,
                            ['SYNTHETIC1', 'SYNTHETIC2'], 'synthetic', tokenizer, model, record, save, 8, 1)
            record.update(status='synthetic_smoke_passed_no_retrieval', output_shape=list(values.shape))
            print('BGE_SMOKE_PASSED：仅合成句1批前向，不是本人正式检索。')
            return 0
        chunks, queries = corpus['chunks'], source['results']
        chunk_texts, query_texts = model_inputs(chunks, queries, model_info['query_instruction'], args.casefold)
        record['query_model_inputs'] = query_texts
        save()
        documents = encode(chunk_texts, [c['id'] for c in chunks], 'documents', tokenizer, model, record, save, 8, 10)
        query_vectors = encode(query_texts, [q['query_id'] for q in queries], 'queries', tokenizer, model, record, save, 8, 10)
        validate_vectors(documents, 65)
        validate_vectors(query_vectors, 3)
        results = rank_dense(chunks, queries, documents, query_vectors, 5)
        np.save(folder/'documents.npy', documents, allow_pickle=False)
        np.save(folder/'queries.npy', query_vectors, allow_pickle=False)
        record.update(results=results, document_vector_shape=list(documents.shape), query_vector_shape=list(query_vectors.shape),
            vector_files_sha256={name: sha((folder/name).read_bytes()) for name in ['documents.npy', 'queries.npy']},
            relevance_labels=None, metrics=None)
        record['checks'] = {
            'source_record_unchanged': sha((source_folder/'record.json').read_bytes()) == plan['source_record_sha256'],
            'frozen_corpus_unchanged': sha((source_folder/'corpus.json').read_bytes()) == plan['source_corpus_sha256'],
            'ten_batches_completed_within_budget': record['embedding_forward_batches'] == 10
                and len(record['batches']) == 10 and all(b['status'] == 'completed' for b in record['batches']),
            'no_input_truncation': max(n for lengths in record['token_lengths'].values() for n in lengths) <= 512,
            'all_three_queries_ranked': len(results) == 3 and all(len(row['hits']) == 5 for row in results),
            'dense_hits_reference_same_frozen_chunks': all(h['chunk_id'] in {c['id'] for c in chunks} for row in results for h in row['hits']),
        }
        passed = all(record['checks'].values())
        record['status'] = 'dense_saved_relevance_review_pending' if passed else 'failed_checks'
        for dense, sparse in zip(results, queries):
            print('\n'+dense['query_id']+'：'+dense['query'])
            print('  原BM25：'+' → '.join(h['chunk_id'] for h in sparse['hits']))
            for i, hit in enumerate(dense['hits'], 1):
                print(f'  语义{i}. {hit["chunk_id"]} {hit["path"]} 第{hit["source_line_start"]}—{hit["source_line_end"]}行，余弦{hit["score"]:.4f}')
                print('     '+' '.join(hit['text'].split())[:160])
        print('\n技术检查：'+json.dumps(record['checks'], ensure_ascii=False))
        print('查询中的UNK数量：'+json.dumps(record['unknown_token_counts']['queries']))
        print('WIKI_DENSE_SAVED：仅语义检索；相关性待核对，尚未融合或生成回答。')
        return 0 if passed else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('本步未完成，不重试：'+str(exc))
        return 1
    finally:
        save()
        print('本地记录路径：'+str(path), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

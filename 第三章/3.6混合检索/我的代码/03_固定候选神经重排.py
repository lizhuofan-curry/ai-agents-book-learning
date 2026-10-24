"""3-6第二步：冻结已核对的融合候选，用本地交叉编码器打分；不生成回答。"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
from project_paths import project_path

FIXTURE = ROOT / '测试材料/重排输入来源.json'
MODEL_MANIFEST = ROOT / '官方资料/轻量重排模型清单.json'
HELPER = ROOT / '我的代码/01_同语料两路融合对照.py'
MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L6-v2'
REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
MODEL_DIRECTORY = PROJECT / ('.uv-cache/huggingface/local_models/ms-marco-MiniLM-L6-v2_' + REVISION[:12])
CHINA = timezone(timedelta(hours=8))
MAX_LENGTH = 512
THREADS = 4
K = 5

# 本脚本只从核验过的E盘本地目录读取权重，不隐式访问Hub或读取凭证。
os.environ['HF_HOME'] = str(PROJECT / '.uv-cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_materials():
    fixture = read(FIXTURE)
    if sha(HELPER) != fixture['fusion_helper_sha256']:
        raise ValueError('已冻结的融合脚本发生变化')
    spec = importlib.util.spec_from_file_location('frozen_fusion_source_checks', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    _, dense, _ = helper.load_materials()  # 只核验来源，不搜索、不融合、不执行模型。
    path = (PROJECT / fixture['fusion_record_relative_to_project']).resolve()
    if not path.is_relative_to((ROOT / '运行结果').resolve()):
        raise ValueError('融合来源不在本实验运行结果目录')
    audit_path = path.parent / '核对记录.json'
    if sha(path) != fixture['fusion_record_sha256'] or sha(audit_path) != fixture['fusion_audit_sha256']:
        raise ValueError('融合来源或核对记录发生变化')
    fusion, audit = read(path), read(audit_path)
    if (fusion['status'] != 'hybrid_fusion_saved' or audit['status'] != 'hybrid_fusion_record_verified'
            or audit['record_sha256'] != sha(path)):
        raise ValueError('需要完整且已核对的融合记录')
    if fusion['sources'] != read(ROOT / '测试材料/融合输入来源.json')['sources']:
        raise ValueError('融合与两路来源关系不同')
    if fusion['corpus'] != dense['corpus'] or fusion['labels'] != dense['labels']:
        raise ValueError('融合语料或标注未与已核对来源对齐')
    if fusion['script_sha256'] != sha(HELPER) or fusion['official_manifest_sha256'] != sha(ROOT / '官方资料/manifest.json'):
        raise ValueError('融合代码或官方资料与实测时不一致')
    if fusion['fixture_sha256'] != sha(ROOT / '测试材料/融合输入来源.json'):
        raise ValueError('融合输入清单与实测时不一致')
    if len(fusion['queries']) != 5 or [q['query'] for q in fusion['queries']] != list(fusion['labels']):
        raise ValueError('固定五条查询集合不符')
    total = 0
    for q in fusion['queries']:
        ids = [r['doc_id'] for r in q['fused_rankings']['rrf']]
        if len(ids) != len(set(ids)) or set(ids) != set(q['candidate_union']):
            raise ValueError('RRF完整列表必须包含且仅包含候选并集')
        if set(ids) != {r['doc_id'] for r in q['fused_rankings']['weighted']}:
            raise ValueError('两种融合的候选集合不同')
        if not 1 <= len(ids) <= 8:
            raise ValueError('本步固定每题至多8对、单批处理；不隐式扩大预算')
        total += len(ids)
    if total != 27:
        raise ValueError('本步固定27个查询—候选配对')
    manifest = read(MODEL_MANIFEST)
    if sha(MODEL_MANIFEST) != fixture['model_manifest_sha256']:
        raise ValueError('模型清单发生变化')
    if (manifest['status'] != 'downloaded_verified' or manifest['model_id'] != MODEL_ID
            or manifest['revision'] != REVISION or project_path(manifest['local_path']) != MODEL_DIRECTORY.resolve()):
        raise ValueError('需要完整、固定版本的本地重排模型')
    for item in manifest['files']:
        f = (MODEL_DIRECTORY / item['name']).resolve()
        if (not f.is_relative_to(MODEL_DIRECTORY.resolve()) or f.stat().st_size != item['bytes']
                or sha(f) != item['sha256']):
            raise ValueError(f"重排模型文件校验失败：{item['name']}")
    return fixture, fusion, manifest, helper


def load_local_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(THREADS)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIRECTORY), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIRECTORY), local_files_only=True, trust_remote_code=False, use_safetensors=True)
    model = model.to(device='cpu', dtype=torch.float32).eval()
    if model.config.num_labels != 1:
        raise ValueError('重排器必须对每个配对输出一个logit')
    return torch, tokenizer, model


def rank_scores(ids, scores):
    if len(ids) != len(scores) or len(set(ids)) != len(ids) or any(not math.isfinite(v) for v in scores):
        raise ValueError('分数数量、候选ID或数值无效；不重试或回退')
    return [{'rank': i, 'doc_id': doc, 'score': score}
            for i, (doc, score) in enumerate(sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True), 1)]


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='只核验来源和权重文件，不加载模型、不前向、不写正式结果')
    args = parser.parse_args(argv)
    fixture, fusion, manifest, helper = load_materials()
    print('3-6神经重排：冻结融合候选，12篇英文文档、5条查询、27个查询—文档配对。')
    print('重排模型：', MODEL_ID, '；固定版本：', REVISION)
    print('CPU float32，线程4，最大长度512；按每条查询一批，预计5批本机交叉编码器前向。')
    print('生成式模型、云端和网络请求均0；不重新编码或召回，无下载、自动安装、回退或重试。')
    print('使用完整RRF候选并集（5/5/5/7/5篇），不提前截成5篇；重排后取Top-5。')
    if args.preview:
        print('RERANK_PREVIEW_ONLY：融合来源、核对记录和本地权重校验通过；模型前向0批。')
        return 0
    directory = ROOT / '运行结果' / ('neural_rerank_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'record.json'
    record = {'stage': 'chapter3_fixed_candidate_neural_rerank', 'status': 'running',
              'created_at': datetime.now(CHINA).isoformat(), 'script_sha256': sha(__file__),
              'fixture_sha256': sha(FIXTURE), 'sources': fixture, 'upstream_commit': fusion['upstream_commit'],
              'official_manifest_sha256': fusion['official_manifest_sha256'],
              'model': {'id': MODEL_ID, 'revision': REVISION, 'path': str(MODEL_DIRECTORY),
                        'manifest_sha256': sha(MODEL_MANIFEST)},
              'protocol': {'corpus_count': 12, 'query_count': 5, 'planned_pairs': 27, 'planned_forward_batches': 5,
                           'device': 'cpu', 'dtype': 'float32', 'threads': THREADS, 'seed': 42,
                           'max_length': MAX_LENGTH, 'padding': 'per-query longest',
                           'truncation': 'longest_first', 'model_inputs': 'query and original document text only; no labels, titles or IDs',
                           'candidate_pool': 'entire saved RRF-ranked union; no extra retrieval or pre-rerank top-5 cutoff',
                           'ties': 'stable sort preserves saved RRF candidate order', 'score': 'raw sequence classification logit, not probability',
                           'top_k': K, 'precision_denominator': K, 'rr_scope': 'top 5'},
              'corpus': fusion['corpus'], 'labels': fusion['labels'],
              'baseline_macro': fusion['macro'], 'generative_model_requests': 0, 'cloud_requests': 0,
              'network_requests': 0, 'embedding_forward_batches': 0,
              'reranker_forward_batches_attempted': 0, 'reranker_forward_batches_completed': 0,
              'reranker_pairs_completed': 0, 'queries': [],
              'scope': 'Same-corpus fixed-candidate English MiniLM reranking; not BGE service reproduction or general benchmark.'}
    dump(path, record)
    try:
        start = time.perf_counter()
        torch, tokenizer, model = load_local_model()
        record['model_load_seconds'] = time.perf_counter() - start
        record['environment'] = {'python': sys.version.split()[0], 'torch': torch.__version__}
        import transformers
        record['environment']['transformers'] = transformers.__version__
        documents = {d['doc_id']: d['text'] for d in fusion['corpus']}
        record['model_parameter_count'] = sum(p.numel() for p in model.parameters())
        dump(path, record)
        for number, q in enumerate(fusion['queries'], 1):
            query = q['query']
            ids = [r['doc_id'] for r in q['fused_rankings']['rrf']]
            texts = [documents[d] for d in ids]
            print(f'\n查询{number}/5：{query}；候选{len(ids)}篇……', flush=True)
            lengths = [len(tokenizer(query, text, truncation=False)['input_ids']) for text in texts]
            enc = tokenizer([query] * len(texts), texts, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors='pt')
            row = {'query': query, 'candidate_ids_in_model_order': ids,
                   'candidate_text_sha256': {doc: hashlib.sha256(documents[doc].encode()).hexdigest() for doc in ids},
                   'pair_lengths_before_truncation': lengths,
                   'truncated_pairs': sum(v > MAX_LENGTH for v in lengths),
                   'input_tensors': {name: tensor.tolist() for name, tensor in enc.items()},
                   'status': 'forward_pending'}
            record['queries'].append(row)
            record['reranker_forward_batches_attempted'] += 1
            row['status'] = 'forward_started'
            dump(path, record)
            start = time.perf_counter()
            with torch.no_grad():
                logits = model(**enc).logits
            row['forward_seconds'] = time.perf_counter() - start
            record['reranker_forward_batches_completed'] += 1
            if tuple(logits.shape) != (len(ids), 1):
                raise ValueError(f'意外logits形状：{tuple(logits.shape)}')
            scores = logits[:, 0].float().tolist()
            if not all(math.isfinite(v) for v in scores):
                row['nonfinite_logits_repr'] = [repr(v) for v in scores]
                raise ValueError('模型产生非有限分数；不回退或重试')
            row['raw_logits'] = scores
            ranking = rank_scores(ids, scores)
            # 评分和排序完成后才读取相关标注计算评测指标。
            relevant = fusion['labels'][query]
            row.update(reranked_full_ranking=ranking, relevant_doc_ids=relevant,
                       rerank_metrics=helper.metrics([r['doc_id'] for r in ranking], relevant),
                       baseline_metrics=q['metrics'], candidate_recall=q['candidate_recall'],
                       rank_changes=[{'doc_id': doc,
                                      'dense': next((r['rank'] for r in q['source_rankings']['dense'] if r['doc_id'] == doc), None),
                                      'rrf': next(r['rank'] for r in q['fused_rankings']['rrf'] if r['doc_id'] == doc),
                                      'weighted': next(r['rank'] for r in q['fused_rankings']['weighted'] if r['doc_id'] == doc),
                                      'rerank': next(r['rank'] for r in ranking if r['doc_id'] == doc)} for doc in ids],
                       status='scored')
            record['reranker_pairs_completed'] += len(ids)
            dump(path, record)
            print('  RRF前5：', [r['doc_id'] for r in q['fused_rankings']['rrf'][:K]])
            print('  重排前5：', [r['doc_id'] for r in ranking[:K]])
            print('  标注相关：', relevant, '；重排指标：', row['rerank_metrics'])
            if query == 'training neural networks':
                print('  候选原始logit（不是概率）：', dict(zip(ids, scores)))
                print('  排名变化：', row['rank_changes'])
        record['rerank_macro'] = helper.macro([q['rerank_metrics'] for q in record['queries']])
        record['truncated_pairs_total'] = sum(q['truncated_pairs'] for q in record['queries'])
        record['forward_seconds_total'] = sum(q['forward_seconds'] for q in record['queries'])
        record['status'] = 'neural_rerank_saved'
        dump(path, record)
        print('\n保存的融合基线：', record['baseline_macro'])
        print('神经重排宏平均：', record['rerank_macro'])
        print('前向完成：', record['reranker_forward_batches_completed'], '批；配对：', record['reranker_pairs_completed'],
              '；发生截断的配对：', record['truncated_pairs_total'])
        print('NEURAL_RERANK_SAVED：仅本次重排保存完成，来源、分数、排名和指标待核对。')
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

"""只读12页个人Wiki，冻结三条真实问题与分块，再运行BM25关键词基线。"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'测试材料/检索方案.json'
DEFAULT_VAULT = Path('/Users/zhuo/Agent相关/知识库/我的Wiki')
CHINA = timezone(timedelta(hours=8))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def source_path(vault, relative):
    root = vault.resolve()
    candidate = Path(relative)
    path = (root/candidate).resolve()
    if (candidate.is_absolute() or not path.is_relative_to(root/'wiki') or path.suffix != '.md'
            or path.relative_to(root/'wiki').parts[0] not in {'sources', 'concepts', 'entities', 'synthesis'}
            or not candidate.parts or candidate.parts[0] != 'wiki'
            or len(candidate.parts) < 3 or candidate.parts[1] not in {'sources', 'concepts', 'entities', 'synthesis'}):
        raise ValueError('仅允许指定Wiki中的知识页面，不访问raw、系统页或越界路径')
    return path


def tokenize(text):
    tokens = []
    for item in re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]+', text.casefold()):
        if re.fullmatch(r'[a-z0-9_]+', item):
            tokens.append(item)
        else:
            tokens.extend(item)
            tokens.extend(item[i:i+2] for i in range(len(item)-1))
    return tokens


def normalize_query(text):
    # 统一去掉请求措辞；原句仍保留，后续各路检索复用这里保存的文本。
    text = re.sub(r'帮我|请问|解释|什么|一下|请', ' ', text)
    text = re.sub(r'^\s*是|是\s*$', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def split_document(document_id, relative, text, size, overlap):
    if not 0 <= overlap < size:
        raise ValueError('分块重叠必须小于分块大小')
    frontmatter = re.match(r'\A\ufeff?---\r?\n.*?\r?\n---(?:\r?\n|$)', text, re.DOTALL)
    begin = frontmatter.end() if frontmatter else 0
    body = text[begin:]
    chunks = []
    for start in range(0, len(body), size-overlap):
        end = min(start+size, len(body))
        content = body[start:end]
        if content.strip():
            absolute_start, absolute_end = begin+start, begin+end
            chunks.append({'id': f'{document_id}:C{len(chunks)+1:03d}', 'document_id': document_id,
                'path': relative, 'source_char_start': absolute_start, 'source_char_end': absolute_end,
                'source_line_start': text[:absolute_start].count('\n')+1,
                'source_line_end': text[:max(absolute_start, absolute_end-1)].count('\n')+1,
                'text': content, 'text_sha256': digest(content.encode('utf-8'))})
        if end == len(body):
            break
    return chunks


def rank_bm25(chunks, queries, k1, b, top_k):
    counts = [Counter(tokenize(chunk['text'])) for chunk in chunks]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths)/len(lengths) if lengths else 0
    df = Counter(term for count in counts for term in count)
    rows = []
    for query in queries:
        retrieval_query = normalize_query(query['text'])
        terms = set(tokenize(retrieval_query))
        ranked = []
        for chunk, count, length in zip(chunks, counts, lengths):
            score = 0.0
            for term in sorted(terms):
                tf = count[term]
                if tf and average:
                    idf = math.log(1 + (len(chunks)-df[term]+0.5)/(df[term]+0.5))
                    score += idf*tf*(k1+1)/(tf+k1*(1-b+b*length/average))
            if score > 0:
                ranked.append({'chunk_id': chunk['id'], 'document_id': chunk['document_id'],
                    'path': chunk['path'], 'source_line_start': chunk['source_line_start'],
                    'source_line_end': chunk['source_line_end'], 'score': score, 'text': chunk['text']})
        ranked.sort(key=lambda item: (-item['score'], item['chunk_id']))
        rows.append({'query_id': query['id'], 'query': query['text'], 'retrieval_query': retrieval_query, 'hits': ranked[:top_k],
                     'positive_score_candidates': len(ranked), 'relevance_review': 'pending'})
    return rows


def load_plan():
    raw = PLAN.read_bytes()
    plan = json.loads(raw)
    docs, queries = plan['documents'], plan['queries']
    if (len(docs) != 12 or len(queries) != 3 or len({d['id'] for d in docs}) != 12
            or len({d['path'] for d in docs}) != 12 or len({q['id'] for q in queries}) != 3
            or any(not isinstance(q['text'], str) or not q['text'].strip() for q in queries)):
        raise ValueError('本步固定12页、3个非空问题，ID和页面路径须唯一')
    if plan['relevance_labels'] is not None:
        raise ValueError('当前尚未冻结相关性标注，不在本步使用或生成金标')
    return plan, digest(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vault', type=Path, default=DEFAULT_VAULT)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    plan, plan_sha = load_plan()
    paths = [(d, source_path(args.vault, d['path'])) for d in plan['documents']]
    print('个人Wiki第一步：12页固定语料＋3条真实问题，运行关键词BM25。')
    print('0次嵌入前向、生成模型或网络请求；原Wiki只读，私有快照与摘录仅保存在本地忽略目录。')
    if args.preview:
        missing = [d['path'] for d, path in paths if not path.is_file()]
        print(json.dumps({'questions': plan['queries'], 'planned_pages': len(paths), 'missing_pages': missing}, ensure_ascii=False, indent=2))
        print('WIKI_SPARSE_PREVIEW_ONLY：未读取页面正文、未执行检索、无结果目录。')
        return 1 if missing else 0
    folder = ROOT/'本地运行结果'/('wiki_sparse_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    output = folder/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
        'script_sha256': digest(Path(__file__).read_bytes()), 'plan_sha256': plan_sha,
        'vault_root': str(args.vault.resolve()), 'plan': plan,
        'embedding_forward_batches': 0, 'generation_model_requests': 0, 'network_requests': 0,
        'scope': 'Twelve selected Wiki pages, frozen chunks and sparse baseline only; not full hybrid RAG or answer generation; relevance unjudged.'}
    dump(output, record)
    print('本地记录路径：' + str(output), flush=True)
    try:
        documents, chunks = [], []
        for item, path in paths:
            raw = path.read_bytes()
            text = raw.decode('utf-8')
            if not text.strip() or '\x00' in text:
                raise ValueError('页面为空或不是纯文本：' + item['path'])
            parts = split_document(item['id'], item['path'], text,
                                   plan['chunking']['characters'], plan['chunking']['overlap_characters'])
            if not parts:
                raise ValueError('页面正文为空：' + item['path'])
            documents.append({**item, 'sha256': digest(raw), 'bytes': len(raw), 'text': text, 'chunk_count': len(parts)})
            chunks.extend(parts)
        rows = rank_bm25(chunks, plan['queries'], plan['sparse']['k1'], plan['sparse']['b'], plan['sparse']['top_k'])
        unchanged = all(digest(path.read_bytes()) == document['sha256']
                        for (_, path), document in zip(paths, documents))
        record['checks'] = {'all_twelve_pages_nonempty': len(documents) == 12,
            'source_pages_unchanged': unchanged,
            'every_chunk_matches_source_span': all(
                next(d['text'] for d in documents if d['id'] == chunk['document_id'])[
                    chunk['source_char_start']:chunk['source_char_end']] == chunk['text'] for chunk in chunks),
            'three_queries_processed': len(rows) == 3,
            'all_returned_scores_positive_and_finite': all(math.isfinite(h['score']) and h['score'] > 0 for row in rows for h in row['hits'])}
        corpus = {'documents': documents, 'chunks': chunks}
        dump(folder/'corpus.json', corpus)
        record.update(document_count=len(documents), chunk_count=len(chunks),
                      corpus_sha256=digest((folder/'corpus.json').read_bytes()), results=rows,
                      relevance_labels=None, metrics=None)
        record['status'] = 'sparse_baseline_saved_relevance_review_pending' if all(record['checks'].values()) else 'failed_checks'
        print(f'已固定：{len(documents)}页，{len(chunks)}块，3条问题。')
        for row in rows:
            print('\n' + row['query_id'] + '：' + row['query'])
            print('  检索文本：' + row['retrieval_query'])
            if not row['hits']:
                print('  无正分匹配；不补造结果。')
            for rank, hit in enumerate(row['hits'], 1):
                print(f'  {rank}. {hit["path"]} 第{hit["source_line_start"]}—{hit["source_line_end"]}行，分数{hit["score"]:.4f}')
                print('     ' + re.sub(r'\s+', ' ', hit['text'])[:160])
        print('\n技术检查：' + json.dumps(record['checks'], ensure_ascii=False))
        print('WIKI_SPARSE_SAVED：仅关键词基线；相关性待核对，尚未运行稠密、融合或回答生成。')
        return 0 if all(record['checks'].values()) else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('本步未完成，不重试：' + str(exc))
        return 1
    finally:
        dump(output, record)
        print('本地记录路径：' + str(output), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

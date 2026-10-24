"""核对Windows迁移后的保存结果；历史证据只读。"""
from pathlib import Path
import argparse, collections, hashlib, json, math, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import project_path

parser = argparse.ArgumentParser(description='只读核验历史记录、哈希和指标；不恢复批次或请求模型。')
parser.add_argument('--output', type=Path, help='可选：将核验结果写入指定文件；默认只打印')
args = parser.parse_args()
checks = []
def check(name, condition, detail=None):
    checks.append({'check': name, 'passed': bool(condition), 'detail': detail})
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def find(pattern):
    return next(ROOT.glob(pattern))
def relocate(s):
    return project_path(s)
def metrics(ids, relevant):
    top = ids[:5]
    hits = sum(x in relevant for x in top)
    first = next((i for i,x in enumerate(top,1) if x in relevant), None)
    return np.array([hits/len(relevant), hits/5, 1/first if first else 0.0])
def metric_values(d):
    return [d['recall_at_5'],d['precision_at_5_fixed_slots'],d['mrr_at_5']]

for pattern in ('第三章/3.1用户记忆评估/运行结果/official_batch_*/batch.json',
                '第三章/3.2记忆格式对照/运行结果/four_mode_campaign_*/batch.json'):
    p = find(pattern)
    batch = read(p); summary = read(p.parent/'summary.json')
    linked, bad, scores = 0, [], []
    for entry in batch['entries']:
        for stage in ('writer','judge'):
            item=entry[stage]
            if 'path' not in item: continue
            f=relocate(item['path']); linked+=1
            if not f.exists() or sha(f)!=item['sha256']: bad.append(str(f))
            if stage=='judge' and item['status']=='scored':
                result=item['result']; scores.append(result)
                jd=read(f)
                if jd['writer_record_sha256']!=entry['writer']['sha256']:
                    bad.append('judge-writer-link:'+entry['test_id'])
    wc=dict(collections.Counter(e['writer']['status'] for e in batch['entries']))
    jc=dict(collections.Counter(e['judge']['status'] for e in batch['entries']))
    passed=sum(s['passed'] for s in scores)
    check(p.parent.name+': linked record hashes',not bad,{'linked_records':linked,'problems':bad})
    check(p.parent.name+': state counts',wc==summary['writer_states'] and jc==summary['judge_states'],{'writer':wc,'judge':jc})
    expected_scored=summary.get('scored_cases',summary.get('scored_variants'))
    expected_passed=summary.get('passed_cases',summary.get('passed_variants'))
    expected_reward=summary.get('mean_reward_scored_cases_only',summary.get('mean_reward_scored_only'))
    mean_reward=sum(s['reward'] for s in scores)/len(scores)
    check(p.parent.name+': aggregate scores',len(scores)==expected_scored and passed==expected_passed and math.isclose(mean_reward,expected_reward),
          {'scored':len(scores),'passed':passed,'mean_reward':mean_reward})

p=find('第三章/3.4稠密向量检索/运行结果/synthetic_ann_*/record.json');d=read(p)
arrays={k:np.load(p.parent/v['file'],allow_pickle=False) for k,v in d['arrays'].items()}
check('ANN arrays SHA256',all(sha(p.parent/v['file'])==v['sha256'] for v in d['arrays'].values()))
exact=[[f'v{i}' for i in np.argsort(-(arrays['base']@q))[:10]] for q in arrays['queries']]
check('ANN exact top 10 recomputed',exact==d['exact_top_ids'])
for name,result in d['results'].items():
    values=[len(set(row['ids']) & set(exact[i]))/10 for i,row in enumerate(result['queries'])]
    check('ANN recall recomputed: '+name,math.isclose(float(np.mean(values)),result['recall_at_k']),{'recall':float(np.mean(values))})

p=find('第三章/3.4稠密向量检索/运行结果/text_embedding_*/record.json');d=read(p)
vectors={k:np.load(p.parent/v['file'],allow_pickle=False) for k,v in d['arrays'].items()}
check('text embedding arrays SHA256',all(sha(p.parent/v['file'])==v['sha256'] for v in d['arrays'].values()))
ids=[x['doc_id'] for x in d['corpus']];rows=[];matches=True
for q,old in zip(vectors['queries'],d['exact']['queries']):
    ranked=[ids[i] for i in np.argsort(-(vectors['documents']@q))]
    matches &= ranked==[x['doc_id'] for x in old['full_ranking']]
    rows.append(metrics(ranked,old['relevant_doc_ids']))
check('text embedding all 5 full rankings recomputed',matches)
check('text embedding macro metrics',np.allclose(np.mean(rows,axis=0),[1,.32,1]),np.mean(rows,axis=0).tolist())

p=find('第三章/3.5BM25稀疏检索/运行结果/*/record.json');d=read(p)
for name,key in [('bm25','full_matched_ranking'),('dense_saved','dense_top_5')]:
    rows=[]
    for q in d['queries']:
        ranked=[x['doc_id'] for x in q[key]] if name=='bm25' else q[key]
        rows.append(metrics(ranked,q['relevant_doc_ids']))
    macro=np.mean(rows,axis=0)
    check('BM25 comparison macro: '+name,np.allclose(macro,metric_values(d['common_macro'][name])),macro.tolist())

p=find('第三章/3.6混合检索/运行结果/hybrid_fusion_*/record.json');d=read(p)
for name in ('dense','sparse','rrf','weighted'):
    rows=[]
    for q in d['queries']:
        ranking=q['source_rankings'][name] if name in ('dense','sparse') else q['fused_rankings'][name]
        rows.append(metrics([x['doc_id'] for x in ranking],q['relevant_doc_ids']))
    macro=np.mean(rows,axis=0)
    check('fusion metrics: '+name,np.allclose(macro,metric_values(d['macro'][name])),macro.tolist())

p=find('第三章/3.6混合检索/运行结果/neural_rerank_*/record.json');d=read(p);rows=[];matches=True;pairs=0
for q in d['queries']:
    ranked=[i for i,_ in sorted(zip(q['candidate_ids_in_model_order'],q['raw_logits']),key=lambda x:x[1],reverse=True)]
    matches &= ranked==[x['doc_id'] for x in q['reranked_full_ranking']]
    pairs+=len(ranked);rows.append(metrics(ranked,q['relevant_doc_ids']))
check('rerank rankings reconstructed from logits',matches,{'pairs':pairs,'queries':len(rows)})
macro=np.mean(rows,axis=0)
check('rerank macro metrics',np.allclose(macro,metric_values(d['rerank_macro'])),macro.tolist())

for p in ROOT.glob('第三章/3.7结构化索引/运行结果/*/record.json'):
    d=read(p);files=list(p.parent.iterdir());verified=[]
    for key in ('artifact_sha256','reviewed_artifact_sha256'):
        if key in d:
            matching=[f.name for f in files if f.is_file() and sha(f)==d[key]]
            check(p.parent.name+': '+key,bool(matching),matching)

result={'project':str(ROOT),'checks':checks,'passed':sum(x['passed'] for x in checks),'failed':sum(not x['passed'] for x in checks)}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

raise SystemExit(1 if result['failed'] else 0)

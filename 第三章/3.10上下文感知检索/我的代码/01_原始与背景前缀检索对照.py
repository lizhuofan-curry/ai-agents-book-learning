"""冻结22块/15题：同一BM25参数对比原文与历史前缀+原文，不调用模型。"""
import argparse
import ast
from collections import Counter
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'官方资料/upstream'
PROJECT=UP/'chapter3/contextual-retrieval'
MANIFEST=ROOT/'官方资料/manifest.json'
EVIDENCE=PROJECT/'validation/runs/20260729T205807Z-3_10-35531780/evidence.json'
CHINA=timezone(timedelta(hours=8))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    for item in manifest['files']:
        p=(UP/item['path']).resolve()
        if not p.is_relative_to(UP.resolve()) or sha(p)!=item['sha256']:
            raise ValueError('冻结来源校验失败：'+item['path'])
    history=json.loads(EVIDENCE.read_text(encoding='utf-8'))
    chunks=history['chunks']
    queries=json.loads((PROJECT/'evaluation/retrieval_eval.json').read_text(encoding='utf-8'))['queries']
    ids=[c['chunk_id'] for c in chunks]
    if len(chunks)!=22 or len(set(ids))!=22 or len(queries)!=15:
        raise ValueError('固定语料规模变化')
    for chunk in chunks:
        if not chunk['plain'].strip() or not chunk['prefix'].strip() or chunk['contextual']!=chunk['prefix']+'\n\n'+chunk['plain']:
            raise ValueError('原文/前缀拼接不一致')
    if len({q['id'] for q in queries})!=15 or any(q['gold_chunk_id'] not in ids for q in queries):
        raise ValueError('题目ID或目标片段异常')
    # 仅加载官方无jieba分词函数，不导入numpy/rank_bm25或任何服务。
    tree=ast.parse((PROJECT/'compare_retrieval.py').read_text(encoding='utf-8'))
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tokenize']
    if len(definitions)!=1:raise ValueError('分词定义异常')
    namespace={'List':list,'_HAS_JIEBA':False}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'frozen_tokenizer','exec'),namespace)
    tokenize=lambda text:namespace['tokenize'](text,use_jieba=False)
    return chunks,queries,manifest,tokenize


class BM25:
    """平滑非负IDF的本地实现；不同于官方rank_bm25实现，不移用其数值。"""
    def __init__(self,chunks,field,tokenize):
        self.ids=[c['chunk_id'] for c in chunks]
        self.tokenize=tokenize
        self.tfs=[Counter(tokenize(c[field])) for c in chunks]
        self.lengths=[sum(tf.values()) for tf in self.tfs]
        self.avg=sum(self.lengths)/len(self.lengths)
        self.df=Counter(t for tf in self.tfs for t in tf)

    def rank(self,query):
        qt=Counter(self.tokenize(query));rows=[]
        for i,tf in enumerate(self.tfs):
            score=0.0
            for t,count in qt.items():
                freq=tf[t]
                if freq:
                    idf=math.log(1+(len(self.ids)-self.df[t]+.5)/(self.df[t]+.5))
                    score+=count*idf*freq*2.5/(freq+1.5*(.25+.75*self.lengths[i]/(self.avg or 1)))
            rows.append({'chunk_id':self.ids[i],'score':score})
        # 全部片段均保留；同分按稳定ID次序，防止两组不一致。
        return sorted(rows,key=lambda r:(-r['score'],r['chunk_id']))


def evaluate(ranking,gold):
    rank=next(i+1 for i,r in enumerate(ranking) if r['chunk_id']==gold)
    return {'gold_rank':rank,'reciprocal_rank':1/rank,
            'hits':{str(k):rank<=k for k in (1,3,5)}}


def metrics(rows,mode):
    return {'n':len(rows),'recall_at_k':{str(k):sum(r[mode]['hits'][str(k)] for r in rows)/len(rows)
                                       for k in (1,3,5)},
            'mrr':sum(r[mode]['reciprocal_rank'] for r in rows)/len(rows)}


def reading(chunks,rows):
    by_id={c['chunk_id']:c for c in chunks}
    lines=['# 背景前缀检索对照阅读材料\n',
           '前缀来自官方历史模型输出，本次没有重新生成或逐条认证其正确性。\n']
    for row in rows:
        chunk=by_id[row['gold_chunk_id']]
        lines.extend(['## '+row['id']+' · '+row['query']+'\n',
            f"目标{row['gold_chunk_id']}：原文排名{row['plain']['gold_rank']}，加前缀排名{row['contextual']['gold_rank']}。\n",
            '### 历史背景前缀\n',chunk['prefix']+'\n','### 目标原文\n',chunk['plain']+'\n'])
    lines.append('## 全部22份前缀\n')
    for c in chunks:lines.append('### '+c['chunk_id']+'\n\n'+c['prefix']+'\n')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    chunks,queries,manifest,tokenize=load_sources()
    manifest_hash=sha(MANIFEST)
    plain=BM25(chunks,'plain',tokenize);contextual=BM25(chunks,'contextual',tokenize)
    print('3-10第一步：22个固定片段、15个问题，原文 vs 官方历史前缀+原文。')
    print('固定字符/双字分词、平滑非负IDF BM25；两组参数相同，与官方jieba/rank_bm25数字不直接比较。')
    print('本次模型/嵌入/网络请求均0，不生成新前缀或回答。')
    if args.preview:
        print('CONTEXTUAL_PREVIEW_ONLY：哈希、拼接、目标和双索引校验通过；未执行正式15题查询。')
        return 0
    rows=[]
    for q in queries:
        a=plain.rank(q['query']);b=contextual.rank(q['query'])
        row={'id':q['id'],'query':q['query'],'gold_chunk_id':q['gold_chunk_id'],
             'plain':{**evaluate(a,q['gold_chunk_id']),'ranking':a},
             'contextual':{**evaluate(b,q['gold_chunk_id']),'ranking':b}}
        rows.append(row)
        print(q['id'],q['query'],'目标排名：',row['plain']['gold_rank'],'→',row['contextual']['gold_rank'])
    summary={mode:metrics(rows,mode) for mode in ('plain','contextual')}
    counts={'improved':sum(r['contextual']['gold_rank']<r['plain']['gold_rank'] for r in rows),
            'unchanged':sum(r['contextual']['gold_rank']==r['plain']['gold_rank'] for r in rows),
            'worsened':sum(r['contextual']['gold_rank']>r['plain']['gold_rank'] for r in rows)}
    print('\n指标：',json.dumps(summary,ensure_ascii=False,indent=2))
    print('逐题排名变化：',counts)
    if sha(MANIFEST)!=manifest_hash:
        raise ValueError('运行期间来源清单变化')
    load_sources()
    directory=ROOT/'运行结果'/('contextual_compare_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=False)
    record={'status':'contextual_comparison_saved_review_pending','created_at':datetime.now(CHINA).isoformat(),
        'script_sha256':sha(Path(__file__)),'manifest_sha256':sha(MANIFEST),'manifest':manifest,
        'chunks':chunks,'queries':queries,'results':rows,'summary':summary,'rank_changes':counts,
        'configuration':{'tokenizer':'official char fallback; no jieba','k1':1.5,'b':.75,
                         'idf':'log(1+(N-df+0.5)/(df+0.5))','ties':'chunk_id ascending'},
        'model_requests':0,'embedding_requests':0,'network_requests':0,
        'scope':'offline historical-prefix BM25 comparison, not live prefix generation or canonical reproduction',
        'prefix_semantic_review':'pending; no automatic claim that source supports every generated prefix'}
    (directory/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (directory/'前缀与原文对照.md').write_text(reading(chunks,rows),encoding='utf-8')
    print('\nCONTEXTUAL_RETRIEVAL_COMPARISON_SAVED：完整排名、前缀和原文已保存，待核对。')
    print('记录路径：',directory/'record.json')
    print('阅读材料：',directory/'前缀与原文对照.md')
    return 0


if __name__=='__main__':raise SystemExit(main())

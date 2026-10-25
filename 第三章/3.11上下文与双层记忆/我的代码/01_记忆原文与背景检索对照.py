"""官方受控12块/8题的离线对照，不生成前缀/卡片或回答。"""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'官方资料/upstream'
OFFICIAL=UP/'chapter3/contextual-retrieval-for-user-memory'
MANIFEST=ROOT/'官方资料/manifest.json'
CHINA=timezone(timedelta(hours=8))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=(UP/item['path']).resolve()
        if not path.is_relative_to(UP.resolve()) or sha(path)!=item['sha256']:
            raise ValueError('冻结资料哈希不符')
    data=json.loads((OFFICIAL/'memory_qa_eval.json').read_text(encoding='utf-8'))
    validate_dataset(data)
    spec=importlib.util.spec_from_file_location('official_memory_context',OFFICIAL/'contextual_compare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,data,manifest


def validate_dataset(data):
    chunks,queries=data['chunks'],data['queries'];ids=[c['id'] for c in chunks]
    if len(chunks)!=12 or len(queries)!=8 or len(set(ids))!=12:raise ValueError('固定教学集规模或ID变化')
    if any(q['gold'] not in ids for q in queries):raise ValueError('目标片段不存在')
    if any(not isinstance(c.get(k),str) or not c[k].strip() for c in chunks for k in ('text','context')):
        raise ValueError('原文或前缀缺失')


def compare(module,data):
    chunks,queries=data['chunks'],data['queries'];rows=[];summaries={}
    for mode in ('plain','contextual'):
        metrics,details=module.evaluate(chunks,queries,mode)
        summaries[mode]=metrics
        docs=[c['text'] if mode=='plain' else c['context']+'\n'+c['text'] for c in chunks]
        index=module.BM25([module.tokenize(d) for d in docs])
        for i,(q,detail) in enumerate(zip(queries,details)):
            if mode=='plain':rows.append({'query_number':i+1,'query':q['q'],'gold':q['gold']})
            qt=module.tokenize(q['q']);order=index.rank(qt)
            ranking=[{'chunk_id':chunks[j]['id'],'score':index.score(qt,j)} for j in order]
            if detail['rank']!=next(i+1 for i,r in enumerate(ranking) if r['chunk_id']==q['gold']):
                raise ValueError('保存排名与官方指标不一致')
            rows[i][mode]={**detail,'ranking':ranking}
    return rows,summaries


def render(data,rows):
    by={c['id']:c for c in data['chunks']}
    lines=['# 用户记忆前缀对照阅读\n',
           '这是官方受控教学数据，不是用户真实记录。本步使用现有背景，不生成卡片或答案。\n',
           '部分前缀有短片段中未出现的信息；资料未附完整上文，不能仅凭排名提升认证这些背景的来源。\n']
    for row in rows:
        c=by[row['gold']]
        lines.extend(['## '+str(row['query_number'])+' '+row['query']+'\n',
            f"目标：{row['gold']}，排名{row['plain']['rank']}→{row['contextual']['rank']}。\n",
            '### 原始片段\n',c['text']+'\n','### 已提供背景\n',c['context']+'\n'])
    lines.append('## 全部片段\n')
    for c in data['chunks']:lines.append('### '+c['id']+'\n\n'+c['text']+'\n\n背景：'+c['context']+'\n')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    module,data,manifest=prepare();manifest_hash=sha(MANIFEST)
    print('3-11第一步：官方受控12个记忆块、8个问题；原文 vs 已提供背景+原文。')
    print('复用官方纯Python BM25与分词；模型/嵌入/网络请求均0，不生成卡片或新前缀。')
    print('部分背景超出短片段文字且缺完整上文；结果仅说明给定背景的词法检索作用。')
    if args.preview:
        print('MEMORY_CONTEXT_PREVIEW_ONLY：哈希、12块8题及目标检查通过，未执行正式查询。')
        return 0
    rows,summary=compare(module,data)
    for row in rows:print(row['query_number'],row['query'],'目标排名：',row['plain']['rank'],'→',row['contextual']['rank'])
    print('\n指标：',json.dumps(summary,ensure_ascii=False,indent=2))
    changes={name:sum(test(r['contextual']['rank'],r['plain']['rank']) for r in rows)
             for name,test in [('improved',lambda a,b:a<b),('unchanged',lambda a,b:a==b),('worsened',lambda a,b:a>b)]}
    print('排名变化：',changes)
    if sha(MANIFEST)!=manifest_hash:raise ValueError('运行期间来源清单变化')
    prepare()
    directory=ROOT/'运行结果'/('memory_context_compare_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=False)
    record={'status':'memory_context_comparison_saved_review_pending','created_at':datetime.now(CHINA).isoformat(),
        'script_sha256':sha(Path(__file__)),'manifest_sha256':manifest_hash,'manifest':manifest,
        'dataset':data,'results':rows,'summary':summary,'rank_changes':changes,
        'model_requests':0,'embedding_requests':0,'network_requests':0,
        'configuration':{'bm25':'official pure Python','k1':1.5,'b':.75,'tokenizer':'official ASCII words and contiguous CJK unigrams/bigrams','ties':'original index ascending'},
        'scope':'controlled supplied-context lexical test; not live prefix generation, structured card extraction or dual-layer end-to-end evaluation',
        'context_provenance_limit':'Full preceding dialogues are not included; contextual facts cannot all be verified from short chunks.'}
    (directory/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (directory/'记忆片段与背景对照.md').write_text(render(data,rows),encoding='utf-8')
    print('\nMEMORY_CONTEXT_COMPARISON_SAVED：完整排序和阅读材料已保存，待核对。')
    print('记录路径：',directory/'record.json')
    print('阅读材料：',directory/'记忆片段与背景对照.md')
    return 0


if __name__=='__main__':raise SystemExit(main())

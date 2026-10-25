"""离线三组：原文、历史前缀、仅chunk_12换成本机原始生成前缀。"""
import argparse
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREFIX_RECORD=ROOT/'运行结果/single_prefix_20261025_094745_877598/record.json'
PREFIX_SHA='201248a1b0caa8a5aa6e2da658dda5410d4afc4e1023417da39711e8dec2bdd3'
TARGET='宪法_chunk_12'


def load_step():
    spec=importlib.util.spec_from_file_location('single_replacement_baseline',ROOT/'我的代码/01_原始与背景前缀检索对照.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def replace_one(chunks,target,prefix):
    if not isinstance(prefix,str) or not prefix.strip():raise ValueError('前缀为空')
    if sum(c['chunk_id']==target for c in chunks)!=1:raise ValueError('目标必须唯一')
    result=deepcopy(chunks)
    for chunk in result:
        if chunk['chunk_id']==target:
            chunk['prefix']=prefix
            chunk['contextual']=prefix+'\n\n'+chunk['plain']
    return result


def prepare():
    h=load_step();chunks,queries,manifest,tokenize=h.load_sources()
    if h.sha(PREFIX_RECORD)!=PREFIX_SHA:raise ValueError('固定模型生成记录变化')
    generated=json.loads(PREFIX_RECORD.read_text(encoding='utf-8'))
    if (generated['status']!='single_prefix_saved_semantic_review_pending'
            or generated['target_chunk_id']!=TARGET
            or generated['manifest_sha256']!=h.sha(h.MANIFEST)
            or generated['comparison_script_sha256']!=h.sha(Path(h.__file__))):
        raise ValueError('前缀状态、目标或基线来源变化')
    original=next(c for c in chunks if c['chunk_id']==TARGET)
    if generated['target_plain']!=original['plain']:raise ValueError('目标原文不一致')
    audit=json.loads(PREFIX_RECORD.with_name('核对记录.json').read_text(encoding='utf-8'))
    if audit['record_sha256']!=PREFIX_SHA:raise ValueError('生成核对记录不符')
    changed=replace_one(chunks,TARGET,generated['candidate']['prefix'])
    if sum(a!=b for a,b in zip(chunks,changed))!=1:raise ValueError('变化不止一块')
    return h,chunks,changed,queries,tokenize,manifest,generated


def compare(h,chunks,changed,queries,tokenize):
    indexes={'plain':h.BM25(chunks,'plain',tokenize),
             'historical':h.BM25(chunks,'contextual',tokenize),
             'single_new':h.BM25(changed,'contextual',tokenize)}
    rows=[]
    for q in queries:
        row={'id':q['id'],'query':q['query'],'gold_chunk_id':q['gold_chunk_id']}
        for mode,index in indexes.items():
            ranking=index.rank(q['query'])
            row[mode]={**h.evaluate(ranking,q['gold_chunk_id']),'ranking':ranking}
        rows.append(row)
    summary={mode:h.metrics(rows,mode) for mode in indexes}
    change={key:sum(test(r['single_new']['gold_rank'],r['historical']['gold_rank']) for r in rows)
            for key,test in [('improved',lambda a,b:a<b),('unchanged',lambda a,b:a==b),('worsened',lambda a,b:a>b)]}
    return rows,summary,change


def reading(old,new,rows):
    lines=['# 单块前缀替换检索对照\n','## 官方历史前缀\n',old['prefix']+'\n',
        '## 本机原始生成前缀\n',new['prefix']+'\n',
        '保留范围限定不足的原始输出，不使用助手修订版本。\n',
        '| 问题 | 原文排名 | 历史前缀排名 | 单块新前缀排名 |\n| --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['id']} {row['query']} | {row['plain']['gold_rank']} | {row['historical']['gold_rank']} | {row['single_new']['gold_rank']} |")
    lines.extend(['\n## 目标原文\n',old['plain'],
        '\n单块文本变化会改变整个索引的平均长度和词频统计，其他问题的排名也可能变化。排名提高不等于前缀语义通过。\n'])
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    h,chunks,changed,queries,tokenize,manifest,generated=prepare()
    print('3-10第三步：原文 / 历史前缀 / 仅一块替换成本机生成前缀；同一22块15题。')
    print('目标：',TARGET,'；其余21块、目标原文及检索参数不变。')
    print('保留模型原始前缀及已记录的范围问题；模型/嵌入/网络请求均0。')
    if args.preview:
        print('SINGLE_PREFIX_COMPARE_PREVIEW_ONLY：来源、固定生成记录和单块变化校验通过，未执行正式15题。')
        return 0
    rows,summary,change=compare(h,chunks,changed,queries,tokenize)
    for row in rows:
        print(row['id'],row['query'],'排名：',row['plain']['gold_rank'],'/',row['historical']['gold_rank'],'/',row['single_new']['gold_rank'])
    print('\n指标：',json.dumps(summary,ensure_ascii=False,indent=2))
    print('单块新前缀相对历史前缀：',change)
    if h.sha(PREFIX_RECORD)!=PREFIX_SHA or h.sha(h.MANIFEST)!=generated['manifest_sha256']:
        raise ValueError('运行期间来源变化')
    h.load_sources()
    directory=ROOT/'运行结果'/('single_prefix_compare_'+datetime.now(h.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=False)
    record={'status':'single_prefix_comparison_saved_review_pending',
        'created_at':datetime.now(h.CHINA).isoformat(),'script_sha256':h.sha(Path(__file__)),
        'comparison_script_sha256':h.sha(Path(h.__file__)),'manifest_sha256':h.sha(h.MANIFEST),
        'prefix_record_sha256':PREFIX_SHA,'prefix_record_relative_path':PREFIX_RECORD.relative_to(ROOT).as_posix(),
        'target_chunk_id':TARGET,'historical_chunks':chunks,'single_new_chunks':changed,
        'queries':queries,'results':rows,'summary':summary,'rank_changes_vs_historical':change,
        'model_requests':0,'embedding_requests':0,'network_requests':0,
        'configuration':{'tokenizer':'official char fallback','k1':1.5,'b':.75,
                         'idf':'log(1+(N-df+0.5)/(df+0.5))','ties':'chunk_id ascending'},
        'scope':'post-hoc single selected chunk replacement; other 21 prefixes historical, not full local model prefix generation',
        'semantic_findings_retained':'Prefix does not qualify partial article 89; quotes cover State Council only. Not corrected for this run.'}
    (directory/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    old=next(c for c in chunks if c['chunk_id']==TARGET);new=next(c for c in changed if c['chunk_id']==TARGET)
    (directory/'单块前缀替换对照.md').write_text(reading(old,new,rows),encoding='utf-8')
    print('\nSINGLE_PREFIX_RETRIEVAL_COMPARISON_SAVED：完整三组排名已保存，待核对。')
    print('记录路径：',directory/'record.json')
    return 0


if __name__=='__main__':raise SystemExit(main())

"""已核对六块上的离线检索；车辆实体规则追查，不是模型自主决策。"""
import argparse
import ast
from collections import Counter
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
CHUNK_RECORD = ROOT/'运行结果/conversation_chunks_20261024_210405_183521/record.json'
CHUNK_SHA = 'b9de307389611e5b9b8d9ed0988afb6d1f44a6675dfa60b2d88350dbe038fc38'
MARKERS = {'confirmed_booking_marker': 'FS-447291', 'second_vehicle_marker': 'Model 3'}
TOP_K = 3


def helper():
    spec = importlib.util.spec_from_file_location('memory_reading_step', ROOT/'我的代码/01_阅读会话与固定轮次分块.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_definitions(path, names, namespace):
    """仅执行冻结源码中指定定义；不执行原模块import或服务初始化。"""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = []
    found = set()
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name = node.targets[0].id
        if name in names:
            selected.append(node)
            found.add(name)
    if found != set(names):
        raise ValueError('冻结源码定义不完整')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def index_text(chunk):
    # 只索引原消息，不加来源编号、测试标题、摘要、未来片段预览或人工标签。
    return '\n'.join(('Customer' if m['role']=='user' else 'Representative')+': '+m['content']
                     for m in chunk['messages'])


def prepare():
    h = helper()
    data, _ = h.load_case()
    if h.sha(CHUNK_RECORD) != CHUNK_SHA:
        raise ValueError('固定分块记录变化')
    record = json.loads(CHUNK_RECORD.read_text(encoding='utf-8'))
    if h.sha(Path(h.__file__)) != record['script_sha256'] or h.sha(h.MANIFEST)!=record['manifest_sha256']:
        raise ValueError('分块来源或代码变化')
    expected = [chunk for conv in data['conversation_histories']
                for chunk in h.split_conversation(conv,data['test_id'])]
    if expected != record['chunks']:
        raise ValueError('保存分块无法复现')
    ref = ROOT/'检索参考资料/indexer.py'
    meta = json.loads(ref.with_name('manifest.json').read_text())
    if h.sha(ref) != meta['sha256']:
        raise ValueError('BM25参考源码变化')
    manifest = json.loads(h.MANIFEST.read_text())
    demo = h.UPSTREAM/'chapter3/agentic-rag-for-user-memory/offline_demo.py'
    ns = extract_definitions(ref,{'_tokenize','LocalBM25Backend'},
        {'re':re,'math':math,'Counter':Counter,'List':List,'Dict':Dict,'Tuple':Tuple})
    rules = extract_definitions(demo,{'_ENTITY_PATTERN','_extract_vehicle_entities'},
        {'re':re,'List':List,'Dict':Dict})
    backend = ns['LocalBM25Backend']()
    chunks = {c['chunk_id']:c for c in record['chunks']}
    for cid,chunk in chunks.items():
        backend.add(cid,index_text(chunk))
    backend._build()
    return h,record,chunks,backend,rules['_extract_vehicle_entities'],manifest


def search(backend,chunks,query):
    return {'query':query,'top_k':TOP_K,'hits':[
        {'chunk_id':cid,'score':score,'conversation_id':chunks[cid]['conversation_id'],
         'session_timestamp':chunks[cid]['session_timestamp'],
         'source_message_numbers':chunks[cid]['source_message_numbers'],
         'text':index_text(chunks[cid])} for cid,score in backend.search(query,TOP_K)]}


def union_hits(steps):
    by_id = {}
    for step in steps:
        for hit in step['hits']:
            by_id.setdefault(hit['chunk_id'],hit)
    return list(by_id.values())


def coverage(hits):
    # 只测试字面证据标记，不宣称预约状态理解正确或可完整回答。
    present = {key:any(marker in hit['text'] for hit in hits) for key,marker in MARKERS.items()}
    return {'marker_presence':present,'marker_recall':sum(present.values())/len(present)}


def compare(backend,chunks,question,extract_entities):
    first = search(backend,chunks,question)
    entities = extract_entities([hit['text'] for hit in first['hits']],limit=4)
    steps = [first]
    for entity in entities:
        query = f'{entity} service appointment scheduled'
        steps.append(search(backend,chunks,query))
    merged = union_hits(steps)
    return {'naive':{'steps':[first],'evaluation':coverage(first['hits'])},
            'rule_multistep':{'entities_from_first_hits':entities,'steps':steps,
                             'union_hits':merged,'evaluation':coverage(merged)},
            'query_origin':'first query is original question; follow-ups use official regex and fixed template, no LLM'}


def render(result):
    lines = ['# 单次与规则多步检索：实际命中片段\n',
             '这是规则检索结果，不是模型生成的答案。标记命中不保证状态或时间理解正确。\n']
    for mode in ('naive','rule_multistep'):
        lines.append('## '+mode+'\n')
        for i,step in enumerate(result[mode]['steps'],1):
            lines.append(f"### 第{i}次查询：{step['query']}\n")
            for hit in step['hits']:
                lines.append(f"#### {hit['chunk_id']}\n\n会话时间：{hit['session_timestamp']}\n\n{hit['text']}\n")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    args = parser.parse_args()
    h,original,chunks,backend,extract_entities,manifest = prepare()
    print('3-9第二步：单次检索 vs 规则多步检索；6个已核对分块，每次Top-3。')
    print('追加查询来自官方车辆正则及模板，不是模型决策；模型/网络请求均0。')
    if args.preview:
        print('MEMORY_RETRIEVAL_PREVIEW_ONLY：来源、6块和本地索引校验通过，未执行正式查询。')
        return 0
    result = compare(backend,chunks,original['question'],extract_entities)
    for mode in ('naive','rule_multistep'):
        print('\n策略：',mode)
        if mode=='rule_multistep':print('首轮发现的车辆实体：',result[mode]['entities_from_first_hits'])
        for step in result[mode]['steps']:
            print(' 查询：',step['query'])
            for hit in step['hits']:
                print('  ',hit['chunk_id'],'分数',round(hit['score'],5))
        print('字面证据标记覆盖：',result[mode]['evaluation'])
    directory = ROOT/'运行结果'/('memory_retrieval_'+datetime.now(h.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=False)
    record = {'status':'memory_rule_retrieval_saved_review_pending',
        'created_at':datetime.now(h.CHINA).isoformat(),'script_sha256':h.sha(Path(__file__)),
        'chunk_record_sha256':h.sha(CHUNK_RECORD),'manifest_sha256':h.sha(h.MANIFEST),
        'bm25_source_sha256':h.sha(ROOT/'检索参考资料/indexer.py'),
        'question':original['question'],'indexed_texts':{cid:index_text(c) for cid,c in chunks.items()},
        'top_k':TOP_K,'markers_for_evaluation_only':MARKERS,'result':result,
        'model_requests':0,'network_requests':0,
        'scope':'single case, six chunks, raw message text only; no metadata enrichment or generated answer; not official full replication'}
    (directory/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (directory/'实际检索片段.md').write_text(render(result),encoding='utf-8')
    print('\nMEMORY_RULE_RETRIEVAL_SAVED：查询及完整片段已保存，待核对，不代表回答正确。')
    print('记录路径：',directory/'record.json')
    print('阅读材料：',directory/'实际检索片段.md')
    return 0


if __name__=='__main__':raise SystemExit(main())

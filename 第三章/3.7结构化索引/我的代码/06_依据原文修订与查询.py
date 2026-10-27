"""读取助手复核方案，另建可追溯修订视图并查询；不修改模型原始输出。"""
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'测试材料/助手复核修订方案.json'
FIXTURE=ROOT/'测试材料/Intel原文两段.json'
MANIFEST=ROOT/'官方资料/manifest.json'
UPSTREAM=ROOT/'官方资料/upstream'
BASELINE='real_source_build_20261018_223955_552003'
CHINA=timezone(timedelta(hours=8))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def matches_frozen_sha(path, expected):
    """验证原始字节，或Git仅将完整CRLF规范化成LF的副本；不改写文件。

    备用路径只尝试一种可逆表示：LF→CRLF，并且必须精确匹配原始
    SHA256。原始预期哈希不变，文字、空白或其他字节变化仍然失败。
    不用于上游源码或引文归一化，只用于原始JSON及其材料清单。
    """
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()==expected:
        return True
    if b'\r' not in data and b'\n' in data:
        return hashlib.sha256(data.replace(b'\n',b'\r\n')).hexdigest()==expected
    return False

def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('重复JSON字段：'+key)
        result[key]=value
    return result

def read(path):
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=unique)

def dump(path,obj):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)

def anchor(evidence,chunks):
    """Locate every occurrence of an exact normalized quote, retaining raw source offsets."""
    chunk=chunks[evidence['chunk_id']]
    quote=evidence['quote']
    if not isinstance(quote,str) or not quote.strip():
        raise ValueError('修订依据引句不能为空')
    raw=chunk['raw_text']
    chars=[]
    offsets=[]
    for word in re.finditer(r'\S+',raw):
        if chars:
            chars.append(' ')
            offsets.append(word.start()-1)
        chars.extend(word.group())
        offsets.extend(range(word.start(),word.end()))
    text=''.join(chars)
    citations=[]
    for match in re.finditer('(?='+re.escape(quote)+')',text):
        start=match.start()
        end=start+len(quote)
        a,b=offsets[start],offsets[end-1]+1
        citations.append({'chunk_id':chunk['chunk_id'],'quote':quote,
                          'normalized_char_start':start,'normalized_char_end':end,
                          'source_char_start':chunk['source_char_start']+a,
                          'source_char_end':chunk['source_char_start']+b,
                          'raw_source_quote':raw[a:b]})
    if not citations:
        raise ValueError('修订依据不是原文中的精确引句；停止，不补写')
    return citations

def attach_evidence(item,chunks):
    if not item.get('evidence'):
        raise ValueError('每处修订必须附来源')
    return [citation for evidence in item['evidence'] for citation in anchor(evidence,chunks)]

def load_materials():
    plan=read(PLAN)
    fixture=read(FIXTURE)
    manifest=read(MANIFEST)
    if (plan['kind']!='assistant_authored_review_plan' or plan['is_model_output'] is not False
        or plan['review_author']!='Codex assistant' or plan['user_review_status']!='not_yet_confirmed'):
        raise ValueError('本入口只读取明确标记为助手编写、尚未经用户逐条验收的修订方案')
    if plan['baseline_directory']!=BASELINE:
        raise ValueError('基线目录与本次固定原始运行不符')
    directory=ROOT/'运行结果'/BASELINE
    for filename,key in [('record.json','baseline_record_sha256'),
                         ('结构草稿.json','baseline_artifact_sha256'),('核对记录.json','baseline_audit_sha256')]:
        if not matches_frozen_sha(directory/filename,plan[key]):
            raise ValueError('原始基线或核对记录变化：'+filename)
    if (not matches_frozen_sha(FIXTURE,plan['fixture_sha256'])
        or not matches_frozen_sha(MANIFEST,plan['official_manifest_sha256'])):
        raise ValueError('冻结材料发生变化')
    for item in manifest['files']:
        path=(UPSTREAM/item['path']).resolve()
        if not path.is_relative_to(UPSTREAM.resolve()) or sha(path)!=item['sha256']:
            raise ValueError('官方文件校验失败：'+item['path'])
    source=(UPSTREAM/fixture['source_path']).resolve()
    if not source.is_relative_to(UPSTREAM.resolve()) or sha(source)!=fixture['source_sha256']:
        raise ValueError('原文校验失败')
    full=source.read_text(encoding='utf-8')
    chunks={c['chunk_id']:c for c in fixture['chunks']}
    for c in chunks.values():
        if (full[c['source_char_start']:c['source_char_end']]!=c['raw_text']
            or ' '.join(c['raw_text'].split())!=c['model_text']):
            raise ValueError('原文区间不一致')
    record=read(directory/'record.json')
    artifact=read(directory/'结构草稿.json')
    audit=read(directory/'核对记录.json')
    if (record['status']!='real_source_build_saved_review_pending'
        or audit['status']!='real_source_build_record_verified_with_semantic_findings'
        or audit['record_sha256']!=plan['baseline_record_sha256']
        or artifact['graph']!=record['graph'] or record['fixture']!=fixture):
        raise ValueError('基线、草稿及独立核对不一致')
    for item in plan['node_revisions']+plan['node_additions']+plan['reviewed_edges']+plan['reviewed_summary_claims']:
        attach_evidence(item,chunks)
    return plan,fixture,record,chunks

def baseline_view(record):
    graph=deepcopy(record['graph'])
    for i,edge in enumerate(graph['edges']):
        edge['id']='baseline_e'+str(i)
        edge['relation']=edge['type']
    graph['view']='original_model_graph'
    return graph

def build_reviewed(plan,record,chunks):
    nodes=deepcopy(record['graph']['nodes'])
    by_id={n['id']:n for n in nodes}
    edits=[]
    for revision in plan['node_revisions']:
        node=by_id[revision['id']]
        if any(key not in ('name','type') for key in revision['changes']):
            raise ValueError('节点修订只允许显式修改名字或类型')
        before={key:node[key] for key in revision['changes']}
        node.update(revision['changes'])
        node['label_review']={'author':plan['review_author'],'citations':attach_evidence(revision,chunks),
                              'reason':revision['reason'],'original_fields':before}
        edits.append({'operation':'revise_node','id':node['id'],'before':before,
                      'after':revision['changes'],'reason':revision['reason']})
    for addition in plan['node_additions']:
        if addition['id'] in by_id:
            raise ValueError('新增节点ID已存在')
        node={key:addition[key] for key in ('id','name','type','description')}
        node.update(authorship='assistant_editorial_addition; not model output',
                    citations=attach_evidence(addition,chunks),reason=addition['reason'])
        by_id[node['id']]=node
        nodes.append(node)
        edits.append({'operation':'add_node','id':node['id'],'name':node['name'],'reason':addition['reason']})
    edges=[]
    edge_ids=set()
    for reviewed in plan['reviewed_edges']:
        if (reviewed['id'] in edge_ids or reviewed['source_id'] not in by_id
            or reviewed['target_id'] not in by_id or not reviewed['review_basis'].strip()
            or reviewed['user_review_status']!='not_yet_confirmed'):
            raise ValueError('修订关系ID、端点或来源解释不符合要求')
        edge_ids.add(reviewed['id'])
        edges.append({**reviewed,'citations':attach_evidence(reviewed,chunks)})
    claims=[{**c,'citations':attach_evidence(c,chunks)} for c in plan['reviewed_summary_claims']]
    return {'view':'assistant_reviewed_proposal','directed':True,'multigraph':True,
            'authorship':'assistant editorial review; not model output',
            'user_review_status':'not_yet_confirmed','nodes':nodes,'edges':edges,
            'node_description_review':'Inherited model descriptions and quotes are kept for provenance; not all independently certified, not used for query answers.',
            'node_edits':edits,'summary':{'claims':claims,'children':record['parent_summary']['children'],
                'origin':'assistant-reviewed fixed single parent; no recursive model summarization or clustering'}}

def paths(graph,start,target,max_hops):
    if not 1<=max_hops<=4:
        raise ValueError('路径预算须为1到4跳')
    ids={n['id'] for n in graph['nodes']}
    if start not in ids or target not in ids:
        return []
    adjacency={n:[] for n in ids}
    for edge in graph['edges']:
        adjacency[edge['source_id']].append(edge)
    queue=deque([(start,[],{start})])
    found=[]
    while queue:
        node,route,seen=queue.popleft()
        if len(route)>=max_hops:
            continue
        for edge in adjacency[node]:
            nxt=edge['target_id']
            if nxt in seen:
                continue
            new_route=route+[edge]
            if nxt==target:
                found.append({'hops':len(new_route),'edges':new_route})
            else:
                queue.append((nxt,new_route,seen|{nxt}))
    return found

def topology(graph):
    degree=Counter(endpoint for e in graph['edges'] for endpoint in (e['source_id'],e['target_id']))
    return {'nodes':len(graph['nodes']),'edges':len(graph['edges']),
            'isolated_nodes':sum(degree[n['id']]==0 for n in graph['nodes'])}

def unchanged(plan):
    directory=ROOT/'运行结果'/BASELINE
    return all(sha(directory/filename)==plan[key] for filename,key in [
        ('record.json','baseline_record_sha256'),('结构草稿.json','baseline_artifact_sha256'),
        ('核对记录.json','baseline_audit_sha256')])

def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true',help='只核验修订方案并组装草稿，不执行正式查询或创建结果目录')
    args=parser.parse_args(argv)
    plan,fixture,original,chunks=load_materials()
    reviewed=build_reviewed(plan,original,chunks)
    baseline=baseline_view(original)
    print('3-7助手复核修订与固定查询：原始模型图单独保留，本次方案不是模型自动改进。')
    print('模型、嵌入、网络请求均0；不安装、下载、调用裁判或自动重试。')
    print('修订由助手依据原文整理，尚未经用户逐条确认；不作公平性能比较或完整RAPTOR/GraphRAG验收。')
    if args.preview:
        print('REVIEWED_STRUCTURE_PREVIEW_ONLY：方案及依据核验完成，未运行正式查询。')
        return 0
    directory=ROOT/'运行结果'/('reviewed_structure_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=False)
    path=directory/'record.json'
    result={'stage':'chapter3_assistant_reviewed_structure_queries','status':'running',
            'created_at':datetime.now(CHINA).isoformat(),'script_sha256':sha(Path(__file__)),
            'plan_sha256':sha(PLAN),'fixture_sha256':sha(FIXTURE),'official_manifest_sha256':sha(MANIFEST),
            'baseline_record_sha256':plan['baseline_record_sha256'],
            'baseline_artifact_sha256':plan['baseline_artifact_sha256'],
            'baseline_audit_sha256':plan['baseline_audit_sha256'],'review_plan':plan,
            'baseline_graph':baseline,'reviewed_graph':reviewed,
            'topology':{'original_model':topology(baseline),'assistant_reviewed':topology(reviewed)},
            'queries':[],'model_requests':0,'embedding_forward_batches':0,'network_requests':0,
            'user_review_status':'not_yet_confirmed',
            'scope':'Assistant-authored editorial view, exact evidence anchors and fixed-ID path queries; not model improvement, natural-language routing, retrieval accuracy or complete index evaluation.'}
    dump(path,result)
    print('记录路径：',path)
    try:
        print('原始模型图：',result['topology']['original_model'])
        print('助手修订视图：',result['topology']['assistant_reviewed'])
        print('明确修改：处理器支持限定、数据传输组名称、补充原文5.5.1指令组节点；关系逐条附修订依据。')
        labels={n['id']:n['name'] for n in reviewed['nodes']}
        for case in plan['queries']:
            row={'case_id':case['case_id'],'question':case['question'],'routing':'fixed endpoints or fixed summary, not automatic language understanding'}
            print('\n问题：',case['question'])
            if case.get('kind')=='display_fixed_reviewed_summary':
                row['original_model_claims']=original['parent_summary']['accepted_claims']
                row['reviewed_claims']=reviewed['summary']['claims']
                for claim in row['reviewed_claims']:
                    print(' ',claim['text'])
            else:
                row.update(source_id=case['source_id'],target_id=case['target_id'],max_hops=case['max_hops'])
                row['original_model_paths']=paths(baseline,case['source_id'],case['target_id'],case['max_hops'])
                row['reviewed_paths']=paths(reviewed,case['source_id'],case['target_id'],case['max_hops'])
                print('  原始图路径数：',len(row['original_model_paths']),'；修订视图路径数：',len(row['reviewed_paths']))
                for route in row['reviewed_paths']:
                    print('  ',labels[case['source_id']]+''.join(
                        ' --'+e['relation']+'--> '+labels[e['target_id']] for e in route['edges']))
                    for edge in route['edges']:
                        print('    依据：',edge['review_basis'])
                        for evidence in edge['evidence']:
                            print('     ',evidence['chunk_id'],'：',evidence['quote'])
            result['queries'].append(row)
            dump(path,result)
        if not unchanged(plan):
            raise ValueError('原始基线发生变化')
        artifact=directory/'助手修订视图.json'
        dump(artifact,reviewed)
        result.update(status='reviewed_structure_queries_saved',baseline_unchanged=True,
                      reviewed_artifact_sha256=sha(artifact))
        dump(path,result)
        print('\nREVIEWED_STRUCTURE_QUERIES_SAVED：助手修订视图与查询已保存，待核对；不代表模型自行修好了关系。')
        print('记录路径：',path)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='incomplete',error={'type':type(error).__name__,'message':str(error)})
        dump(path,result)
        print('本步未完成：',result['error'])
        print('记录路径：',path)
        return 1

if __name__=='__main__':
    raise SystemExit(main())

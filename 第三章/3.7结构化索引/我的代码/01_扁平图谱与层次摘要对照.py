"""3-7第一步：沿用官方人工知识库，对比词频片段检索、关系路径与父摘要。"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '官方资料/upstream'
MANIFEST = ROOT / '官方资料/manifest.json'
FIXTURE = ROOT / '测试材料/官方示意知识与查询.json'
COMMIT = 'dbc046eb896ac4e39aa19c7774c8bf49583b89a6'
CHINA = timezone(timedelta(hours=8))
TOP_K = 3
MAX_HOPS = 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def load_materials():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['commit'] != COMMIT:
        raise ValueError('官方版本发生变化')
    for item in manifest['files']:
        path = (UPSTREAM / item['path']).resolve()
        if not path.is_relative_to(UPSTREAM.resolve()) or sha(path) != item['sha256']:
            raise ValueError(f"官方文件校验失败：{item['path']}")
    fixture = json.loads(FIXTURE.read_text(encoding='utf-8'))
    source = UPSTREAM / 'chapter3/structured-index/structured_vs_flat_demo.py'
    if fixture['demo_source_sha256'] != sha(source):
        raise ValueError('教学输入与官方示例源码不对应')
    return fixture


def load_official_demo(fixture):
    source = UPSTREAM / 'chapter3/structured-index/structured_vs_flat_demo.py'
    spec = importlib.util.spec_from_file_location('pinned_structured_demo', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 只加载数据和检索函数；不调用官方带固定成败结论的演示打印入口。
    if (module.ENTITIES != fixture['entities']
            or [list(t) for t in module.TRIPLES] != fixture['triples']
            or module.TREE_SUMMARY != fixture['tree_summary']):
        raise ValueError('人工结构与冻结教学输入不一致')
    return module


def pack_path(module, path):
    return {'steps': [{'source': src, 'relation': rel, 'target': dst} for src, rel, dst in path],
            'hops': len(path), 'formatted': module.format_path(path)}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='只检查来源、依赖与教学输入，不查询、不写结果')
    args = parser.parse_args(argv)
    fixture = load_materials()
    version = importlib.metadata.version('networkx')
    print('3-7示意对照：官方人工整理的10个实体、13条关系、1个父摘要，3类固定问题。')
    print('扁平侧使用词频余弦Top-3，不是MiniLM或BM25；图侧最多3跳，树侧直接展示预写摘要。')
    print('生成式模型请求0，嵌入前向0，网络请求0；无安装、下载、API Key或自动重试。')
    print('本步观察已有结构如何查询，不进行LLM抽取、社区发现、聚类或递归摘要。')
    if args.preview:
        print('STRUCTURED_DEMO_PREVIEW_ONLY：27份官方资料与输入来源通过校验；networkx=', version)
        return 0
    directory = ROOT / '运行结果' / ('structured_demo_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'record.json'
    record = {'stage': 'chapter3_structured_curated_demo', 'status': 'running',
              'created_at': datetime.now(CHINA).isoformat(), 'script_sha256': sha(__file__),
              'official_manifest_sha256': sha(MANIFEST), 'fixture_sha256': sha(FIXTURE),
              'upstream_commit': COMMIT, 'knowledge': {key: fixture[key] for key in ('entities', 'triples', 'tree_summary')},
              'protocol': {'flat_top_k': TOP_K, 'max_hops': MAX_HOPS, 'flat_scoring': 'upstream token-frequency cosine',
                           'structure_origin': 'upstream hand-curated entities, directed relations and parent summary',
                           'query_routing': 'three fixed task types; not automatic query understanding',
                           'graph_start': 'upstream longest entity-name substring match',
                           'tree_selection': 'directly display fixed parent summary, not similarity-based tree search',
                           'flat_and_graph_information_budget_equal': False,
                           'not_an_accuracy_benchmark': True},
              'environment': {'python': sys.version.split()[0], 'networkx': version},
              'model_requests': 0, 'embedding_forward_batches': 0, 'network_requests': 0,
              'real_index_built_from_document': False, 'queries': [],
              'scope': 'Official curated-structure teaching demo only, not LLM-built RAPTOR or GraphRAG performance evaluation.'}
    dump(path, record)
    try:
        module = load_official_demo(fixture)
        flat = module.FlatRetriever(module.ENTITIES)
        graph = module.build_graph(module.TRIPLES)
        if graph.number_of_nodes() != 10 or graph.number_of_edges() != 13:
            raise ValueError('示意图规模与固定材料不一致')
        record['graph'] = {'node_count': graph.number_of_nodes(), 'edge_count': graph.number_of_edges(),
                           'directed': graph.is_directed(), 'acyclic': module.nx.is_directed_acyclic_graph(graph)}
        dump(path, record)
        for case in fixture['queries']:
            query = case['query']
            hits = flat.search(query, TOP_K)
            if len(hits) != TOP_K or any(not math.isfinite(h['score']) for h in hits):
                raise ValueError('扁平检索结果数量或分数异常')
            row = {'case_id': case['case_id'], 'task': case['task'], 'query': query, 'flat_top_3': hits}
            print(f'\n{case["case_id"]}｜{case["task"]}：{query}')
            print('  扁平片段：')
            for rank, hit in enumerate(hits, 1):
                print(f'   {rank}. {hit["name"]}  cos={hit["score"]:.4f}  {hit["desc"]}')
            if case['case_id'] == 'multi_hop':
                start = module.match_entity(graph, query)
                paths = module.multi_hop_paths(graph, start, MAX_HOPS) if start else []
                control_paths = [p for p in paths if graph.nodes[p[-1][2]]['type'] == 'control-bit']
                enable_paths = [p for p in control_paths if p[-1][1] == '需要启用']
                row.update(graph_start=start, graph_all_paths=[pack_path(module, p) for p in paths],
                           control_bit_paths=[pack_path(module, p) for p in control_paths],
                           enable_paths=[pack_path(module, p) for p in enable_paths],
                           answer_nodes=list(dict.fromkeys(p[-1][2] for p in enable_paths)),
                           selection_rule='fixed task: target type control-bit and final relation 需要启用')
                print('  图关系路径（起点由名字匹配，控制位筛选规则由题型预先指定）：')
                for p in control_paths:
                    print('   ', module.format_path(p))
                print('  在本示意图中，沿“需要启用”到达：', row['answer_nodes'])
                print('  注意：扁平侧若已找回控制位片段也照实保留；图侧额外展示它与起点的关系链。')
            elif case['case_id'] == 'cross_entity':
                relations = []
                for ext in case['extension_entities']:
                    for _, dst, attrs in graph.out_edges(ext, data=True):
                        if attrs['rel'] == case['relation']:
                            relations.append({'source': ext, 'relation': attrs['rel'], 'target': dst,
                                              'target_description': graph.nodes[dst]['desc']})
                row.update(structured_relations=relations,
                           selection_rule='fixed task names SSE/AVX and relation 使用寄存器; not a learned query parser')
                print('  图侧按预定“使用寄存器”关系对齐两侧事实：')
                for item in relations:
                    print(f'    {item["source"]} --{item["relation"]}--> {item["target"]}：{item["target_description"]}')
            elif case['case_id'] == 'hierarchical':
                summary = module.TREE_SUMMARY
                children = [{'name': name, **module.ENTITIES[name]} for name in summary['children']]
                row.update(parent_summary=summary, child_descriptions=children,
                           selection_rule='directly return hand-authored parent summary; no automatic tree retrieval')
                print('  固定父摘要：', summary['id'])
                print('   ', summary['summary'])
                print('  可向下查看的叶子：', summary['children'])
                print('  注意：摘要预先写好；本次没有模型概括、聚类或验证摘要全部内容。')
            else:
                raise ValueError('未知固定题型')
            record['queries'].append(row)
            dump(path, record)
        record['status'] = 'structured_curated_demo_saved'
        dump(path, record)
        print('\nSTRUCTURED_CURATED_DEMO_SAVED：仅示意结构查询完成；片段、路径与父子关系待核对。')
        print('本次不评分或声称结构化一定更优；真实模型构建索引尚未进行。')
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

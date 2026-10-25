"""读取官方126工具定义，对比全量/按需展示的文字量；非模型发现或工具执行。"""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT/'官方资料'
CHINA = timezone(timedelta(hours=8))
DISCOVER = {'type': 'function', 'function': {'name': 'discover_tools',
            'description': '根据当前能力需求查找相关工具说明；此步骤只展示接口定义，不执行检索。',
            'parameters': {'type': 'object', 'properties': {'need': {'type': 'string'}}, 'required': ['need']}}}
TASK = '查找Transformer相关论文，并下载选中的论文。'
MANUAL_NAMES = ['arxiv_search', 'download_file']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(node):
    # 只允许构造定义的数据节点，不import或exec参考源码，不执行mock工具。
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return node.value
    if isinstance(node, ast.Dict):
        return {evaluate(k): evaluate(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [evaluate(v) for v in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [evaluate(v) for v in node.args]
        if node.func.id in ('_s', '_i') and len(args) == 1:
            return {'type': 'string' if node.func.id == '_s' else 'integer', 'description': args[0]}
        if node.func.id == '_tool' and len(args) == 3:
            name, description, params = args
            return {'type': 'function', 'function': {'name': name, 'description': description,
                    'parameters': {'type': 'object', 'properties': params, 'required': list(params)}}}
    raise ValueError('来源定义出现未支持的节点：'+type(node).__name__)


def prepare():
    manifest = json.loads((SOURCES/'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = (SOURCES/item['path']).resolve()
        if not path.is_relative_to(SOURCES.resolve()) or sha(path) != item['sha256']:
            raise ValueError('冻结参考来源发生变化')
    tree = ast.parse((SOURCES/'tools_library.py').read_text(encoding='utf-8'))
    definitions = []
    base_names = None
    for node in tree.body:
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == '_DEFS':
            if not isinstance(node.op, ast.Add):
                raise ValueError('定义拼接方式变化')
            definitions.extend(evaluate(node.value))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BASE_TOOL_NAMES' for t in node.targets):
            base_names = evaluate(node.value)
    by = {t['function']['name']: t for t in definitions}
    if len(definitions) != 126 or len(by) != 126 or base_names != ['calculator', 'get_current_time']:
        raise ValueError('固定126工具或基础工具定义变化')
    full = definitions
    initial = [by[name] for name in base_names]+[DISCOVER]
    # 明确人工指定，仅用于可见信息示意，不运行发现算法或按标准答案计算成绩。
    manual = [by[name] for name in MANUAL_NAMES]
    return {'full': full, 'initial': initial, 'manual_candidates': manual,
            'after_manual_demo': initial+manual}, manifest


def metrics(tools):
    text = json.dumps(tools, ensure_ascii=False, separators=(',', ':'))
    return {'tool_count': len(tools), 'json_characters': len(text), 'utf8_bytes': len(text.encode('utf-8'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    views, manifest = prepare()
    counts = {name: metrics(view) for name, view in views.items()}
    print('4-1第一步：阅读126工具目录及按需展示示意；无模型、嵌入、MCP或网络请求。')
    print('示例任务：', TASK)
    print('人工指定示意候选：', MANUAL_NAMES, '；不是模型自行发现，不执行任何工具。')
    print('说明量对照：', json.dumps(counts, ensure_ascii=False))
    print('字符数和字节数不是token，不代表耗时、模型准确率或累计调用成本。')
    if args.preview:
        print('TOOL_CATALOG_PREVIEW_ONLY：126定义和来源检查通过，未创建正式结果。')
        return 0
    directory = ROOT/'运行结果'/('catalog_reading_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    for name, view in views.items():
        (directory/(name+'.json')).write_text(json.dumps(view, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 官方工具目录\n', '只阅读定义，不执行参考文件的mock实现；以下描述不是实际服务可用性保证。\n']
    for item in views['full']:
        f = item['function']
        lines.append(f"- **{f['name']}**：{f['description']}；参数：{', '.join(f['parameters']['properties'])}\n")
    (directory/'工具目录阅读.md').write_text('\n'.join(lines), encoding='utf-8')
    record = {'status': 'catalog_views_saved_not_model_experiment', 'created_at': datetime.now(CHINA).isoformat(),
              'script_sha256': sha(Path(__file__)), 'manifest_sha256': sha(SOURCES/'manifest.json'),
              'manifest': manifest, 'task': TASK, 'manual_candidate_names': MANUAL_NAMES,
              'metrics': counts, 'model_requests': 0, 'tool_executions': 0, 'network_requests': 0,
              'scope': 'static definition reading and manual schema visibility illustration; no autonomous discovery or accuracy result'}
    (directory/'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('初始仅展示：calculator、get_current_time、discover_tools。人工示意追加后才能看到两个专业工具定义。')
    print('TOOL_CATALOG_READING_SAVED：目录与四种展示视图保存；不代表完成主动发现实验。')
    print('记录路径：', directory/'record.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

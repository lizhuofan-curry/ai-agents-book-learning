"""复用已保存单题基线，仅对Agent增加证据缺口检查提示；不运行旧组。"""
from datetime import datetime
import importlib.util
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT/'运行结果/live_single_20261024_195103_852897/record.json'
BASELINE_SHA256 = '1ef7e0dda429369f472cc4f87b38c7f1ec4735b6a20764569af8c014ddb51dc4'
MODEL = 'qwen3.5:9b'
GAP_PROMPT = '''
本次增加以下证据缺口检查要求：
先把题干明确提到的方面作为核查清单：过失致人重伤、醉酒、盗窃前科。
每次看到检索结果后，区分每个方面已有材料支持、仍缺证据、或题干条件不明确。
在现有limitations字段中用简短核查结论记录这些状态，不需要输出内部思维过程。
有实质缺口且仍可检索时，尝试围绕缺失方面提出针对性查询；不要只是重复整道问题。
不用强制检索满3次。若新检索预计无法澄清、材料始终不足或题干缺少必要事实，可以回答已被支持的部分，并明确不能确定的部分。
醉酒不自动意味着驾驶机动车，有前科也不能未经条件核查就等同某种法律身份。
不要推断题干未说的事实；不要把程序标注或外部常识当证据。
回答前检查每条陈述是否真的由所引片段支持。若没有依据，不把该断言写进claims；在limitations说明缺什么。
仍使用原有search/answer JSON格式、原有工具与预算，不调用额外工具。
'''


def load_live():
    spec = importlib.util.spec_from_file_location('live_gap_adaptation', ROOT/'我的代码/03_单题普通RAG与自主检索.py')
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    return live


def prepare():
    live = load_live()
    retrieval = live.load_step()
    if retrieval.sha(BASELINE) != BASELINE_SHA256:
        raise ValueError('固定基线指纹变化，停止；不覆盖旧结果')
    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    if baseline['model'] != MODEL or baseline['status'] != 'both_answered_review_pending':
        raise ValueError('基线模型或完成状态不符')
    if (retrieval.sha(Path(live.__file__)) != baseline['script_sha256']
            or retrieval.sha(Path(retrieval.__file__)) != baseline['retrieval_script_sha256']
            or retrieval.sha(retrieval.MANIFEST) != baseline['manifest_sha256']):
        raise ValueError('基线代码或来源变化，不能按仅改变提示解释')
    retriever, dataset, _ = retrieval.load_sources()
    case = next(c for c in dataset['cases'] if c['id'] == baseline['case_id'])
    if case['question'] != baseline['question']:
        raise ValueError('问题变化')
    expected = live.payload(MODEL, [])
    first = baseline['arms']['agentic']['calls'][0]['request']
    for key in ('model', 'format', 'stream', 'think', 'options'):
        if first[key] != expected[key]:
            raise ValueError('基线请求配置变化：'+key)
    # 只改本次动态加载模块中的提示，磁盘03源码和历史结果不改。
    original_system = live.SYSTEM
    live.SYSTEM = original_system + '\n' + GAP_PROMPT
    return live, retrieval, retriever, case, baseline, original_system


def compare(baseline, arm):
    before = baseline['arms']['agentic']
    return {
        'baseline_agentic': {'status': before['status'], 'stop_reason': before['stop_reason'],
                            **before['evaluation']},
        'gap_prompt_agentic': {'status': arm['status'], 'stop_reason': arm['stop_reason'],
                              **arm['evaluation']},
        'semantic_review': 'pending; cite IDs alone do not establish support',
        'design_limitations': 'One post-hoc prompt bundle on one known case, saved baseline from earlier run; no causal attribution to one sentence, no statistical or universal accuracy claim.'}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    live, retrieval, retriever, case, baseline, original_system = prepare()
    print('3-8提示对照：复用已保存基线，仅运行增加证据缺口提示的Agent组。', flush=True)
    print('问题：', case['question'], flush=True)
    print('模型：', MODEL, '；Top-5；最多3次检索/4次模型请求；其他采样及预算沿用03。', flush=True)
    print('普通组和原Agent不重跑；不提供目标条文或预写查询，不强制查满3次。', flush=True)
    if args.preview:
        print('GAP_PROMPT_PREVIEW_ONLY：基线/代码/来源/参数核对通过；未查询正式问题或请求模型。')
        return 0
    directory = ROOT/'运行结果'/('gap_prompt_'+datetime.now(retrieval.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(retrieval.CHINA).isoformat(),
        'model': MODEL, 'case_id': case['id'], 'question': case['question'],
        'script_sha256': retrieval.sha(Path(__file__)),
        'live_script_sha256': retrieval.sha(Path(live.__file__)),
        'retrieval_script_sha256': retrieval.sha(Path(retrieval.__file__)),
        'manifest_sha256': retrieval.sha(retrieval.MANIFEST),
        'baseline_relative_path': BASELINE.relative_to(ROOT).as_posix(),
        'baseline_sha256': BASELINE_SHA256, 'original_system': original_system,
        'added_system_prompt': GAP_PROMPT, 'effective_system': live.SYSTEM,
        'maximum_model_calls': 4, 'maximum_searches': 3,
        'automatic_retries': 0, 'cloud_requests': 0, 'embedding_requests': 0,
        'baseline_rerun': False, 'semantic_review': 'pending', 'arm': {}}
    def save():
        retrieval.dump(path, record)
    save()
    print('记录路径：', path, flush=True)
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.mount('http://', requests.adapters.HTTPAdapter(max_retries=0))
            # 仅question进入循环；gold只在该组结束后用于统计。
            live.run_arm('agentic', case['question'], retriever, session, MODEL, record['arm'], save)
        arm = record['arm']
        arm['evaluation'] = live.evaluate(arm, case['gold_articles'], retrieval)
        record['comparison'] = compare(baseline, arm)
        if (retrieval.sha(BASELINE) != BASELINE_SHA256
                or retrieval.sha(retrieval.MANIFEST) != record['manifest_sha256']):
            raise ValueError('运行期间基线或来源清单变化')
        retrieval.load_sources()
        record['status'] = ('answered_review_pending' if arm['status']=='answered_review_pending'
                            else 'finished_with_unanswered_or_failed_arm')
        save()
        print('\n与保存基线比较：', json.dumps(record['comparison'], ensure_ascii=False, indent=2))
        print('GAP_PROMPT_COMPARISON_SAVED：新提示组轨迹已保存，语义及实际变化待核对。')
        print('记录路径：', path)
        return 0 if record['status']=='answered_review_pending' else 1
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        save()
        print('本步未完成：', record['error'])
        print('记录路径：', path)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

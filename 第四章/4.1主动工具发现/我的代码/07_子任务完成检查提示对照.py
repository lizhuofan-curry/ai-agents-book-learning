"""复用已保存基线，只运行增加通用子任务检查提示的按需组。"""
import argparse
import asyncio
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
COMPARE = Path(__file__).with_name('02_全量与按需工具选择对照.py')
spec = importlib.util.spec_from_file_location('unchanged_discovery_compare', COMPARE)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
SOURCE = ROOT/'运行结果/tool_choice_compare_20261025_203942_696397/record.json'
SOURCE_SHA = 'ebc3b6313fc09c60d45654736640482f7e04d67b2483048edf98363b7778bcec'
COMPARE_SHA = '621d06beda6a3dd950a90ef7f34b8dcec80ac82b014f99e3ddac3fc46ea75983'
ADDED_PROMPT = '结束前逐项核对用户明确提出的目标是否已处理；若仍有目标未处理且缺少相应工具，应先描述该能力需求并尝试发现，再依据实际结果说明完成与限制。不要把已有工具调用结束当作整个任务完成。'


def prepare():
    if compare.sdk.sha(SOURCE) != SOURCE_SHA or compare.sdk.sha(COMPARE) != COMPARE_SHA:
        raise ValueError('冻结基线记录或原实验代码变化')
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    full, initial, manifest = compare.prepare()
    if manifest != source['manifest']:
        raise ValueError('资料清单与原运行不一致')
    baseline = source['arms']['discovery']
    if baseline['calls'][0]['request']['messages'][0]['content'] != compare.SYSTEM:
        raise ValueError('原系统提示与保存基线不同')
    if source['per_arm_model_budget'] != compare.MAX_CALLS or source['per_arm_tool_action_budget'] != compare.MAX_ACTIONS:
        raise ValueError('预算与原运行不同')
    return source, full, initial, manifest


async def run_checked(full, initial, arm, save, model_call=compare.call_model):
    original = compare.SYSTEM
    compare.SYSTEM = original+'\n'+ADDED_PROMPT
    try:
        await compare.run_arm('discovery', full, initial, arm, save, model_call)
    finally:
        compare.SYSTEM = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    sdk = compare.sdk
    if Path(sys.prefix).resolve() != sdk.ENVIRONMENT.resolve():
        if not sdk.PYTHON.is_file():
            raise RuntimeError('未找到已有SDK环境')
        print('切换已有第四章环境；不安装依赖。', flush=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.execv(str(sdk.PYTHON), [str(sdk.PYTHON), str(Path(__file__)), *sys.argv[1:]])
    sdk.use_sdk_environment()
    source, full, initial, manifest = prepare()
    print('4-1第四步：复用保存基线，只运行追加通用子任务检查提示的按需组。')
    print('新增提示：', ADDED_PROMPT)
    print('同一126工具、Top-5词项检索、模型/采样/模拟返回、最多5请求/4动作不变。')
    print('原全量和原按需组不重跑；不提供专业工具名或预写need，不强制第二次发现。')
    print('业务操作仍为教学模拟，无真实论文检索、下载、云端、预热或重试。')
    if args.preview:
        print('SUBTASK_PROMPT_PREVIEW_ONLY：基线、提示差异与来源检查通过，未请求模型。')
        return 0
    directory = ROOT/'运行结果'/('subtask_prompt_'+datetime.now(sdk.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    arm = {}
    record = {'status': 'running', 'created_at': datetime.now(sdk.CHINA).isoformat(),
              'script_sha256': sdk.sha(Path(__file__)), 'compare_script_sha256': COMPARE_SHA,
              'source_record_sha256': SOURCE_SHA, 'manifest': manifest, 'task': compare.catalog_step.TASK,
              'added_prompt': ADDED_PROMPT, 'model': compare.MODEL, 'maximum_model_requests': compare.MAX_CALLS,
              'maximum_tool_actions': compare.MAX_ACTIONS, 'candidate_top_k': compare.TOP_K,
              'automatic_retries': 0, 'cloud_requests': 0, 'real_business_operations': 0,
              'baseline_metrics': source['arms']['discovery']['metrics'], 'new_arm': arm,
              'design': 'append one generic task-completion-check prompt to system; all other settings unchanged; independent fresh history, saved baseline not rerun',
              'limitations': 'post-hoc prompt revision on known simulated case; single run, not general accuracy or causal improvement; mock coverage and budgets retain prior limitations'}

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    save()
    print('记录路径：', path, flush=True)
    try:
        asyncio.run(run_checked(full, initial, arm, save))
        prepare()
        record['status'] = 'subtask_prompt_run_saved_review_pending'
        print('原按需组：', json.dumps(record['baseline_metrics'], ensure_ascii=False))
        print('新提示组状态：', arm['status'])
        print('新提示组：', json.dumps(arm['metrics'], ensure_ascii=False))
        print('SUBTASK_PROMPT_RUN_SAVED：新轨迹保存；是否完成模拟调用链和正确陈述仍需核对。')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print('本步中断，不重试：', record['error'])
        return 1
    finally:
        save()
        print('记录路径：', path)


if __name__ == '__main__':
    raise SystemExit(main())

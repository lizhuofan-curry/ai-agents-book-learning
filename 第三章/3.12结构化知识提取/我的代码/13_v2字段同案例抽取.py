"""在已审阅的每类第二条正文上检查v2新增字段；属于事后开发检查。"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import time

import requests

ROOT = Path(__file__).resolve().parents[1]

def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

shared = load_module('reviewed_case_extractor', '07_审阅字段逐案抽取.py')
reader = load_module('synthetic_case_reader', '01_阅读合成案例与字段边界.py')
base = shared.base
PLAN = ROOT/'12_助手审阅字段方案_v2.json'
SHARED_SHA = '6cf7220aa88d38fef473a61316a5051a042e2cd7480a281c306a6f243ba9f936'
PLAN_SHA = '34cdc7daf14b643c225b8174df51e800b58b20f908e8cde42f07520912b356df'
def make_request(rows, plan):
    request = shared.make_request(rows, plan)
    request['messages'][0]['content'] += ('不确定性文字也必须有依据；未知只写未提及，不能写成没有。'
        '追回字段null仅表示未见明确追回陈述，不表示钱款实际未到位。')
    return request
def validate(data, rows, plan):
    result = shared.validate(data, rows, plan)
    for case in result:
        count = case['values']['victim_count_reported']['value']
        if count is not None and (type(count) is not int or count < 1):
            raise ValueError('明确被害人数须为正整数或null')
    return result


def select_new(all_rows):
    # 仅按类别和原文件顺序分层选择，不按正文难度、标签或模型结果挑选。
    selected = [[r for r in all_rows if r['charge'] == charge][1] for charge in reader.CHARGES]
    first = reader.select_samples(all_rows)
    if set(r['id'] for r in first) & set(r['id'] for r in selected):
        raise ValueError('新样本与发现样本重叠')
    if set(r['fact'] for r in first) & set(r['fact'] for r in selected):
        raise ValueError('新样本正文重复')
    return selected


def prepare():
    if base.sha(Path(shared.__file__)) != SHARED_SHA or base.sha(PLAN) != PLAN_SHA:
        raise ValueError('冻结抽取逻辑或字段方案变化')
    all_rows, _ = reader.load_sources()
    selected = select_new(all_rows)
    rows = [{'sample_id': f'new_{i+1}', 'fact': r['fact']} for i, r in enumerate(selected)]
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    for row in rows:
        row['spans'] = [
            {'span_id': row['sample_id']+'_s'+str(i+1), 'text': match.group(),
             'start': match.start(), 'end': match.end()}
            for i, match in enumerate(re.finditer(r'[^。，；！？]+[。，；！？]?', row['fact']))]
        if ''.join(s['text'] for s in row['spans']) != row['fact']:
            raise ValueError('原文分段不完整')
    return rows, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    rows, plan = prepare()
    request = make_request(rows, plan)
    print('3-12第五步：助手审阅v2共13字段，在已见三例上检查新增信息抽取。', flush=True)
    print('复用上一组三例，新增4字段并补充缺失状态提示；不是未见测试或模型自主改进。', flush=True)
    print('最多1次本机qwen3.5:9b请求；3072输出上限；不重试、无云端或聚类。', flush=True)
    print('模型选择原文片段编号，程序复制证据；不发送gold、结果标签或标准答案。', flush=True)
    if args.preview:
        print('V2_EXTRACTION_PREVIEW_ONLY：来源和分段核对通过，无模型请求。')
        return 0
    directory = ROOT/'运行结果'/('v2_case_extraction_'+datetime.now(base.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(base.CHINA).isoformat(),
              'script_sha256': base.sha(Path(__file__)), 'helper_script_sha256': base.sha(Path(base.__file__)),
              'plan_sha256': base.sha(PLAN), 'shared_extractor_sha256': SHARED_SHA, 'source_manifest_sha256': base.sha(reader.MANIFEST),
              'source_dataset_sha256': base.sha(reader.DATA),
              'selection_rule': 'same second occurrence per charge as previous reviewed run',
              'development_baseline_sha256': '4cbc45c42667c7a0c6740a797058c04b3b045c74df6bf087d0d4652fa40dde73',
              'selected_source_ids': [r['id'] for r in select_new(reader.load_sources()[0])], 'plan': plan,
              'request': request, 'source_rows': rows, 'maximum_model_calls': 1, 'model_calls_attempted': 0,
              'automatic_retries': 0, 'semantic_review': 'pending',
              'scope': 'post-hoc assistant v2 fields plus clarification on same three development cases; not unseen test or single-factor comparison'}

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    save()
    print('记录路径：', path, flush=True)
    started = time.perf_counter()
    try:
        with requests.Session() as client:
            client.trust_env = False
            client.mount('http://', requests.adapters.HTTPAdapter(max_retries=0))
            record['model_calls_attempted'] = 1
            save()
            response = client.post(base.ENDPOINT, json=request, timeout=(10, 600), allow_redirects=False)
            record.update(http_status=response.status_code, raw_http_body=response.text)
            save()
            response.raise_for_status()
        raw = json.loads(response.text, object_pairs_hook=base.unique)
        record['response'] = raw
        save()
        if raw.get('model') != base.MODEL or raw.get('done') is not True or raw.get('done_reason') != 'stop':
            raise ValueError('模型名或停止状态异常')
        candidate = json.loads(raw['message']['content'], object_pairs_hook=base.unique)
        result = validate(candidate, rows, plan)
        if prepare() != (rows, plan):
            raise ValueError('输入在运行中发生变化')
        record.update(status='v2_case_extraction_saved_review_pending', candidate=candidate, copied_result=result,
                      prompt_tokens=raw.get('prompt_eval_count'), output_tokens=raw.get('eval_count'))
        save()
        for case in result:
            print('\n'+case['sample_id'])
            for name, item in case['values'].items():
                print(' ', name, '：', json.dumps(item['value'], ensure_ascii=False))
                for span in case['copied_evidence'][name]:
                    print('   ', span['span_id'], '：', span['text'])
        print('不确定性：', candidate['uncertainties'])
        print('V2_CASE_EXTRACTION_SAVED：取值与原文证据保存；支持关系待核对。')
        print('记录路径：', path)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        save()
        print('本步未完成，不重试：', record['error'])
        print('记录路径：', path)
        return 1
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        save()


if __name__ == '__main__':
    raise SystemExit(main())

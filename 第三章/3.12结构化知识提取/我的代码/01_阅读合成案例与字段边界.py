"""固定合成案例盘点及三条正文阅读；标签另存，不执行发现/抽取/聚类。"""
import argparse
from collections import Counter
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'官方资料/upstream'
MANIFEST=ROOT/'官方资料/manifest.json'
DATA=UP/'chapter3/structured-knowledge-extraction/data/cases.jsonl'
CHINA=timezone(timedelta(hours=8))
CHARGES=('盗窃罪','故意伤害罪','诈骗罪')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(rows):
    if len(rows)!=66 or len({r['id'] for r in rows})!=66:
        raise ValueError('需固定66条且ID唯一')
    if set(r['charge'] for r in rows)!=set(CHARGES):raise ValueError('罪名类别变化')
    for row in rows:
        if not isinstance(row['fact'],str) or not row['fact'].strip():raise ValueError('正文为空')
        if not isinstance(row['gold'],dict):raise ValueError('gold格式错误')
        if type(row['label_months']) is not int or row['label_months']<0:raise ValueError('标签格式错误')


def load_sources():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=(UP/item['path']).resolve()
        if not path.is_relative_to(UP.resolve()) or sha(path)!=item['sha256']:raise ValueError('冻结来源变化')
    rows=[json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
    validate(rows)
    return rows,manifest


def select_samples(rows):
    # 不按结果或模型表现筛选，固定每类在原文件中的第一条。
    return [next(r for r in rows if r['charge']==charge) for charge in CHARGES]


def fact_only(rows):
    # 后续发现阶段只能读取该窄视图，连charge和含罪名的ID也不当作正文提示。
    return [{'sample_id':f'sample_{i+1}','fact':row['fact']} for i,row in enumerate(rows)]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preview',action='store_true')
    args=parser.parse_args();rows,manifest=load_sources();samples=select_samples(rows);visible=fact_only(samples)
    print('3-12第一步：合成案例阅读；不是实际判例或现实量刑依据。')
    print('总数：',len(rows),'；类别分布：',dict(Counter(r['charge'] for r in rows)))
    print('fact为正文，gold为生成时结构化真值，label_months为公式加噪声生成的标签。')
    print('不调用模型、不重新生成数据、不执行抽取或聚类；标准字段和标签单独保存。')
    if args.preview:
        print('CASE_READING_PREVIEW_ONLY：来源、66条结构和固定三样本选择通过，未创建结果目录。')
        return 0
    for row in visible:print('\n'+row['sample_id']+'\n'+row['fact'])
    directory=ROOT/'运行结果'/('case_reading_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=False)
    record={'status':'synthetic_case_reading_saved_review_pending','created_at':datetime.now(CHINA).isoformat(),
        'script_sha256':sha(Path(__file__)),'manifest_sha256':sha(MANIFEST),'manifest':manifest,
        'dataset_sha256':sha(DATA),'total_cases':len(rows),'charge_counts':dict(Counter(r['charge'] for r in rows)),
        'selection_rule':'first occurrence per charge in source file order',
        'selected_source_ids':[r['id'] for r in samples],
        'model_requests':0,'network_requests':0,
        'scope':'synthetic three-case reading and input separation only; no discovery, extraction, causal or prediction results'}
    (directory/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (directory/'发现阶段正文输入.json').write_text(json.dumps(visible,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    reference=[{'sample_id':f'sample_{i+1}',**row} for i,row in enumerate(samples)]
    (directory/'标准字段与标签_仅供后验核对.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 三条合成案例正文\n','先只读正文，自己列出候选因素；不把猜测写成事实，没提及的不等于否定。\n']
    for row in visible:lines.extend(['## '+row['sample_id']+'\n',row['fact']+'\n'])
    lines.append('后续核对才打开标准字段参考。标签来自合成公式，不是真实判决，不从三条样例推断因果规律。\n')
    (directory/'先读案例正文.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\nSYNTHETIC_CASE_READING_SAVED：正文和标准参考已分开保存，待核对。')
    print('记录路径：',directory/'record.json')
    print('先阅读：',directory/'先读案例正文.md')
    return 0


if __name__=='__main__':raise SystemExit(main())

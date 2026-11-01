"""只读回放本人三题原始结果与Codex事后审阅，0次模型或网络请求。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'测试材料/复盘输入.json'
LABELS = {
    'over_refusal_for_available_scoped_evidence': '可作限定范围解释，却完全拒答',
    'claims_supported_with_scope_and_citation_issues': '主要内容有依据，但原始结构范围与引用需修正',
    'appropriate_abstention_for_general_definition': '当前片段缺少一般定义，拒答合理',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    plan = json.loads(INPUT.read_text(encoding='utf-8'))
    parent = (ROOT/'本地运行结果').resolve()
    source = (parent/plan['answer_run']).resolve()
    if source.parent != parent or not source.name.startswith('wiki_answers_'):
        raise ValueError('复盘来源越界')
    answer_path, review_path = source/'record.json', source/'核对记录.json'
    answer_raw, review_raw = answer_path.read_bytes(), review_path.read_bytes()
    if sha(answer_raw) != plan['answer_record_sha256'] or sha(review_raw) != plan['review_sha256']:
        raise ValueError('固定原答或审阅记录已变化，不自动替换')
    answer, review = json.loads(answer_raw), json.loads(review_raw)
    if (review['source_record_sha256'] != sha(answer_raw) or len(answer['model_calls']) != 3
            or len(review['question_reviews']) != 3):
        raise ValueError('原答与事后审阅未正确关联')
    if not all(sha(Path(name).read_bytes()) == expected for name, expected in answer['source_file_hashes'].items()):
        raise ValueError('检索来源发生变化')
    if [call['query_id'] for call in answer['model_calls']] != [row['query_id'] for row in review['question_reviews']]:
        raise ValueError('问题顺序或ID不一致')
    print('个人Wiki第五步：离线回放原答与Codex审阅；0次嵌入、生成或网络请求。')
    print('审阅来自Codex逐段阅读，不是脚本自动理解了事实，也不是本人确认金标。')
    if args.preview:
        print('WIKI_REVIEW_PREVIEW_ONLY：原答、审阅与来源SHA通过，无新结果目录。')
        return 0
    folder = parent/('wiki_review_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    for call, row in zip(answer['model_calls'], review['question_reviews']):
        print('\n'+row['query_id']+'：'+row['question'])
        print('  模型原状态：'+call['answer']['status'])
        print('  审阅：'+LABELS[row['review_outcome']])
        print('  依据：'+row['assessment'])
    print('\n三路首条定性对照：')
    for row in review['retrieval_comparison']:
        print(f'  {row["query_id"]}：BM25={row["bm25_top5_ids"][0]}；语义={row["dense_casefold_top5_ids"][0]}；RRF={row["rrf_top5_ids"][0]}')
    print('  未建立完整相关性金标，不报告定量Recall或MRR。')
    usage = review['usage']
    print(f'\n旧回答阶段：{usage["generation_model_requests"]}次生成，报告输入{usage["prompt_tokens_reported"]}/输出{usage["output_tokens_reported"]} tokens。')
    print('本次复盘：新增模型请求0；输入合计含重复提示，不是不重复文本量或云端账单。')
    unchanged = answer_path.read_bytes() == answer_raw and review_path.read_bytes() == review_raw
    report = {'status': 'offline_review_replayed_not_all_answers_accepted',
        'source_answer_record_sha256': sha(answer_raw), 'source_review_sha256': sha(review_raw),
        'script_sha256': sha(Path(__file__).read_bytes()), 'input_plan_sha256': sha(INPUT.read_bytes()),
        'checks': {'original_answer_and_review_unchanged': unchanged},
        'question_reviews': review['question_reviews'], 'retrieval_comparison': review['retrieval_comparison'],
        'original_generation_usage': usage, 'new_generation_requests': 0, 'new_embedding_forward_batches': 0,
        'network_requests': 0, 'quantitative_retrieval_metrics': None,
        'conclusion': '所选12页的检索、融合、生成链路已实测；问答效果未全部通过，原始失败保留。',
        'assessor': review['reviewer']}
    path = folder/'record.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('WIKI_REVIEW_SAVED：复盘完成不代表三题成功；未覆盖或修写任何模型回答。')
    print('本地复盘记录：'+str(path))
    return 0 if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""3-9第一步：阅读冻结车辆用例，保存带消息来源的固定轮次分块，不检索或生成。"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'官方资料/upstream'
MANIFEST = ROOT/'官方资料/manifest.json'
CASE_PATH = UPSTREAM/'chapter3/user-memory-evaluation/test_cases/layer2/01_multiple_vehicles.yaml'
CHINA = timezone(timedelta(hours=8))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = (UPSTREAM/item['path']).resolve()
        if not path.is_relative_to(UPSTREAM.resolve()) or sha(path) != item['sha256']:
            raise ValueError('冻结资料校验失败：'+item['path'])
    data = yaml.safe_load(CASE_PATH.read_text(encoding='utf-8'))
    if data['test_id'] != 'layer2_01_multiple_vehicles':
        raise ValueError('仅接受当前固定官方用例')
    histories = data['conversation_histories']
    ids = [c['conversation_id'] for c in histories]
    if len(set(ids)) != len(ids):
        raise ValueError('重复会话ID')
    return data, manifest


def rounds_for(messages):
    """按官方chunker规则：assistant收尾且本轮至少2条，末尾残余单列。"""
    rounds, current = [], []
    for i, message in enumerate(messages):
        if message.get('role') not in ('user', 'assistant') or not isinstance(message.get('content'), str):
            raise ValueError('角色或正文格式错误')
        current.append({'message_number': i+1, **message})
        if message['role'] == 'assistant' and len(current) >= 2:
            rounds.append(current)
            current = []
    if current:
        rounds.append(current)
    return rounds


def split_conversation(conversation, test_id, size=20, overlap=2):
    if not 0 <= overlap < size:
        raise ValueError('须满足0<=重叠轮数<每块轮数')
    rounds = rounds_for(conversation['messages'])
    chunks = []
    for start in range(0, len(rounds), size-overlap):
        end = min(start+size, len(rounds))
        messages = [m for part in rounds[start:end] for m in part]
        chunks.append({'chunk_id': f"{test_id}/{conversation['conversation_id']}/rounds_{start+1}_{end}",
            'test_id': test_id, 'conversation_id': conversation['conversation_id'],
            'session_timestamp': conversation.get('timestamp'),
            'session_metadata': conversation.get('metadata', {}),
            'start_round': start+1, 'end_round': end, 'messages': messages,
            'source_message_numbers': [m['message_number'] for m in messages],
            'text': '\n'.join(f"[{m['message_number']}] {m['role']}: {m['content']}" for m in messages)})
        if end == len(rounds):
            break
    # 检查无漏失、无跨会话内容，重叠只来自明确窗口。
    covered = {m['message_number'] for c in chunks for m in c['messages']}
    if covered != set(range(1, len(conversation['messages'])+1)):
        raise ValueError('分块未完整覆盖原始消息')
    return chunks


def render(data, chunks):
    lines = ['# 官方车辆用例：原始会话与分块\n', '这是官方测试资料，不是用户本人的经历。\n',
             '## 当前问题\n', data['user_question']+'\n',
             '不展示预期答案或评分规则。本步不检索、不生成摘要或回答。\n']
    for conv in data['conversation_histories']:
        lines.extend(['## 会话：'+conv['conversation_id']+'\n',
                      '会话时间：'+str(conv.get('timestamp'))+'\n'])
        for i,m in enumerate(conv['messages'],1):
            lines.append(f"### 消息{i} · {m['role']}\n\n{m['content']}\n")
    lines.append('## 分块边界与来源\n')
    for chunk in chunks:
        lines.append(f"- {chunk['chunk_id']}：原消息{chunk['source_message_numbers'][0]}—{chunk['source_message_numbers'][-1]}，时间{chunk['session_timestamp']}。")
    lines.append('\n每块完整内容见同目录record.json的chunks；重叠导致部分消息出现在两块中，不代表两次不同事件。\n')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    data, manifest = load_case()
    print('3-9第一步：原始会话阅读与固定20轮/重叠2轮分块；无模型、检索或网络请求。')
    print('问题：', data['user_question'])
    print('官方测试会话，不是你的个人记录；不把文件中的预期答案加入分块。')
    if args.preview:
        print('会话消息数：', [(c['conversation_id'],len(c['messages'])) for c in data['conversation_histories']])
        print('MEMORY_CHUNK_PREVIEW_ONLY：来源及用例读取通过，未生成正式分块或结果目录。')
        return 0
    chunks = []
    for conv in data['conversation_histories']:
        current = split_conversation(conv, data['test_id'])
        chunks.extend(current)
        print('\n会话：', conv['conversation_id'], '时间：', conv.get('timestamp'),
              '消息数：', len(conv['messages']), '轮数：', len(rounds_for(conv['messages'])))
        for c in current:
            print(' ', c['chunk_id'], '原消息：', c['source_message_numbers'][0], '—', c['source_message_numbers'][-1])
    counts = Counter((c['conversation_id'],m['message_number']) for c in chunks for m in c['messages'])
    record = {'status': 'conversation_chunks_saved_review_pending',
        'created_at': datetime.now(CHINA).isoformat(), 'script_sha256': sha(Path(__file__)),
        'manifest_sha256': sha(MANIFEST), 'case_sha256': sha(CASE_PATH), 'manifest': manifest,
        'test_id': data['test_id'], 'question': data['user_question'],
        'histories': data['conversation_histories'], 'chunks': chunks,
        'chunking': {'rounds_per_chunk':20,'overlap_rounds':2,
                     'round_rule':'assistant terminates round with >=2 messages; final remainder retained',
                     'adaptation':'independent local implementation of official fixed-round window logic',
                     'neighbor_context_enrichment':False,'test_title_or_gold_in_chunk':False},
        'unique_source_messages':len(counts), 'duplicated_by_overlap':sum(v-1 for v in counts.values()),
        'model_requests':0,'network_requests':0,'retrieval_queries':0,
        'scope':'single-user official two-session teaching case; no multi-user isolation test or live recall score'}
    directory = ROOT/'运行结果'/('conversation_chunks_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (directory/'原始会话与分块阅读.md').write_text(render(data,chunks),encoding='utf-8')
    print('\nMEMORY_CONVERSATION_CHUNKS_SAVED：分块和阅读材料已保存，待核对。')
    print('记录路径：',directory/'record.json')
    print('阅读材料：',directory/'原始会话与分块阅读.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

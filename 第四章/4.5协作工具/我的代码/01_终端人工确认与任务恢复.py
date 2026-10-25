"""终端真实人工回复→本地任务恢复；无模型、MCP、网络或通知调用。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import select
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CHINA = timezone(timedelta(hours=8))
TARGET_NAME = '教学结果.txt'
APPROVE = {'APPROVE', '同意'}
REJECT = {'REJECT', '拒绝'}


def timestamp():
    return datetime.now(CHINA).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_sources(root=ROOT):
    folder = Path(root)/'官方资料'
    raw = (folder/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    for item in manifest['files']:
        path = (folder/item['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path.read_bytes()) != item['sha256']:
            raise ValueError('固定官方参考变化')
    return sha(raw)


def event(record, kind, **details):
    record.setdefault('events', []).append({'at': timestamp(), 'kind': kind, **details})


def resolve(record, status):
    approval = record['approval']
    if approval['status'] != 'pending':
        return False  # 终态不可被重复或迟到回复改变。
    if status not in {'approved', 'rejected', 'timeout', 'eof', 'input_unavailable', 'cancelled'}:
        raise ValueError('未知决定状态')
    approval.update(status=status, approved=status == 'approved', resolved_at=timestamp())
    event(record, 'decision', status=status, approved=approval['approved'])
    return True


def terminal_reader(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline() if ready else None


def wait_for_reply(record, save, reader=terminal_reader, clock=time.monotonic):
    if record['approval']['status'] != 'pending':
        raise RuntimeError('本次请求已结束；不能再次等待或改写决定')
    start = clock()
    deadline = start + record['timeout_seconds']
    event(record, 'waiting_for_human', timeout_seconds=record['timeout_seconds'])
    save()
    while True:
        remaining = deadline-clock()
        if remaining <= 0:
            resolve(record, 'timeout')
            break
        line = reader(remaining)
        elapsed = clock()-start
        if line is None or elapsed >= record['timeout_seconds']:
            if line is not None:
                event(record, 'late_input_ignored')
            resolve(record, 'timeout')
            break
        if line == '':
            resolve(record, 'eof')
            break
        text = line.strip()
        normalized = text.upper()
        event(record, 'terminal_reply', text=text, elapsed_seconds=elapsed)
        if normalized in APPROVE:
            resolve(record, 'approved')
            break
        if normalized in REJECT:
            resolve(record, 'rejected')
            break
        event(record, 'invalid_reply', elapsed_seconds=elapsed)
        print('未识别该回复。请输入 同意/APPROVE 或 拒绝/REJECT；倒计时不会重置。', flush=True)
        save()
    record['wait_elapsed_seconds'] = clock()-start
    save()


def finish_action(record, directory):
    target = Path(directory)/TARGET_NAME
    if record.get('action_checked'):
        raise RuntimeError('本次动作已核对；不重复写入')
    approved = record['approval']['status'] == 'approved' and record['approval']['approved'] is True
    if approved:
        content = ('人机协同本地教学结果\n请求编号：'+record['request_id']+
                   '\n本次终端回复明确同意，程序继续生成此文件。\n本次没有模型、MCP、网络或外部业务操作。\n')
        with target.open('x', encoding='utf-8') as output:
            output.write(content)
        actual = target.read_text(encoding='utf-8')
        if actual != content:
            raise ValueError('生成后逐字核对失败')
        record['action'] = {'executed': True, 'filename': TARGET_NAME,
                            'sha256': sha(target.read_bytes()), 'content_matches_expected': True}
        event(record, 'task_resumed_and_file_written', filename=TARGET_NAME)
    else:
        if target.exists():
            raise ValueError('未批准却出现目标文件；停止，不声称正常')
        record['action'] = {'executed': False, 'filename': TARGET_NAME, 'target_absent': True}
        event(record, 'task_stopped_without_target', decision=record['approval']['status'])
    record['action_checked'] = True
    record['status'] = 'completed_'+record['approval']['status']


def new_record(timeout):
    return {'status': 'waiting_for_human', 'created_at': timestamp(), 'request_id': str(uuid.uuid4()),
            'request_message': '是否在本次运行目录生成一份固定的本地教学结果文件？',
            'timeout_seconds': timeout, 'input_source': 'live_terminal_reply',
            'approval': {'status': 'pending', 'approved': False}, 'events': [],
            'model_requests': 0, 'mcp_requests': 0, 'network_requests': 0, 'notifications_sent': 0,
            'scope': 'one human terminal decision gating one new local teaching file; no model decision or production permission system'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--timeout', type=float, default=60, help='等待本次人工回复的秒数，默认60')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        parser.error('--timeout必须在1到3600秒之间')
    manifest_sha = check_sources()
    print('4-5第一步：终端人工决定→任务继续或停止。')
    print('无模型、MCP、网络、通知、安装或下载；本次是本地教学流程。')
    if args.preview:
        print('HITL_LOCAL_PREVIEW_ONLY：来源及本地入口检查通过，不等待输入、不创建运行目录。')
        return 0
    directory = ROOT/'运行结果'/('terminal_hitl_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    record = new_record(args.timeout)
    record.update(script_sha256=sha(Path(__file__).read_bytes()), manifest_sha256=manifest_sha,
                  python=sys.version, executable=sys.executable, target_path=str(directory/TARGET_NAME))

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    event(record, 'request_created', request_id=record['request_id'], target_path=record['target_path'])
    save()
    print('请求编号：', record['request_id'])
    print('待决定：', record['request_message'])
    print('批准后生成的文件：', record['target_path'])
    print('记录路径：', path)
    print('审批记录会先保存；需要你同意才生成的是 教学结果.txt。')
    print(f'请在{args.timeout:g}秒内输入 同意/APPROVE 或 拒绝/REJECT，然后回车。空回车无效，超时不批准。', flush=True)
    exit_code = 0
    try:
        if not sys.stdin.isatty():
            resolve(record, 'input_unavailable')
            print('当前不是可交互终端，停止。请使用“在终端中运行Python文件”。')
        else:
            wait_for_reply(record, save)
        finish_action(record, directory)
    except KeyboardInterrupt:
        resolve(record, 'cancelled')
        target = directory/TARGET_NAME
        record.update(status='interrupted', action={'executed': target.exists(),
                      'filename': TARGET_NAME, 'interrupted_requires_review': target.exists()})
        event(record, 'run_interrupted', target_exists=target.exists())
        print('本次运行已中断，不继续生成文件或自动重新请求。')
        exit_code = 130
    except Exception as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print('本步未完成：', record['error'])
        exit_code = 1
    finally:
        save()
        print('决定：', record['approval']['status'], '；是否批准：', record['approval']['approved'])
        print('目标文件是否生成：', record.get('action', {}).get('executed', False))
        print('状态：', record['status'])
        print('HITL_LOCAL_RUN_SAVED：真实回复、状态变化和产物核对已记录；按状态判断继续或停止。')
        print('记录路径：', path)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

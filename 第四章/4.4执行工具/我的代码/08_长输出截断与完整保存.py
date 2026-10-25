"""固定输出1000行→完整保存→仅返回头尾→按行取回中间证据；无模型。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CHINA = timezone(timedelta(hours=8))
CODE = '''for i in range(1000):
    if i == 500:
        print(f"log {i:04d}: CHECKPOINT middle-only teaching marker")
    else:
        print(f"log {i:04d}: completed")
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_log(path, max_lines=200, head_lines=5, tail_lines=5):
    if any(type(n) is not int or n < 0 for n in (max_lines, head_lines, tail_lines)):
        raise ValueError('行数配置须为非负整数')
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    total = len(lines)
    # 不将末尾换行误计为额外空行；覆盖有重叠时避免重复。
    shortened = total > max_lines and head_lines+tail_lines < total
    omitted = total-head_lines-tail_lines if shortened else 0
    if shortened:
        head = lines[:head_lines] if head_lines else []
        tail = lines[-tail_lines:] if tail_lines else []
        excerpt = '\n'.join(head+[f'... [省略{omitted}行；完整输出见stdout_file] ...']+tail)
    else:
        excerpt = text
    return {'stdout': excerpt, 'stdout_file': str(path.resolve()), 'truncated': shortened,
            'original_line_count': total, 'omitted_line_count': omitted,
            'kept_source_lines': total-omitted, 'full_output_sha256': sha(path),
            'policy': {'line_threshold': max_lines, 'head_lines': head_lines, 'tail_lines': tail_lines}}


def read_log_lines(path, start_line, end_line):
    """1起算且包含两端；只对调用方已知日志路径取片段，不开放其他路径工具。"""
    if type(start_line) is not int or type(end_line) is not int or start_line < 1 or end_line < start_line:
        raise ValueError('行号范围无效')
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if end_line > len(lines) or end_line-start_line+1 > 20:
        raise ValueError('行数越界或超过本教学入口20行读取上限')
    return [{'line_number': i, 'text': lines[i-1]} for i in range(start_line, end_line+1)]


def exercise(directory, record, save=lambda: None, verbose=True):
    directory = Path(directory)
    script = directory/'generate_log.py'
    script.write_text(CODE, encoding='utf-8')
    stdout = directory/'stdout_full.txt'
    stderr = directory/'stderr_full.txt'
    command = [sys.executable, '-I', '-B', str(script.resolve())]
    record.update(command=command, generated_script_sha256=sha(script))
    save()
    # 进程输出直接落盘，完整日志不会被摘要覆盖。
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.run(command, cwd=directory, stdout=out, stderr=err, timeout=10,
                                 env={'PATH': str(Path(sys.executable).parent)})
    record.update(returncode=process.returncode, stderr_file=str(stderr.resolve()))
    save()
    if process.returncode != 0:
        raise ValueError('固定日志程序未正常结束')
    before = sha(stdout)
    result = summarize_log(stdout)
    record['tool_result'] = result
    # 程序中的i=500对应文件第501行；行号读取1起算。
    middle = read_log_lines(stdout, 499, 503)
    record['middle_read'] = middle
    lines = stdout.read_text(encoding='utf-8').splitlines()
    expected_lines = ['log '+str(i).zfill(4)+(': CHECKPOINT middle-only teaching marker' if i == 500 else ': completed') for i in range(1000)]
    checks = {'program_exited_zero': process.returncode == 0,
              'saved_exactly_1000_lines': len(lines) == 1000,
              'all_lines_match_fixed_program': lines == expected_lines,
              'truncation_triggered': result['truncated'],
              'kept_head_tail_and_omitted_count': result['kept_source_lines'] == 10 and result['omitted_line_count'] == 990,
              'correct_excerpt': result['stdout'].splitlines()[:5] == lines[:5] and result['stdout'].splitlines()[-5:] == lines[-5:],
              'marker_omitted_in_excerpt': 'CHECKPOINT' not in result['stdout'],
              'marker_retrieved_from_full_file': any(r['line_number'] == 501 and 'CHECKPOINT' in r['text'] for r in middle),
              'saved_file_unchanged_after_summary_and_read': before == sha(stdout) == result['full_output_sha256'],
              'stderr_empty': stderr.stat().st_size == 0}
    record['checks'] = checks
    save()
    if verbose:
        print('工具返回的头尾片段：\n'+result['stdout'], flush=True)
        print('完整日志：', result['stdout_file'])
        print('完整1000行；保留头尾各5行，省略990行。')
        print('按文件行号取回第499—503行：', json.dumps(middle, ensure_ascii=False))
        print('核对：', json.dumps(checks, ensure_ascii=False))
    if not all(checks.values()):
        raise ValueError('长输出保存或核对失败')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print('4-4第四步：长输出只展示头尾，完整日志保存并核对。')
    print('固定1000行，超过200行就展示头尾各5行；无模型、MCP或网络请求。')
    print('中间有一个教学标记，用于观察头尾截断会遗漏内容，但可从完整文件取回。')
    if args.preview:
        compile(CODE, 'generate_log.py', 'exec')
        print('LONG_OUTPUT_PREVIEW_ONLY：生成代码检查通过，未执行或创建正式结果。')
        return 0
    if args.check:
        with tempfile.TemporaryDirectory(prefix='long_output_check_') as temp:
            record = {}
            exercise(temp, record, verbose=False)
            print('LONG_OUTPUT_OFFLINE_CHECK_PASSED：', record['checks'])
        return 0
    directory = ROOT/'运行结果'/('long_output_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
              'script_sha256': sha(Path(__file__)), 'python': sys.version, 'executable': sys.executable,
              'model_requests': 0, 'mcp_requests': 0, 'network_requests': 0,
              'scope': 'fixed 1000-line output; local truncation/persistence and bounded reread; no semantic summary or general log analysis'}

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    save()
    print('记录路径：', path, flush=True)
    started = time.perf_counter()
    try:
        exercise(directory, record, save)
        record['status'] = 'long_output_preserved_and_reread_verified'
        print('LONG_OUTPUT_RUN_SAVED：头尾片段、完整输出与按行读取均已保存并核对。')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print('本步未完成：', record['error'])
        return 1
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        save()
        print('记录路径：', path)


if __name__ == '__main__':
    raise SystemExit(main())

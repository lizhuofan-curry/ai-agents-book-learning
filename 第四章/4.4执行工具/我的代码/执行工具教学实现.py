"""固定教学代码的写入、子进程运行和独立结果核对；不提供任意代码执行器。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SAMPLE = 'apple banana apple cherry banana apple date cherry banana apple\n'
EXPECTED = {'apple': 4, 'banana': 3, 'cherry': 2, 'date': 1}
GOOD = '''from collections import Counter
import json
from pathlib import Path

words = Path('data.txt').read_text(encoding='utf-8').split()
counts = dict(Counter(words))
Path('counts.json').write_text(json.dumps(counts, sort_keys=True)+'\\n', encoding='utf-8')
print(json.dumps(counts, sort_keys=True))
'''
WRONG = GOOD.replace("counts = dict(Counter(words))", "counts = dict(Counter(words))\ncounts['apple'] = 999")
BROKEN = 'def broken(:\n    return 1\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_write(workspace, filename, content):
    try:
        root = Path(workspace).resolve(strict=True)
        name = Path(filename)
        if name.is_absolute() or len(name.parts) != 1 or '..' in name.parts:
            raise ValueError('只接受工作区内单个相对文件名')
        path = root/name
        if path.exists() or path.is_symlink():
            raise FileExistsError('教学入口不覆盖已有文件')
        if not isinstance(content, str):
            raise ValueError('写入内容须为文本')
        if path.suffix == '.py':
            compile(content, filename, 'exec')
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
        if path.read_text(encoding='utf-8') != content:
            raise ValueError('写入内容核对失败')
        return {'success': True, 'path': str(path), 'bytes_written': path.stat().st_size,
                'syntax_check': 'passed' if path.suffix == '.py' else 'not_applicable', 'sha256': sha(path)}
    except (OSError, ValueError, SyntaxError, TypeError) as error:
        return {'success': False, 'error_type': type(error).__name__, 'error': str(error)}


def run_fixed_python(workspace, filename):
    # 只执行本步骤已知两段代码，源码哈希不符就停止；工作目录隔离不是OS安全沙盒。
    allowed = {'wordcount.py': GOOD, 'wrong_count.py': WRONG}
    if filename not in allowed:
        return {'success': False, 'error_type': 'ValueError', 'error': '不在固定教学脚本清单内'}
    root = Path(workspace).resolve(strict=True)
    script = root/filename
    if script.is_symlink() or script.read_text(encoding='utf-8') != allowed[filename]:
        return {'success': False, 'error_type': 'ValueError', 'error': '固定教学代码发生变化'}
    if (root/'data.txt').is_symlink() or (root/'data.txt').read_text(encoding='utf-8') != SAMPLE:
        return {'success': False, 'error_type': 'ValueError', 'error': '固定输入数据发生变化'}
    if (root/'counts.json').exists() or (root/'counts.json').is_symlink():
        return {'success': False, 'error_type': 'FileExistsError', 'error': '输出文件已存在，拒绝覆盖'}
    command = [sys.executable, '-I', '-B', str(script)]
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                encoding='utf-8', timeout=10, env={'PATH': str(Path(sys.executable).parent)})
        return {'success': result.returncode == 0, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr, 'command': command,
                'script_sha256': sha(script), 'output_exists': (root/'counts.json').is_file()}
    except subprocess.TimeoutExpired as error:
        return {'success': False, 'error_type': type(error).__name__, 'error': '固定脚本运行超时'}


def verify_counts(workspace):
    root = Path(workspace)
    path = root/'counts.json'
    try:
        actual = json.loads(path.read_text(encoding='utf-8'))
        # 独立按输入中逐词出现次数重新计算，不调用被测脚本的Counter。
        words = (root/'data.txt').read_text(encoding='utf-8').split()
        independent = {word: words.count(word) for word in set(words)}
        exact = (type(actual) is dict and set(actual) == set(independent)
                 and all(type(actual[k]) is int and actual[k] == independent[k] for k in independent))
        return {'goal_correct': exact, 'actual': actual, 'expected': independent,
                'total_words': len(words), 'result_sha256': sha(path),
                'scope': 'exact token counts for this fixed whitespace-separated input'}
    except (OSError, ValueError, TypeError) as error:
        return {'goal_correct': False, 'error_type': type(error).__name__, 'error': str(error)}

"""固定调用执行工具：写代码→检查语法→真实运行→独立验证结果。"""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name('执行工具教学实现.py')
spec = importlib.util.spec_from_file_location('execution_demo_tools', HELPER)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
CHINA = timezone(timedelta(hours=8))


def exercise(workspace, record, save=lambda: None, verbose=True):
    root = Path(workspace)

    def step(name, action):
        result = action()
        record.setdefault('steps', []).append({'step': name, 'result': result})
        save()
        if verbose:
            print(name+'：', json.dumps(result, ensure_ascii=False), flush=True)
        return result

    good = root/'正确版本'
    wrong = root/'错误版本'
    good.mkdir()
    wrong.mkdir()
    for folder in (good, wrong):
        result = tools.file_write(folder, 'data.txt', tools.SAMPLE)
        if not result['success']:
            raise ValueError('写入固定输入失败')
        record.setdefault('inputs', []).append(result)
    written = step('1. 写入合法Python代码', lambda: tools.file_write(good, 'wordcount.py', tools.GOOD))
    rejected = step('2. 尝试写入语法错误代码', lambda: tools.file_write(good, 'broken.py', tools.BROKEN))
    run = step('3. 运行正确版本', lambda: tools.run_fixed_python(good, 'wordcount.py'))
    checked = step('4. 核对生成的词频文件', lambda: tools.verify_counts(good))
    wrong_written = tools.file_write(wrong, 'wrong_count.py', tools.WRONG)
    record['wrong_script_write'] = wrong_written
    wrong_run = step('5. 运行故意算错的版本', lambda: tools.run_fixed_python(wrong, 'wrong_count.py'))
    wrong_check = step('6. 核对错误版本的结果', lambda: tools.verify_counts(wrong))
    checks = {'valid_script_written': written['success'],
              'syntax_error_rejected_before_write': rejected.get('error_type') == 'SyntaxError' and not (good/'broken.py').exists(),
              'correct_program_exited_zero': run.get('returncode') == 0,
              'correct_output_matches_goal': checked['goal_correct'],
              'wrong_program_also_exited_zero': wrong_run.get('returncode') == 0,
              'wrong_output_detected': wrong_check['goal_correct'] is False,
              'controlled_wrong_case': wrong_check.get('actual', {}).get('apple') == 999}
    record['checks'] = checks
    if not wrong_written['success'] or not all(checks.values()):
        raise ValueError('执行工具教学检查未全部满足')
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print('4-4第一步：执行工具写入、运行及结果核对。')
    print('固定词频任务：apple=4、banana=3、cherry=2、date=1，总词数10。')
    print('无模型、MCP或网络请求；先观察本地工具内部执行，不测试模型决策。')
    print('包含故意的语法错误和计数错误，它们被发现才符合本练习预期。')
    if args.preview:
        compile(tools.GOOD, 'wordcount.py', 'exec')
        compile(tools.WRONG, 'wrong_count.py', 'exec')
        print('EXECUTION_PREVIEW_ONLY：入口与两段可运行代码检查通过，未执行或写入正式结果。')
        return 0
    if args.check:
        with tempfile.TemporaryDirectory(prefix='execution_teaching_check_') as temp:
            record = {}
            exercise(temp, record, verbose=False)
            print('EXECUTION_OFFLINE_CHECK_PASSED：', record['checks'])
        return 0
    directory = ROOT/'运行结果'/('execution_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    workspace = directory/'工作区'
    workspace.mkdir()
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(),
              'script_sha256': tools.sha(Path(__file__)), 'helper_sha256': tools.sha(HELPER),
              'python': sys.version, 'executable': sys.executable, 'workspace': str(workspace),
              'model_requests': 0, 'network_requests': 0, 'mcp_requests': 0,
              'scope': 'fixed local execution tool functions and two known scripts; not arbitrary sandbox or official full experiment'}

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    save()
    print('记录路径：', path, flush=True)
    started = time.perf_counter()
    try:
        exercise(workspace, record, save)
        record['status'] = 'fixed_execution_checks_passed'
        print('EXECUTION_TOOLS_RUN_SAVED：正确输出与两种受控错误均已核对，记录保存。')
        print('打开工作区，可查看脚本、输入及counts.json。')
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

"""复制已核对的问候脚本，让教材Agent读取、编辑并执行需求变更。"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

# 复用第二步传输适配和范围检查；不改动第二步脚本及其历史产物。
import importlib.util
spec = importlib.util.spec_from_file_location('coding_base', Path(__file__).with_name('02_官方主循环本机编码任务.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ROOT = base.ROOT
BASELINE = ROOT/'运行结果/local_coding_task_20261101_161002_541836'
BASELINE_RECORD_SHA = 'a3190d30c346e2e29d0505ed1ecae83014755f89190080d602ce5b43b95d15cb'
BASELINE_CODE_SHA = '7ff441173a407e892ee6abe5a36bc2fe8b17d8df81a6d2592479fd07a118e07e'
TOOLS = {'Read', 'Edit', 'Bash'}
EXPECTED_LINES = ['Hello, World!', 'Hello, Alice!', 'Hello, World!']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline():
    record_path = BASELINE/'record.json'
    code_path = BASELINE/'练习项目/hello_world.py'
    if sha(record_path) != BASELINE_RECORD_SHA or sha(code_path) != BASELINE_CODE_SHA:
        raise ValueError('指定的已核对基线已变化，停止；不重跑或覆盖旧实验')
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if not all(record['completion_checks'].values()):
        raise ValueError('基线完成检查未通过')
    base.validate_code(code_path.read_text(encoding='utf-8'))
    return code_path.read_bytes()


class RevisionRegistry(base.TaskRegistry):
    def get_tool(self, name, state):
        if name not in TOOLS:
            self.record['tool_dispatch_attempts'] = self.record.get('tool_dispatch_attempts', 0) + 1
            self.save()
            raise ValueError('本次只提供Read、Edit、Bash，不提供Write重建文件')
        tool = super().get_tool(name, state)

        def execute(params):
            observations = self.record.setdefault('tool_observations', [])
            request_number = len(self.record['model_calls'])
            if name == 'Edit' and not any(
                x['tool'] == 'Read' and x['success'] and x['model_request_number'] < request_number
                for x in observations
            ):
                raise ValueError('先Read并在后一轮收到真实文件内容，再Edit；不要同轮猜测替换')
            target = self.workspace/'hello_world.py'
            before = sha(target)
            result = tool.execute(params)
            data = result.to_dict()
            observations.append({
                'tool': name, 'model_request_number': request_number,
                'source_sha256_before': before, 'source_sha256_after': sha(target),
                'success': 'error' not in data and data.get('success', True) is not False,
                'result': deepcopy(data),
            })
            self.save()
            return result
        return SimpleNamespace(execute=execute)


def check_artifact(workspace):
    target = workspace/'hello_world.py'
    source = target.read_text(encoding='utf-8')
    tree = base.validate_code(source)
    process = subprocess.run([sys.executable, '-I', str(target)], cwd=workspace,
                             capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    # 独立检查函数返回值，防止只改main的打印而遗漏空名字逻辑。
    verify = ('import runpy,sys,json,contextlib,io\n'
              'buffer=io.StringIO()\n'
              'with contextlib.redirect_stdout(buffer): n=runpy.run_path(sys.argv[1],run_name="verified_module")\n'
              'import_output=buffer.getvalue(); f=n.get("greet")\n'
              'cases=[("Bob","Hello, Bob!"),("","Hello, World!"),("小明","Hello, 小明!"),(" ","Hello,  !")]\n'
              'values={x:bool(callable(f) and f(x)==y) for x,y in cases}\n'
              'buffer=io.StringIO()\n'
              'with contextlib.redirect_stdout(buffer): n["main"]()\n'
              'print(json.dumps({"returns":values,"import_stdout":import_output,"main_stdout":buffer.getvalue()},ensure_ascii=False))')
    function = subprocess.run([sys.executable, '-I', '-c', verify, str(target)], cwd=workspace,
                              capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    ast = base.ast
    checks = {
        'script_exit_zero': process.returncode == 0,
        'script_output_matches': process.stdout.splitlines() == EXPECTED_LINES,
        'has_main_guard': any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__'
            and any(isinstance(x, ast.Constant) and x.value == '__main__' for x in n.test.comparators) for n in tree.body),
    }
    try:
        result = json.loads(function.stdout)
        checks['greet_return_values'] = (function.returncode == 0
            and set(result['returns']) == {'Bob', '', '小明', ' '} and all(result['returns'].values()))
        checks['main_function_behavior'] = result['main_stdout'].splitlines() == EXPECTED_LINES
        checks['import_has_no_output'] = result['import_stdout'] == ''
    except (ValueError, KeyError, TypeError):
        checks.update(greet_return_values=False, main_function_behavior=False, import_has_no_output=False)
    return {'source': source, 'sha256': sha(target),
            'script_run': {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr},
            'function_run': {'returncode': function.returncode, 'stdout': function.stdout, 'stderr': function.stderr},
            'checks': checks, 'passed': all(checks.values())}


def completion_checks(record):
    events = record['events']
    observations = record.get('tool_observations', [])
    edits = [x for x in observations if x['tool'] == 'Edit' and x['success']
             and x['result'].get('replacements', 0) > 0
             and x['source_sha256_before'] != x['source_sha256_after']]
    last_edit = edits[-1] if edits else None
    final_sha = record['artifact_verification']['sha256']
    return {
        'agent_reported_done': any(e['type'] == 'done' for e in events),
        'no_engine_error_or_budget_stop': not any(e['type'] in {'error', 'max_iterations_reached'} for e in events),
        'within_tool_dispatch_budget': record.get('tool_dispatch_attempts', 0) <= 12,
        'model_read_before_edit': bool(last_edit and any(x['tool'] == 'Read' and x['success']
            and x['model_request_number'] < last_edit['model_request_number'] for x in observations)),
        'model_actually_edited': bool(edits),
        'model_ran_final_revision': bool(last_edit and any(x['tool'] == 'Bash'
            and x['model_request_number'] > last_edit['model_request_number']
            and x['source_sha256_before'] == final_sha and x['source_sha256_after'] == final_sha
            and x['result'].get('exit_code') == 0
            and x['result'].get('output', '').splitlines() == EXPECTED_LINES for x in observations)),
        'baseline_unchanged': sha(BASELINE/'record.json') == BASELINE_RECORD_SHA
            and sha(BASELINE/'练习项目/hello_world.py') == BASELINE_CODE_SHA,
        'independent_artifact_checks': record['artifact_verification']['passed'],
    }


def run_task(record, save, workspace, post=base.local_post):
    reference = ROOT/'官方资料/coding-agent'
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    sys.path.insert(0, str(reference))
    from agent import CodingAgent
    old_cwd = Path.cwd()
    old_ceiling = os.environ.get('GIT_CEILING_DIRECTORIES')
    os.chdir(workspace)
    os.environ['GIT_CEILING_DIRECTORIES'] = str(workspace.parent)
    agent = None
    try:
        agent = CodingAgent(api_key='ollama', model=base.MODEL,
                            base_url='http://127.0.0.1:11434/v1', provider='openai')
        agent.client.close()
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=base.NativeCompletions(record, save, post)))
        instruction = ('本次为已有问候脚本的需求变更，只提供Read、Edit、Bash，其他工具包括TodoWrite均未提供。'
            '先Read，收到真实内容后在下一轮Edit，最后在后续轮次Bash验证；不要重建整个文件。'
            '只访问当前目录hello_world.py；Bash仅运行python3 hello_world.py，也可python或当前解释器，'
            '默认10000毫秒、最多10000毫秒，前台执行，不串联命令。代码仅使用greet/main/print、字符串、'
            'if条件与相等比较（空名条件可用name == ""），无需import、属性调用、not或其他语法。只有真实读改测后才能报告完成。')
        agent.system_prompt += '\n\n' + instruction
        agent.tools = [deepcopy(x) for x in agent.tools if x['name'] in TOOLS]
        for tool in agent.tools:
            tool['description'] += ' ' + instruction
            if tool['name'] == 'Bash':
                properties = tool['input_schema']['properties']
                properties['timeout'].update(minimum=1, maximum=10000, default=10000)
                properties['run_in_background'].update(enum=[False], default=False)
        agent.tool_registry = RevisionRegistry(agent.tool_registry, workspace, record, save)
        record.update(effective_system_prompt=agent.system_prompt, model_visible_tools=deepcopy(agent.tools))
        save()
        for event in agent.run(record['task'], max_iterations=base.MAX_REQUESTS):
            record['events'].append(deepcopy(event))
            record['messages'] = deepcopy(agent.messages)
            save()
            if event['type'] == 'tool_call':
                print('模型选择：' + event['tool'] + ' ' + json.dumps(event['input'], ensure_ascii=False), flush=True)
            elif event['type'] == 'tool_execution_complete':
                print('工具返回：' + json.dumps(event['result'], ensure_ascii=False), flush=True)
            elif event['type'] == 'text_delta':
                print(event['delta'], end='', flush=True)
            elif event['type'] in {'error', 'max_iterations_reached'}:
                print('\n停止：' + json.dumps(event, ensure_ascii=False), flush=True)
        record['artifact_verification'] = check_artifact(workspace)
        record['completion_checks'] = completion_checks(record)
        record['status'] = ('revision_checked_review_pending' if all(record['completion_checks'].values())
                            else 'finished_with_failed_checks')
        save()
    finally:
        if agent:
            record['messages'] = deepcopy(agent.messages)
            for session in agent.system_state.shell_sessions.values():
                process = session.process
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                    for pipe in (process.stdin, process.stdout, process.stderr):
                        if pipe is not None and not pipe.closed:
                            pipe.close()
                    session.process = None
        os.chdir(old_cwd)
        if old_ceiling is None:
            os.environ.pop('GIT_CEILING_DIRECTORIES', None)
        else:
            os.environ['GIT_CEILING_DIRECTORIES'] = old_ceiling
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if Path(sys.prefix).resolve() != base.SDK_ENV.resolve():
        python = base.SDK_ENV/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        os.execv(str(python), [str(python), str(Path(__file__)), *sys.argv[1:]])
    offline = base.load(Path(__file__).with_name('01_官方Starter离线检查.py'), 'coding_entry_check')
    _, _, manifest_sha, count = offline.prepare()
    initial = baseline()
    print('第五章第三步：复制已核对产物，Read/Edit/Bash完成空名字回退需求。')
    print('最多6次qwen3.5:9b本机请求、12次工具分发；输出2048、上下文16384；无重试、云端或模型下载。')
    if args.preview:
        print(f'CODING_REVISION_PREVIEW_ONLY：{count}份参考与基线哈希通过；0次模型/工具，无结果目录。')
        return 0
    directory = ROOT/'运行结果'/('local_coding_revision_' + base.datetime.now(base.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    workspace = directory/'练习项目'
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace/'hello_world.py').write_bytes(initial)
    path = directory/'record.json'
    task = (f'当前练习目录{workspace}已有hello_world.py。需求变更：greet(name)在name为空字符串时'
            '返回Hello, World!；其余名字保持原样拼接（空格不算空字符串）。main保持World、Alice的'
            '两次演示，并在末尾增加print(greet(""))，直接运行应依次输出Hello, World!、Hello, Alice!、'
            'Hello, World!三行。保留main入口条件。先Read查看真实文件，在收到结果后的下一轮使用Edit'
            '做局部替换，再在后续轮次Bash执行python3 hello_world.py，根据真实输出报告结果。'
            '只修改此文件，不用Write重建，不创建其他文件，不运行其他命令；无需import。最后简洁中文说明。')
    record = {'status': 'running', 'created_at': base.datetime.now(base.CHINA).isoformat(), 'task': task,
        'manifest_sha256': manifest_sha, 'script_sha256': sha(Path(__file__)),
        'adapter_script_sha256': sha(Path(__file__).with_name('02_官方主循环本机编码任务.py')),
        'baseline_record': str(BASELINE/'record.json'), 'baseline_record_sha256': BASELINE_RECORD_SHA,
        'baseline_code_sha256': BASELINE_CODE_SHA, 'initial_source': initial.decode('utf-8'),
        'model': base.MODEL, 'endpoint': base.URL, 'model_calls': [], 'events': [], 'messages': [],
        'tool_observations': [], 'maximum_model_requests': 6, 'maximum_tool_dispatches': 12,
        'automatic_retries': 0, 'cloud_requests': 0,
        'scope': 'official main loop and Read/Edit/Bash with fixed local task; model edits a copied artifact; not independent handwritten practice or error-repair evidence'}

    def save():
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.replace(path)
    save()
    print('记录路径：' + str(path), flush=True)
    try:
        run_task(record, save, workspace)
        print('\n验收：' + json.dumps(record['completion_checks'], ensure_ascii=False))
        print('CODING_REVISION_SAVED：真实轨迹与新副本已保存；仍需核对解释，不计作本人手写修改。')
        return 0 if all(record['completion_checks'].values()) else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('本步未完成，不重试：' + str(exc))
        return 1
    finally:
        save()
        print('记录路径：' + str(path), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())

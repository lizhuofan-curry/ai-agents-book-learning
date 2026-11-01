"""第三步专项检查：合成模型响应＋真实教材Read/Edit/Bash；0次网络请求。"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
ENV = ROOT.parents[1]/'.venv-ch5'
if Path(sys.prefix).resolve() != ENV.resolve():
    python = ENV/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    os.execv(str(python), [str(python), str(Path(__file__)), *sys.argv[1:]])
spec = importlib.util.spec_from_file_location('revision_step', ROOT/'我的代码/03_读取并修改已有代码.py')
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def response(calls=()):
    message = {'role': 'assistant', 'content': '' if calls else '合成测试完成'}
    if calls:
        message['tool_calls'] = [{'function': {'name': name, 'arguments': args}} for name, args in calls]
    return {'model': step.base.MODEL, 'done': True, 'done_reason': 'stop', 'message': message}


class RevisionChecks(unittest.TestCase):
    def run_fake(self, workspace, bad_edit=False):
        record = {'task': '合成测试：空名字回退World', 'model_calls': [], 'events': [], 'messages': [], 'test_only': True}
        requests = []
        initial = step.baseline().decode('utf-8')
        path = str(workspace/'hello_world.py')

        def fake(payload, entry, save):
            requests.append(payload)
            number = len(requests)
            if number == 1:
                self.assertEqual({x['function']['name'] for x in payload['tools']}, step.TOOLS)
                return response([('Read', {'file_path': path})])
            if number == 2:
                tool = [m for m in payload['messages'] if m['role'] == 'tool'][-1]
                self.assertEqual(tool['tool_name'], 'Read')
                self.assertIn('return "Hello, " + name + "!"', json.loads(tool['content'])['content'])
                if bad_edit:
                    return response([('Edit', {'file_path': path, 'old_string': 'NOT_PRESENT', 'new_string': 'anything'})])
                return response([
                    ('Edit', {'file_path': path, 'old_string': 'def greet(name):\n',
                              'new_string': 'def greet(name):\n    if name == "":\n        name = "World"\n'}),
                    ('Edit', {'file_path': path, 'old_string': '    print(greet("Alice"))',
                              'new_string': '    print(greet("Alice"))\n    print(greet(""))'}),
                ])
            if number == 3:
                tools = [json.loads(m['content']) for m in payload['messages'] if m['role'] == 'tool']
                if bad_edit:
                    self.assertIn('String not found', tools[-1]['error'])
                    return response()
                self.assertEqual([x.get('replacements') for x in tools[-2:]], [1, 1])
                return response([('Bash', {'command': 'python3 hello_world.py'})])
            tool = [m for m in payload['messages'] if m['role'] == 'tool'][-1]
            self.assertEqual(tool['tool_name'], 'Bash')
            self.assertEqual(json.loads(tool['content'])['output'].splitlines(), step.EXPECTED_LINES)
            return response()

        (workspace/'hello_world.py').write_text(initial, encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            step.run_task(record, lambda: None, workspace, fake)
        return record, requests

    def test_real_loop_read_edit_execute_and_feedback(self):
        with tempfile.TemporaryDirectory(prefix='ch5-revision-offline-') as folder:
            record, requests = self.run_fake(Path(folder).resolve())
            self.assertEqual(len(requests), 4)
            self.assertTrue(all(record['completion_checks'].values()))
            self.assertEqual([x['tool'] for x in record['tool_observations']], ['Read', 'Edit', 'Edit', 'Bash'])
            self.assertTrue(record['artifact_verification']['checks']['greet_return_values'])
            # 实际执行的内容必须与最终文件一致；篡改此证据后不能通过。
            record['tool_observations'][-1]['source_sha256_before'] = 'stale'
            self.assertFalse(step.completion_checks(record)['model_ran_final_revision'])

    def test_failed_edit_is_returned_and_not_counted_as_change(self):
        with tempfile.TemporaryDirectory(prefix='ch5-revision-fail-') as folder:
            workspace = Path(folder).resolve()
            record, _ = self.run_fake(workspace, bad_edit=True)
            self.assertEqual(step.sha(workspace/'hello_world.py'), step.BASELINE_CODE_SHA)
            self.assertFalse(record['completion_checks']['model_actually_edited'])
            self.assertFalse(record['completion_checks']['independent_artifact_checks'])
            self.assertEqual(record['status'], 'finished_with_failed_checks')

    def test_write_and_edit_without_read_rejected(self):
        # 原注册表在get_tool时只创建工具；用不会执行的对象检查包装层。
        class Original:
            def get_tool(self, *args):
                return object()
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder).resolve()
            (workspace/'hello_world.py').write_bytes(step.baseline())
            registry = step.RevisionRegistry(Original(), workspace, {'model_calls': [{}]}, lambda: None)
            with self.assertRaises(ValueError):
                registry.get_tool('Write', None)
            with self.assertRaisesRegex(ValueError, '先Read'):
                registry.get_tool('Edit', None).execute({'file_path': str(workspace/'hello_world.py')})

    def test_correct_printing_does_not_replace_function_check(self):
        source = step.baseline().decode('utf-8').replace('    print(greet("Alice"))',
            '    print(greet("Alice"))\n    print("Hello, World!")')
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder).resolve()
            (workspace/'hello_world.py').write_text(source, encoding='utf-8')
            check = step.check_artifact(workspace)
            self.assertTrue(check['checks']['script_output_matches'])
            self.assertFalse(check['checks']['greet_return_values'])
            self.assertFalse(check['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

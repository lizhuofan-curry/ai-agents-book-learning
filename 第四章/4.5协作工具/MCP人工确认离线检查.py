"""测试服务端状态及真实stdio通信；测试回复明确为自动化输入。"""
import asyncio
import importlib.util
import importlib.metadata
import os
import sys
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'我的代码'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
server = load('hitl_server_test', '02_人工确认SDK服务.py')
client = load('hitl_client_test', '03_通过MCP提交人工决定.py')


class Checks(unittest.TestCase):
    def test_id_pending_and_duplicate_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            store = server.ApprovalStore(directory)
            req = store.create_request()
            self.assertFalse(store.finish_request(req['request_id'])['success'])
            self.assertFalse(store.submit_decision('wrong', 'approved')['success'])
            self.assertFalse(store.create_request()['success'])
            self.assertFalse(store.submit_decision(req['request_id'], 'unknown')['success'])
            self.assertTrue(store.submit_decision(req['request_id'], 'approved')['success'])
            self.assertFalse(store.submit_decision(req['request_id'], 'rejected')['success'])
            self.assertTrue(store.finish_request(req['request_id'])['action']['executed'])
            self.assertFalse(store.finish_request(req['request_id'])['success'])

    def test_server_deadline_rejects_late_approve(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [0]
            store = server.ApprovalStore(directory, lambda: now[0])
            req = store.create_request(2)
            self.assertFalse(store.submit_decision(req['request_id'], 'timeout')['success'])
            now[0] = 3
            result = store.submit_decision(req['request_id'], 'approved')
            self.assertFalse(result['success'])
            self.assertEqual(result['current_status'], 'timeout')
            self.assertFalse(store.finish_request(req['request_id'])['action']['executed'])

    def test_reject_does_not_write_target(self):
        with tempfile.TemporaryDirectory() as directory:
            store = server.ApprovalStore(directory)
            req = store.create_request()
            store.submit_decision(req['request_id'], 'rejected')
            self.assertFalse(store.finish_request(req['request_id'])['action']['executed'])
            self.assertFalse((Path(directory)/'MCP教学结果.txt').exists())

    def test_existing_target_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            store = server.ApprovalStore(directory)
            req = store.create_request()
            store.submit_decision(req['request_id'], 'approved')
            target = Path(directory)/'MCP教学结果.txt'
            target.write_text('existing')
            with self.assertRaises(FileExistsError):
                store.finish_request(req['request_id'])
            self.assertEqual(target.read_text(), 'existing')

    def mcp_case(self, decision):
        with tempfile.TemporaryDirectory() as directory:
            record = {'test_input_source': 'automated fixture, not real human reply'}
            def input_fn(human, save):
                human['input_source'] = 'automated_test_fixture_not_human'
                if decision == 'timeout':
                    import time
                    time.sleep(1.1)
                client.local.resolve(human, decision)
                save()
            asyncio.run(asyncio.wait_for(client.exercise(record, lambda: None, directory, 1 if decision == 'timeout' else 5, input_fn), 25))
            self.assertEqual(record['status'], 'mcp_hitl_completed_'+decision)
            self.assertEqual(len(record['calls']), 3)
            self.assertTrue(all(record['checks'].values()))
            self.assertEqual((Path(directory)/'MCP教学结果.txt').exists(), decision == 'approved')

    def test_real_stdio_approved(self):
        self.mcp_case('approved')

    def test_real_stdio_rejected(self):
        self.mcp_case('rejected')

    def test_real_stdio_timeout(self):
        self.mcp_case('timeout')


if __name__ == '__main__':
    if Path(sys.prefix).resolve() != client.SDK_ENV.resolve():
        if not client.PYTHON.is_file():
            raise RuntimeError('未找到已配置的第四章SDK环境')
        print('切换已有第四章SDK环境运行检查；不安装依赖。', flush=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.execv(str(client.PYTHON), [str(client.PYTHON), str(Path(__file__)), *sys.argv[1:]])
    if importlib.metadata.version('mcp') != '2.3.0':
        raise ValueError('需固定mcp 2.3.0')
    unittest.main()

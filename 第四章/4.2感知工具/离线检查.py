import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/'我的代码'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load('reader_server', '文件读取服务.py')
client = load('reader_client', '01_观察MCP文件读取.py')


class FileProtocolChecks(unittest.TestCase):
    def test_actual_stdio_lifecycle(self):
        record = {}
        client.exercise(record)
        self.assertTrue(all(record['checks'].values()))
        self.assertEqual(record['server_exit_code'], 0)
        self.assertEqual(len(record['transcript']), 9)

    def test_truncated_result_is_explicit(self):
        result = server.read_file({'file_path': '工具说明.txt', 'max_length': 5})
        data = json.loads(result['content'][0]['text'])
        self.assertTrue(data['message']['truncated'])
        self.assertEqual(len(data['message']['content']), 5)

    def test_boundary_and_invalid_arguments_rejected(self):
        for arguments in ({'file_path': '../README.md'}, {'file_path': '/etc/hosts'},
                          {'file_path': '工具说明.txt', 'max_length': True}, {'file_path': '工具说明.txt', 'extra': 1}):
            self.assertTrue(server.read_file(arguments)['isError'])


if __name__ == '__main__':
    unittest.main()

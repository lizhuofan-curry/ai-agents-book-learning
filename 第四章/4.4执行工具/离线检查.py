import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('execution_entry', HERE/'我的代码/01_写入运行与结果核对.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExecutionChecks(unittest.TestCase):
    def test_real_execution_and_independent_goal_check(self):
        with tempfile.TemporaryDirectory() as temp:
            record = {}
            m.exercise(temp, record, verbose=False)
            self.assertTrue(all(record['checks'].values()))
            wrong_run = record['steps'][4]['result']
            wrong_goal = record['steps'][5]['result']
            self.assertTrue(wrong_run['success'])
            self.assertFalse(wrong_goal['goal_correct'])

    def test_invalid_syntax_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as temp:
            result = m.tools.file_write(temp, 'bad.py', m.tools.BROKEN)
            self.assertEqual(result['error_type'], 'SyntaxError')
            self.assertFalse((Path(temp)/'bad.py').exists())

    def test_boundary_overwrite_and_changed_script_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(m.tools.file_write(temp, '../escape.txt', 'test')['success'])
            self.assertTrue(m.tools.file_write(temp, 'wordcount.py', 'print(1)')['success'])
            self.assertFalse(m.tools.file_write(temp, 'wordcount.py', m.tools.GOOD)['success'])
            self.assertFalse(m.tools.run_fixed_python(temp, 'wordcount.py')['success'])


if __name__ == '__main__':
    unittest.main()

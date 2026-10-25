import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('long_output_step', HERE/'我的代码/08_长输出截断与完整保存.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LongOutputChecks(unittest.TestCase):
    def test_real_subprocess_output_preserved_and_middle_retrieved(self):
        with tempfile.TemporaryDirectory() as temp:
            record = {}
            m.exercise(temp, record, verbose=False)
            self.assertTrue(all(record['checks'].values()))
            self.assertEqual(len(Path(record['tool_result']['stdout_file']).read_text().splitlines()), 1000)

    def test_threshold_and_trailing_newline(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'log.txt'
            p.write_text('line\n'*200)
            self.assertFalse(m.summarize_log(p)['truncated'])
            p.write_text('line\n'*201)
            result = m.summarize_log(p)
            self.assertTrue(result['truncated'])
            self.assertEqual(result['original_line_count'], 201)
            self.assertEqual(result['omitted_line_count'], 191)

    def test_zero_tail_and_no_overlap_duplication(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'log.txt'
            p.write_text('one\ntwo\nthree\n')
            result = m.summarize_log(p, max_lines=1, head_lines=1, tail_lines=0)
            self.assertEqual(result['kept_source_lines'], 1)
            self.assertNotIn('three', result['stdout'])
            result = m.summarize_log(p, max_lines=1, head_lines=2, tail_lines=2)
            self.assertFalse(result['truncated'])
            self.assertEqual(result['stdout'], p.read_text())

    def test_bad_read_ranges_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'log.txt'
            p.write_text('line\n'*30)
            for start, end in [(0, 1), (3, 2), (1, 31), (1, 21)]:
                with self.assertRaises(ValueError):
                    m.read_log_lines(p, start, end)


if __name__ == '__main__':
    unittest.main()

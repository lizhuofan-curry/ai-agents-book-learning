"""显式测试输入与虚拟时钟；不冒充真实用户回复，不连接模型或网络。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('terminal_hitl', ROOT/'我的代码/01_终端人工确认与任务恢复.py')
hitl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hitl)


class FakeClock:
    now = 0
    def __call__(self):
        return self.now


class Checks(unittest.TestCase):
    def make(self):
        record = hitl.new_record(5)
        record['input_source'] = 'offline_test_fixture_not_human'
        return record

    def test_approve_resumes_and_writes_only_after_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.make()
            target = Path(directory)/hitl.TARGET_NAME
            def reader(remaining):
                self.assertFalse(target.exists())
                self.assertEqual(record['approval']['status'], 'pending')
                return '同意\n'
            hitl.wait_for_reply(record, lambda: None, reader, FakeClock())
            self.assertFalse(target.exists())
            hitl.finish_action(record, directory)
            self.assertTrue(record['action']['content_matches_expected'])
            self.assertIn(record['request_id'], target.read_text())

    def test_reject_stops_without_target(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.make()
            hitl.wait_for_reply(record, lambda: None, lambda _: 'REJECT\n', FakeClock())
            hitl.finish_action(record, directory)
            self.assertEqual(record['status'], 'completed_rejected')
            self.assertFalse(record['action']['executed'])
            self.assertFalse((Path(directory)/hitl.TARGET_NAME).exists())

    def test_timeout_is_not_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.make(); clock = FakeClock()
            def reader(remaining):
                clock.now += remaining
                return None
            hitl.wait_for_reply(record, lambda: None, reader, clock)
            hitl.finish_action(record, directory)
            self.assertEqual(record['approval']['status'], 'timeout')
            self.assertFalse(record['action']['executed'])

    def test_invalid_reply_does_not_reset_deadline_and_late_approve_ignored(self):
        record = self.make(); clock = FakeClock(); remaining_seen = []
        def reader(remaining):
            remaining_seen.append(remaining)
            clock.now += 2 if len(remaining_seen) == 1 else 4
            return '\n' if len(remaining_seen) == 1 else 'APPROVE\n'
        hitl.wait_for_reply(record, lambda: None, reader, clock)
        self.assertEqual(remaining_seen, [5, 3])
        self.assertEqual(record['approval']['status'], 'timeout')
        self.assertFalse(record['approval']['approved'])
        self.assertTrue(any(x['kind'] == 'late_input_ignored' for x in record['events']))

    def test_eof_not_approval(self):
        record = self.make()
        hitl.wait_for_reply(record, lambda: None, lambda _: '', FakeClock())
        self.assertEqual(record['approval']['status'], 'eof')
        self.assertFalse(record['approval']['approved'])

    def test_terminal_decision_and_action_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.make()
            hitl.wait_for_reply(record, lambda: None, lambda _: 'APPROVE\n', FakeClock())
            self.assertFalse(hitl.resolve(record, 'rejected'))
            with self.assertRaises(RuntimeError):
                hitl.wait_for_reply(record, lambda: None, lambda _: 'REJECT\n', FakeClock())
            hitl.finish_action(record, directory)
            with self.assertRaises(RuntimeError):
                hitl.finish_action(record, directory)

    def test_existing_target_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self.make()
            hitl.resolve(record, 'approved')
            path = Path(directory)/hitl.TARGET_NAME
            path.write_text('existing')
            with self.assertRaises(FileExistsError):
                hitl.finish_action(record, directory)
            self.assertEqual(path.read_text(), 'existing')


if __name__ == '__main__':
    unittest.main()

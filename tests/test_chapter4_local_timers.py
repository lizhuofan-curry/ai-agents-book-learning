"""单进程计时器状态及本地通知检查；使用虚拟时钟，不发送外部消息。"""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('local_timers',ROOT/'第四章/4.5协作工具/我的代码/18_本地通知与定时器.py')
step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(step)


class Clock:
    def __init__(self): self.now=0.0
    def __call__(self): return self.now
    def sleep(self,seconds): self.now+=seconds


class TimerChecks(unittest.TestCase):
    def setUp(self):
        self.clock=Clock()
        self.sent=[]
        self.events=[]
        self.timers=step.LocalTimers(lambda *args:self.sent.append(args),
                                    lambda kind,**details:self.events.append((kind,details)),self.clock)

    def test_due_boundary_and_no_second_delivery(self):
        self.timers.set_timer('a',2,'消息')
        self.clock.now=1.999
        self.timers.tick()
        self.assertEqual(self.sent,[])
        self.clock.now=2
        self.timers.tick()
        self.timers.tick()
        self.clock.now=10
        self.timers.tick()
        self.assertEqual(self.sent,[('a','消息')])
        self.assertEqual(self.timers.timers['a']['status'],'fired')

    def test_cancelled_timer_never_delivered_and_cancel_is_idempotent(self):
        self.timers.set_timer('a',2,'取消')
        self.assertTrue(self.timers.cancel('a'))
        self.assertFalse(self.timers.cancel('a'))
        self.assertFalse(self.timers.cancel('unknown'))
        self.clock.now=100
        self.timers.tick()
        self.assertEqual(self.sent,[])
        self.assertEqual(self.timers.timers['a']['status'],'cancelled')

    def test_recurring_delay_skips_missed_periods(self):
        self.timers.set_timer('r',1,'循环',recurring=True)
        self.clock.now=10
        self.timers.tick()
        self.timers.tick()
        self.assertEqual(len(self.sent),1)
        self.assertEqual(self.timers.timers['r']['due_monotonic'],11)
        self.clock.now=11
        self.timers.tick()
        self.assertEqual(len(self.sent),2)
        self.assertTrue(self.timers.cancel('r'))
        self.clock.now=30
        self.timers.tick()
        self.assertEqual(len(self.sent),2)

    def test_notification_failure_not_retried_or_marked_sent(self):
        def fail(*args): raise OSError('合成本地输出错误')
        self.timers.notify=fail
        self.timers.set_timer('a',1,'消息')
        self.clock.now=1
        with self.assertRaises(OSError): self.timers.tick()
        self.assertEqual(self.timers.timers['a']['status'],'failed')
        self.assertEqual(self.timers.timers['a']['fire_count'],0)
        self.timers.tick()

    def test_reentrant_tick_does_not_repeat_firing(self):
        def notify(*args):
            self.sent.append(args)
            self.timers.tick()
        self.timers.notify=notify
        self.timers.set_timer('a',1,'消息')
        self.clock.now=1
        self.timers.tick()
        self.assertEqual(len(self.sent),1)

    def test_bad_delay_and_duplicate_id_rejected(self):
        for delay in [0,-1,math.inf,math.nan,True,'2']:
            with self.assertRaises(ValueError): self.timers.set_timer('a',delay,'消息')
        self.timers.set_timer('a',1,'消息')
        with self.assertRaises(ValueError): self.timers.set_timer('a',1,'覆盖')

    def test_full_demo_local_log_and_counts(self):
        record={'events':[],'notifications':[]}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(step.exercise(record,Path(folder),lambda:None,self.clock,self.clock.sleep))
            self.assertEqual(record['notification_counts'],{'once_demo':1,'cancelled_demo':0,'recurring_demo':3})
            self.assertEqual(len((Path(folder)/'notifications.jsonl').read_text().splitlines()),4)
            self.assertTrue(all(record['checks'].values()))

    def test_preview_no_files_or_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(step,'ROOT',Path(folder)), patch.object(step,'exercise') as run:
                with patch('sys.argv',['timers','--preview']), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(step.main(),0)
                run.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()),[])


if __name__=='__main__': unittest.main()

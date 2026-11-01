"""本地一次提醒、取消和有限循环提醒；只写本次目录和终端，无模型或外部渠道。"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CHINA = timezone(timedelta(hours=8))


class LocalTimers:
    """单进程轮询计时器；monotonic计时，循环采用固定延迟，不补发错过的周期。"""
    def __init__(self, notify, event, clock=time.monotonic):
        self.clock, self.notify, self.event = clock, notify, event
        self.timers = {}

    def set_timer(self, timer_id, delay, message, recurring=False):
        if timer_id in self.timers:
            raise ValueError('计时器编号重复')
        if not isinstance(delay, (int,float)) or isinstance(delay,bool) or not math.isfinite(delay) or delay <= 0:
            raise ValueError('延迟必须为有限正数')
        item = {'timer_id': timer_id, 'message': message, 'interval_seconds': delay,
                'recurring': recurring, 'status': 'pending', 'due_monotonic': self.clock()+delay,
                'fire_count': 0}
        self.timers[timer_id] = item
        self.event('timer_set', timer_id=timer_id, delay_seconds=delay, recurring=recurring)

    def cancel(self, timer_id):
        item = self.timers.get(timer_id)
        if item is None or item['status'] != 'pending':
            self.event('cancel_ignored', timer_id=timer_id)
            return False
        item['status'] = 'cancelled'
        self.event('timer_cancelled', timer_id=timer_id, fire_count=item['fire_count'])
        return True

    def tick(self):
        now = self.clock()
        for item in self.timers.values():
            if item['status'] != 'pending' or now < item['due_monotonic']:
                continue
            self.event('timer_due', timer_id=item['timer_id'])
            # 先进入firing，防止回调中再次tick造成重复触发。
            item['status'] = 'firing'
            try:
                self.notify(item['timer_id'], item['message'])
            except BaseException as exc:
                item['status'] = 'failed'
                self.event('notification_failed', timer_id=item['timer_id'], error_type=type(exc).__name__, error=str(exc))
                raise
            item['fire_count'] += 1
            if item['recurring']:
                item.update(status='pending', due_monotonic=self.clock()+item['interval_seconds'])
            else:
                item['status'] = 'fired'


def notification_sink(record, directory, save):
    """本地日志和终端是两个可核对的输出；终端写成功不代表人已阅读。"""
    def notify(timer_id, message):
        item = {'notification_id': str(uuid.uuid4()), 'timer_id': timer_id, 'message': message,
                'at': datetime.now(CHINA).isoformat(), 'local_log_written': False, 'terminal_written': False}
        record['notifications'].append(item)
        save()
        # 日志是通知内容，不是所有通道状态的副本。
        content = {k:item[k] for k in ('notification_id','timer_id','message','at')}
        with (directory/'notifications.jsonl').open('a',encoding='utf-8') as output:
            output.write(json.dumps(content,ensure_ascii=False)+'\n')
        item['local_log_written'] = True
        save()
        print('本地通知：'+timer_id+' → '+message,flush=True)
        item['terminal_written'] = True
        save()
    return notify


def exercise(record, directory, save, clock=time.monotonic, sleep=time.sleep):
    start = clock()
    def event(kind, **details):
        record['events'].append({'kind':kind,'elapsed_seconds':clock()-start,
                                 'at':datetime.now(CHINA).isoformat(),**details})
        save()
    timers = LocalTimers(notification_sink(record,directory,save),event,clock)
    # record引用实时状态，异常退出时仍能看到pending/fired/cancelled/failed的区别。
    record['timers'] = timers.timers
    timers.set_timer('once_demo',2.0,'一次性提醒到期。')
    timers.set_timer('cancelled_demo',4.0,'这条提醒应被提前取消，不应出现。')
    timers.set_timer('recurring_demo',1.0,'循环提醒的一次到期。',recurring=True)
    timers.cancel('cancelled_demo')
    print('已设置：一次提醒2秒；循环间隔1秒、最多观察3次；4秒提醒已提前取消。',flush=True)
    print('观察约4.5秒，到原定4秒之后检查被取消项没有触发。',flush=True)
    while clock()-start < 4.5:
        timers.tick()
        recurring = timers.timers['recurring_demo']
        if recurring['status']=='pending' and recurring['fire_count'] >= 3:
            timers.cancel('recurring_demo')
        sleep(0.02)
    record['elapsed_seconds'] = clock()-start
    counts = {name:sum(n['timer_id']==name for n in record['notifications']) for name in timers.timers}
    logs = [json.loads(line) for line in (directory/'notifications.jsonl').read_text(encoding='utf-8').splitlines()]
    saved_contents = [{k:n[k] for k in ('notification_id','timer_id','message','at')} for n in record['notifications']]
    record['checks'] = {
        'once_exactly_one': counts['once_demo']==1,
        'cancelled_timer_zero': counts['cancelled_demo']==0,
        'recurring_exactly_three': counts['recurring_demo']==3,
        'once_terminal_state_fired': timers.timers['once_demo']['status']=='fired',
        'cancelled_terminal_state': timers.timers['cancelled_demo']['status']=='cancelled',
        'recurring_cancelled_after_three': timers.timers['recurring_demo']['status']=='cancelled',
        'all_local_logs_and_terminal_writes_completed': all(n['local_log_written'] and n['terminal_written'] for n in record['notifications']),
        'log_contents_match_record': logs==saved_contents,
        'unique_notifications': len({n['notification_id'] for n in record['notifications']})==len(record['notifications']),
    }
    record['notification_counts'] = counts
    record['status'] = 'local_timer_demo_checked' if all(record['checks'].values()) else 'finished_with_failed_checks'
    save()
    return all(record['checks'].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    args = parser.parse_args()
    print('4-5本地通知与定时器：一次到期、提前取消、循环三次后停止。')
    print('0次模型、MCP或网络请求；通知仅写本次目录和终端。')
    if args.preview:
        print('LOCAL_TIMER_PREVIEW_ONLY：不等待、不创建目录或启动计时器。')
        return 0
    directory = ROOT/'运行结果'/('local_timers_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True,exist_ok=False)
    path = directory/'record.json'
    record = {'status':'running','created_at':datetime.now(CHINA).isoformat(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'official_readme_sha256':hashlib.sha256((ROOT/'官方资料/README.md').read_bytes()).hexdigest(),
              'events':[],'notifications':[],'model_requests':0,'network_requests':0,
              'external_messages_sent':0,'mcp_calls':0,'automatic_retries':0,
              'scope':'single process local teaching timers and notification log; no external delivery or restart recovery; human reading not verified'}
    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        tmp.replace(path)
    save()
    print('记录路径：'+str(path),flush=True)
    try:
        passed = exercise(record,directory,save)
        print('通知数量：'+json.dumps(record['notification_counts'],ensure_ascii=False))
        print('核对：'+json.dumps(record['checks'],ensure_ascii=False))
        print('LOCAL_TIMER_RUN_SAVED：状态与本地输出已保存。')
        return 0 if passed else 1
    except (Exception,KeyboardInterrupt) as exc:
        record.update(status='incomplete',error={'type':type(exc).__name__,'message':str(exc)})
        print('本步未完成，不重试：'+str(exc))
        return 1
    finally:
        save()
        print('记录路径：'+str(path),flush=True)


if __name__=='__main__':
    raise SystemExit(main())

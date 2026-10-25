"""三个本地MCP工具：创建请求、提交决定、根据服务端决定执行。"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


class ApprovalStore:
    def __init__(self, directory, clock=time.monotonic):
        self.directory = Path(directory).resolve(strict=True)
        self.clock = clock
        self.request = None
        self.deadline = None
        self.events = []

    def save(self, kind):
        self.events.append({'kind': kind, 'at': datetime.now(timezone(timedelta(hours=8))).isoformat()})
        raw = json.dumps({'request': self.request, 'events': self.events}, ensure_ascii=False, indent=2)+'\n'
        path = self.directory/'server_state.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(raw, encoding='utf-8')
        tmp.replace(path)

    def expire(self):
        if self.request and self.request['status'] == 'pending' and self.clock() >= self.deadline:
            self.request['status'] = 'timeout'
            self.save('server_timeout')

    def snapshot(self):
        return {'success': True, **self.request,
                'remaining_seconds': max(0, self.deadline-self.clock()) if self.request['status'] == 'pending' else 0}

    def create_request(self, timeout_seconds=60):
        if self.request is not None:
            return {'success': False, 'error': '本次服务只创建一个请求'}
        if not 1 <= timeout_seconds <= 3600:
            return {'success': False, 'error': '等待时间不符'}
        self.deadline = self.clock()+timeout_seconds
        self.request = {'request_id': str(uuid.uuid4()), 'status': 'pending',
                        'message': '是否在本次运行目录生成一份固定的MCP教学结果文件？',
                        'target_filename': 'MCP教学结果.txt', 'timeout_seconds': timeout_seconds,
                        'finished': False}
        self.save('request_created_on_server')
        return self.snapshot()

    def submit_decision(self, request_id, decision):
        if self.request is None or request_id != self.request['request_id']:
            return {'success': False, 'error': '请求编号不匹配'}
        self.expire()
        if self.request['status'] != 'pending':
            return {'success': False, 'error': '已结束，拒绝迟到或重复决定', 'current_status': self.request['status']}
        if decision not in {'approved', 'rejected', 'timeout', 'eof', 'input_unavailable', 'cancelled'}:
            return {'success': False, 'error': '未知决定'}
        # 客户端只能报告已结束等待；服务端不接受提前宣称超时。
        if decision == 'timeout':
            return {'success': False, 'error': '服务端尚未到期', 'current_status': 'pending'}
        self.request['status'] = decision
        self.save('client_decision_received_over_mcp')
        return self.snapshot()

    def finish_request(self, request_id):
        if self.request is None or request_id != self.request['request_id']:
            return {'success': False, 'error': '请求编号不匹配'}
        self.expire()
        if self.request['finished']:
            return {'success': False, 'error': '动作已结束，不能重复执行'}
        status = self.request['status']
        if status == 'pending':
            return {'success': False, 'error': '尚未取得决定，不执行'}
        target = self.directory/self.request['target_filename']
        if status == 'approved':
            content = 'MCP人机协同教学结果\n请求编号：'+request_id+'\n服务端收到本次明确批准后生成此文件。\n'
            with target.open('x', encoding='utf-8') as file:
                file.write(content)
            if target.read_text(encoding='utf-8') != content:
                raise ValueError('产物核对失败')
            action = {'executed': True, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        else:
            if target.exists():
                raise ValueError('未批准却存在目标文件')
            action = {'executed': False, 'target_absent': True}
        self.request.update(finished=True, action=action)
        self.save('task_finished_on_server')
        return self.snapshot()


def main():
    from mcp.server import MCPServer
    from mcp.types import CallToolResult
    store = ApprovalStore(os.environ['HITL_RUN_DIRECTORY'])
    server = MCPServer('learning-human-approval', version='0.1.0', log_level='ERROR')

    def result(data):
        return CallToolResult.model_validate({'content': [{'type': 'text', 'text': json.dumps(data, ensure_ascii=False)}],
                                              'isError': not data['success']})

    @server.tool()
    def create_request(timeout_seconds: float = 60) -> CallToolResult:
        """创建本次固定本地写入请求；返回编号及待处理状态，不执行写入。"""
        return result(store.create_request(timeout_seconds))

    @server.tool()
    def submit_decision(request_id: str, decision: str) -> CallToolResult:
        """提交客户端实际收到的人工决定，编号须匹配，终态不可改。"""
        return result(store.submit_decision(request_id, decision))

    @server.tool()
    def finish_request(request_id: str) -> CallToolResult:
        """依据服务端保存的决定继续或停止，不接收客户端自称的批准字段。"""
        return result(store.finish_request(request_id))

    server.run(transport='stdio')


if __name__ == '__main__':
    main()

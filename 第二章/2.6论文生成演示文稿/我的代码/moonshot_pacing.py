"""仅在用户启动Agent时运行：本机串行转发Moonshot API，降低请求频率。"""

import hmac
import json
import secrets
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PacedMoonshot:
    def __init__(self, base_url, api_key, log_path, interval=25, retry_floor=65):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token = secrets.token_urlsafe(32)
        self.log_path = log_path
        self.interval = interval
        self.retry_floor = retry_floor
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.last_start = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            # HTTP/1.0关闭连接表示流式响应结束，避免伪造Content-Length。
            def log_message(self, *args):
                pass  # 不记录URL、请求头或请求正文。

            def do_POST(self):
                owner.forward(self)

            def do_GET(self):
                owner.forward(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def local_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def start(self):
        self.thread.start()

    def close(self):
        self.stopped.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def note(self, **fields):
        record = {"time": datetime.now().isoformat(timespec="seconds"), **fields}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def wait(self, seconds, reason):
        if seconds > 0:
            self.note(wait_seconds=round(seconds, 3), reason=reason)
            print(f"[请求节流] {reason}，等待 {seconds:.1f} 秒。", flush=True)
            if self.stopped.wait(seconds):
                raise InterruptedError("Agent已结束")

    def forward(self, client):
        if not hmac.compare_digest(client.headers.get("Authorization", ""), "Bearer " + self.token):
            client.send_error(401)
            return
        allowed = {("POST", "/v1/chat/completions"), ("GET", "/v1/models")}
        if (client.command, client.path) not in allowed:
            client.send_error(404)
            return
        try:
            length = int(client.headers.get("Content-Length", "0"))
        except ValueError:
            client.send_error(400)
            return
        if not 0 <= length <= 64 * 1024 * 1024:
            client.send_error(413)
            return
        body = client.rfile.read(length) if client.command == "POST" else None
        headers = {"Authorization": "Bearer " + self.api_key,
                   "Content-Type": "application/json", "Accept-Encoding": "identity"}
        # 不信任本机请求提供的目标/Authorization，只向已核验的官方端点转发。
        target = self.base_url + client.path[len("/v1"):]
        sent_headers = False
        try:
            with self.lock:
                for attempt in range(4):
                    if self.stopped.is_set():
                        return
                    wait = 0 if self.last_start is None else self.interval - (time.monotonic() - self.last_start)
                    self.wait(max(0, wait), "保持API请求间隔")
                    self.last_start = time.monotonic()
                    request = Request(target, data=body, headers=headers, method=client.command)
                    try:
                        response = urlopen(request, timeout=600)
                    except HTTPError as error:
                        response = error
                    status = response.code
                    self.note(method=client.command, status=status, attempt=attempt + 1)
                    if status == 429 and attempt < 3:
                        try:
                            advised = float(response.headers.get("Retry-After", "0"))
                        except ValueError:
                            advised = 0
                        response.close()
                        self.wait(max(self.retry_floor * (2 ** attempt), advised), "服务端429退避")
                        continue
                    with response:
                        client.send_response(status)
                        client.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        client.send_header("Connection", "close")
                        client.end_headers()
                        sent_headers = True
                        while chunk := response.read1(65536):
                            client.wfile.write(chunk)
                            client.wfile.flush()
                    break
        except (BrokenPipeError, ConnectionResetError, InterruptedError):
            self.note(reason="客户端连接关闭或任务停止")
        except (OSError, URLError) as error:
            self.note(error_type=type(error).__name__)
            if not sent_headers:
                client.send_error(502, "Upstream connection failed; see pacing log")

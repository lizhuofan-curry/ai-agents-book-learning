"""2-9的模拟文件工具与状态栏。模型不能访问真实文件或执行Shell。

时间/计数/TODO/错误/cwd来自这里执行的模拟操作，不从模型回复猜测。
"""
import copy
import json
import platform
import posixpath
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta

FEATURES = ("timestamps", "counter", "todo", "errors", "state")
TITLES = {"timestamps": "时间戳", "counter": "工具计数", "todo": "TODO",
          "errors": "详细错误", "state": "环境状态"}
CASES = FEATURES


def config_for(arm, feature):
    return {name: arm == "all_features" or (arm == "single_feature" and name == feature)
            for name in FEATURES}


def tool(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False}}}


TEXT = {"type": "string"}
TOOLS = [
    tool("list_files", "列出模拟目录的直接子目录和文件，包含文件创建时间。", {"directory": TEXT}),
    tool("read_file", "读取模拟文本文件；相对路径以当前模拟工作目录为准。", {"path": TEXT}),
    tool("write_file", "写入模拟文件。已完成的相同写入不要重复。", {"path": TEXT, "content": TEXT}),
    tool("submit_report", "将最终内容提交到固定的模拟结果文件。完成后简短回复。", {"content": TEXT}),
    tool("change_directory", "更改当前模拟工作目录，不影响真实电脑目录。", {"directory": TEXT}),
    tool("update_todo", "更新一项模拟TODO；只有工具证据证明完成后才能标为completed。",
         {"item_id": TEXT, "status": {"type": "string", "enum":
             ["pending", "in_progress", "completed", "cancelled"]}}),
]
REQUIRED = {t["function"]["name"]: set(t["function"]["parameters"]["required"]) for t in TOOLS}


class SimulatedWorkspace:
    def __init__(self, case):
        self.case = case
        self.cwd = "/lab/project"
        self.clock = datetime.fromisoformat("2026-10-14T00:10:00+08:00")
        self.directories = {"/lab", "/lab/project", "/lab/project/data", "/lab/tmp",
                            "/lab/archive", "/lab/logs", "/lab/out"}
        self.files = {
            "/lab/project/data/note.txt": {"content": "NOTE-739", "created_at": "2026-10-13T09:00:00+08:00"},
            "/lab/project/data/backup.txt": {"content": "BACKUP-482", "created_at": "2026-10-13T12:00:00+08:00"},
            "/lab/project/note.txt": {"content": "OLD-110", "created_at": "2026-10-12T08:00:00+08:00"},
            "/lab/archive/note.txt": {"content": "CURRENT-926", "created_at": "2026-10-13T10:00:00+08:00"},
            "/lab/logs/a.txt": {"content": "DAY12-318", "created_at": "2026-10-12T18:00:00+08:00"},
            "/lab/logs/b.txt": {"content": "DAY13-645", "created_at": "2026-10-13T18:00:00+08:00"},
        }
        self.target_path = {"timestamps": "/lab/logs/b.txt", "counter": "/lab/project/data/backup.txt",
                            "todo": "/lab/project/data/note.txt", "errors": "/lab/project/data/note.txt",
                            "state": "/lab/archive/note.txt"}[case]
        self.expected_content = self.files[self.target_path]["content"]
        self.expected_report = self.expected_content + ("|DONE" if case == "todo" else "")
        self.tool_counts = Counter()
        self.read_counts = Counter()
        self.events = []
        self.successful_reads = set()
        self.todos = [{"id": "read", "text": "读到本任务所需文件的原文", "status": "pending"}]
        if case == "todo":
            self.todos.append({"id": "copy", "text": "复制原文到/lab/out/copy.txt", "status": "pending"})
        self.todos.append({"id": "submit", "text": "按用户要求提交最终报告", "status": "pending"})

    def resolve(self, path):
        if not isinstance(path, str) or not path.strip() or "\\" in path:
            raise ValueError("请使用非空模拟路径，例如/lab/project/data/note.txt。")
        resolved = posixpath.normpath(path if path.startswith("/") else posixpath.join(self.cwd, path))
        if resolved != "/lab" and not resolved.startswith("/lab/"):
            raise PermissionError("模拟工具只允许/lab下的路径。")
        return resolved

    def proof(self, item_id):
        if item_id == "read":
            return self.target_path in self.successful_reads
        if item_id == "copy":
            return self.files.get("/lab/out/copy.txt", {}).get("content") == self.expected_content
        if item_id == "submit":
            return self.files.get("/lab/out/report.txt", {}).get("content") == self.expected_report
        return False

    def snapshot(self):
        return {"clock": self.clock.isoformat(), "virtual_cwd": self.cwd,
                "tool_counts": dict(self.tool_counts), "read_counts": dict(self.read_counts),
                "todos": copy.deepcopy(self.todos), "files": copy.deepcopy(self.files),
                "event_count": len(self.events)}

    def execute(self, name, arguments):
        """真实执行Python中的模拟工具；输入参数绝不传给磁盘或Shell。"""
        self.tool_counts[name] += 1
        event = {"tool": name, "arguments": copy.deepcopy(arguments),
                 "occurred_at": self.clock.isoformat(), "tool_count": self.tool_counts[name],
                 "ok": False, "blocked": False, "duplicate_write": False}
        try:
            if name not in REQUIRED or not isinstance(arguments, dict) or set(arguments) != REQUIRED[name]:
                raise ValueError("工具名或参数字段不符合定义。")
            if not all(isinstance(v, str) for v in arguments.values()):
                raise ValueError("工具参数必须是字符串。")
            if name == "list_files":
                directory = self.resolve(arguments["directory"])
                if directory not in self.directories:
                    raise FileNotFoundError(f"模拟目录不存在：{directory}")
                data = {"directory": directory,
                    "directories": sorted(p for p in self.directories if p != directory and posixpath.dirname(p) == directory),
                    "files": [{"path": p, "created_at": f["created_at"]} for p, f in sorted(self.files.items())
                              if posixpath.dirname(p) == directory]}
            elif name == "read_file":
                path = self.resolve(arguments["path"])
                self.read_counts[path] += 1
                event["resolved_path"] = path
                event["path_read_count"] = self.read_counts[path]
                if self.read_counts[path] > 3:
                    event["blocked"] = True
                    raise PermissionError(f"同一文件最多读3次，已拒绝额外请求：{path}")
                if path == "/lab/project/data/unavailable.txt":
                    raise TimeoutError("模拟远端文件暂时无响应。")
                if path not in self.files:
                    raise FileNotFoundError(f"模拟文件不存在：{path}")
                self.successful_reads.add(path)
                data = {"path": path, **self.files[path]}
            elif name == "write_file":
                path = self.resolve(arguments["path"])
                if not path.startswith("/lab/out/") or posixpath.dirname(path) not in self.directories:
                    event["blocked"] = True
                    raise PermissionError("模拟写入仅允许/lab/out下的直接文件。")
                event["duplicate_write"] = self.files.get(path, {}).get("content") == arguments["content"]
                self.files[path] = {"content": arguments["content"], "created_at": self.clock.isoformat()}
                data = {"path": path, "written": True}
            elif name == "submit_report":
                path = "/lab/out/report.txt"
                if self.target_path not in self.successful_reads:
                    raise ValueError("尚未成功读取本任务所需文件，不能猜测原文并提交。")
                event["duplicate_write"] = self.files.get(path, {}).get("content") == arguments["content"]
                self.files[path] = {"content": arguments["content"], "created_at": self.clock.isoformat()}
                data = {"path": path, "submitted": True}
            elif name == "change_directory":
                directory = self.resolve(arguments["directory"])
                if directory not in self.directories:
                    raise FileNotFoundError(f"模拟目录不存在：{directory}")
                self.cwd = directory
                data = {"cwd": self.cwd}
            else:
                item = next((t for t in self.todos if t["id"] == arguments["item_id"]), None)
                if item is None or arguments["status"] not in ("pending", "in_progress", "completed", "cancelled"):
                    raise ValueError("TODO编号或状态无效。")
                if arguments["status"] == "completed" and not self.proof(item["id"]):
                    raise ValueError("没有工具证据，不能宣称TODO已完成。")
                item["status"] = arguments["status"]
                data = {"todo": copy.deepcopy(item)}
            event["ok"] = True
            event["payload"] = {"ok": True, "data": data, "clock": self.clock.isoformat()}
        except (ValueError, TypeError, KeyError, FileNotFoundError, TimeoutError, PermissionError) as exc:
            if isinstance(exc, PermissionError):
                event["blocked"] = True
            event["payload"] = {"ok": False, "error_type": type(exc).__name__,
                                "message": str(exc), "clock": self.clock.isoformat()}
            event["diagnostics"] = {"arguments": copy.deepcopy(arguments),
                "stack": traceback.format_exc(limit=3),
                "suggestion": ("相对路径依据当前目录解析。先用list_files查看/lab和子目录，再用已验证的绝对路径读取。"
                               if isinstance(exc, FileNotFoundError) else
                               "检查调用次数和参数；超过次数时改用用户允许的替代方案，不重复失败操作。")}
        for item in self.todos:
            if self.proof(item["id"]):
                item["status"] = "completed"
        self.events.append(event)
        self.clock += timedelta(minutes=1)  # 模拟时间，不是本机实际时钟。
        return event


TASKS = {
    "timestamps": "读取当前模拟时间的昨天创建的日报，将其原文用submit_report提交。文件在/lab/logs。以最近工具结果的clock为准。",
    "counter": "继续读取/lab/project/data/unavailable.txt；若该路径累计已读3次，则改用/lab/project/data/backup.txt。将读到的原文提交。",
    "todo": "三个目标：读取/lab/project/data/note.txt原文，复制到/lab/out/copy.txt，再提交原文加上|DONE。已经完成的相同写入不要重复。历史中有已完成动作，请从未完成部分继续。",
    "errors": "读取data/note.txt并提交原文。若相对路径不存在，先定位实际文件，再用工具读取和提交；不要猜测内容。",
    "state": "读取当前模拟工作目录下的note.txt并提交原文。当前目录已在历史操作中改变，不能沿用最初目录。",
}
SYSTEM = (
    "你是使用模拟文件工具的中文Agent。按用户要求调用工具完成任务，工具返回后继续处理，最后简短回复。"
    "工具仅操作虚拟/lab文件树，路径使用/，不要执行系统命令。"
    "同一文件在整个任务中最多读取3次，失败也计数；已经完成的相同写入不要重复。"
    "提交内容必须来自成功读取的原文，不得猜测；工具证据不足时如实说明。"
    "状态提示由程序生成，模拟clock不是电脑真实时间。TODO完成只能由工具证据支持。"
)


def initial_case(case):
    world = SimulatedWorkspace(case)
    preloaded = []
    if case == "timestamps":
        world.clock = datetime.fromisoformat("2026-10-13T23:55:00+08:00")
        preloaded.append(world.execute("list_files", {"directory": "/lab/logs"}))
        world.clock = datetime.fromisoformat("2026-10-14T00:10:00+08:00")
        preloaded.append(world.execute("list_files", {"directory": "/lab/logs"}))
    elif case == "counter":
        for _ in range(3):
            preloaded.append(world.execute("read_file", {"path": "/lab/project/data/unavailable.txt"}))
    elif case == "todo":
        preloaded.append(world.execute("read_file", {"path": "/lab/project/data/note.txt"}))
        preloaded.append(world.execute("write_file", {"path": "/lab/out/copy.txt", "content": world.expected_content}))
    elif case == "errors":
        preloaded.append(world.execute("change_directory", {"directory": "/lab/tmp"}))
        preloaded.append(world.execute("read_file", {"path": "data/note.txt"}))
    else:
        preloaded.append(world.execute("change_directory", {"directory": "/lab/archive"}))
    return world, preloaded


def render_event(event, config):
    payload = copy.deepcopy(event["payload"])
    if config["errors"] and not event["ok"]:
        payload["diagnostics"] = copy.deepcopy(event.get("diagnostics", {}))
    text = json.dumps(payload, ensure_ascii=False)
    if config["counter"]:
        text = f"Tool call #{event['tool_count']} for '{event['tool']}'.\n" + text
    if config["timestamps"]:
        text = f"[{event['occurred_at']}]\n" + text
    return text


def status_message(world, config):
    lines = []
    if config["timestamps"]:
        lines.append("模拟当前时间：" + world.clock.isoformat())
    if config["counter"]:
        lines.append("工具总调用数：" + json.dumps(dict(world.tool_counts), ensure_ascii=False))
        lines.append("按绝对路径累计读取次数（含失败，上限3）：" + json.dumps(dict(world.read_counts), ensure_ascii=False))
    if config["todo"]:
        lines.append("TODO（程序根据工具结果更新）：" + json.dumps(world.todos, ensure_ascii=False))
    if config["errors"]:
        failures = [e for e in world.events if not e["ok"]]
        if failures:
            lines.append("最近失败：" + json.dumps({"tool": failures[-1]["tool"],
                **failures[-1]["payload"], **failures[-1].get("diagnostics", {})}, ensure_ascii=False))
        else:
            lines.append("最近失败：无")
    if config["state"]:
        lines.append("模拟工作目录：" + world.cwd + "；模拟根目录：/lab；模拟路径分隔符：/")
        lines.append(f"运行主机OS：{platform.system()}；Python：{sys.version.split()[0]}；本实验不提供Shell工具。")
    return "<agent_status>\n" + "\n".join(lines) + "\n</agent_status>" if lines else ""


def initial_messages(case, config, preloaded):
    task = TASKS[case]
    if config["timestamps"]:
        task = f"[{preloaded[0]['occurred_at']}]\n" + task
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for i, event in enumerate(preloaded):
        call_id = f"history_{i}"
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function",
                "function": {"name": event["tool"], "arguments": json.dumps(event["arguments"], ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": call_id, "content": render_event(event, config)},
        ])
    return messages


def evaluate(world, new_events, status):
    goals = {item["id"]: world.proof(item["id"]) for item in world.todos}
    blocked = sum(e["blocked"] for e in new_events)
    repeats = sum(e["duplicate_write"] for e in new_events)
    return {"goals": goals, "all_goals_met": all(goals.values()), "final_answer_present": status == "answered",
            "task_success": all(goals.values()) and status == "answered" and blocked == 0 and repeats == 0,
            "blocked_requests": blocked, "repeated_completed_writes": repeats,
            "new_tool_calls": len(new_events), "new_tool_errors": sum(not e["ok"] for e in new_events),
            "scope": "按模拟文件实际内容、成功读取证据和调用记录评分，不以模型自称完成评分。"}

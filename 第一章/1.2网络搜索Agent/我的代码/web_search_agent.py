"""实验1.2：把已学习的模型与工具循环整理为WebSearchAgent类。

方法名和调用关系对应官方WebSearchAgent，网络步骤复用03至06的学习实现。
学习改动：显式传入根目录配置；失败抛出异常；记录状态和完整消息；使用K3参数。
来源：https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent
"""

import json
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
from openai import OpenAI


class WebSearchAgent:
    """保存客户端、工具定义和对话历史，完成一个问题的搜索与回答。"""

    def __init__(self, api_key, base_url="https://api.moonshot.cn/v1", timeout=180, verbose=True):
        # self指当前这个Agent对象；self.xxx让不同方法能够使用同一份数据。
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY不能为空。")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.model = "kimi-k3"
        self.formula_uri = "moonshot/web-search:latest"
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.tools = None
        # 保存给kimi看的对话内容
        self.conversation_history = []
        self.trace = []
        # api_turns 保存给我们检查的接口调用记录
        self.api_turns = []
        self.question = ""
        self.model_turns = 0
        self.finish_reason = None
        self.status = "not_started"
        self.answer_complete = False
        self.answer = ""
        # 创建对象到这里仅准备配置与容器，还没有发送网络请求。

    def _get_system_prompt(self):
        """对应03中的system消息内容。"""
        return "你是Kimi搜索助手。请使用web_search查询官方资料，再依据搜索结果回答并提供来源链接。"

    def _get_tools(self):
        """对应03的GET /tools；同一个问题内复用已经获取的定义。"""
        if self.tools is not None:
            return self.tools
        url = f"{self.base_url}/formulas/{self.formula_uri}/tools"
        if self.verbose:
            print("获取Moonshot官方搜索工具定义……", flush=True)
        # 只对获取定义的GET处理429；总共最多3次，累计等待不超过60秒。
        # max_iterations仍只限制模型轮数，不把重试算作新的模型回答。
        waited = 0.0
        for attempt in range(1, 4):
            response = requests.get(
                url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout,
            )
            if response.status_code != 429:
                if not response.ok:
                    self.api_turns.append(self._http_error_record(response, attempt))
                response.raise_for_status()
                break
            detail = self._http_error_record(response, attempt)
            self.api_turns.append(detail)
            error_text = f"{detail['error_type']} {detail['error_message']}".lower()
            # 余额/配额或日额度问题不通过短间隔连续重试解决。
            persistent = any(word in error_text for word in (
                "exceeded_current_quota", "insufficient_quota", "insufficient_balance",
                "insufficient balance", "tpd", "daily", "per day", "余额不足", "日额度",
            ))
            delay = self._retry_after_seconds(detail["retry_after"], fallback=20 * 2 ** (attempt - 1))
            can_retry = not persistent and attempt < 3 and waited + delay <= 60
            detail["retry_scheduled"] = can_retry
            detail["retry_wait_seconds"] = delay if can_retry else None
            if not can_retry:
                detail["stop_reason"] = ("quota_or_daily_limit" if persistent else
                    "attempt_limit" if attempt == 3 else "wait_budget_exceeded")
                raise RuntimeError(
                    "获取搜索工具定义被限流（HTTP 429）；尚未执行搜索。"
                    f" error.type={detail['error_type'] or '未知'}；"
                    f"Retry-After={detail['retry_after'] or '未提供'}；"
                    f"服务端说明：{detail['error_message'] or '未提供'}。"
                    "已保存响应诊断；稍后再试或按服务端说明检查额度。"
                )
            # 即使quiet=True，仍显示等待，避免用户误以为程序卡住。
            print(f"工具定义接口返回429；等待{delay:g}秒后进行第{attempt + 1}/3次尝试。", flush=True)
            response.close()
            time.sleep(delay)
            waited += delay
        tools = response.json().get("tools")
        if not isinstance(tools, list) or not tools:
            raise RuntimeError("官方接口没有返回有效工具定义。")
        if not any(tool.get("type") == "function" and tool.get("function", {}).get("name") == "web_search" for tool in tools):
            raise RuntimeError("官方工具定义中没有web_search。")
        self.tools = tools
        self.api_turns.append({"kind": "formula_tools", "http_status": response.status_code, "tools": tools})
        return self.tools

    def _http_error_record(self, response, attempt):
        """只记录服务端错误及诊断头，不保存请求头中的密钥。"""
        try:
            body = response.json()
            error = body.get("error", {}) if isinstance(body, dict) else {}
        except ValueError:
            error = {}
        if not isinstance(error, dict):
            error = {}
        message = str(error.get("message") or response.text or "")
        message = message.replace(self.api_key, "[REDACTED]")[:2000]
        error_type = str(error.get("type") or error.get("code") or "")
        return {"kind": "formula_tools_error", "attempt": attempt,
                "http_status": response.status_code, "error_type": error_type,
                "error_message": message, "retry_after": response.headers.get("Retry-After"),
                "request_id": response.headers.get("X-Request-Id"),
                "recorded_at": datetime.now(timezone.utc).isoformat()}

    @staticmethod
    def _retry_after_seconds(value, fallback):
        """Retry-After支持秒数和HTTP日期；较长等待交给用户，不提前重试。"""
        if value:
            try:
                seconds = float(value)
                if seconds >= 0 and seconds < float("inf"):
                    return max(1.0, seconds)
            except (ValueError, TypeError):
                try:
                    when = parsedate_to_datetime(value)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    return max(1.0, (when - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    pass
        return float(fallback)

    def _chat(self):
        """对应03、05、06的模型请求；返回本轮choice。"""
        tools = self._get_tools()
        response = self.client.chat.completions.create(
            model=self.model, messages=self.conversation_history, tools=tools, tool_choice="auto",
            temperature=1, max_tokens=32768, reasoning_effort="max",
        )
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage is not None else None
        self.api_turns.append({
            "kind": "chat_completion", "model": response.model,
            "sent_messages": len(self.conversation_history), "finish_reason": choice.finish_reason,
            "usage": usage,
        })
        return choice

    def _execute_formula(self, name, raw_arguments):
        """对应04和06的POST /fibers；输入工具名和参数，返回结果字符串。"""
        body = {"name": name, "arguments": raw_arguments}
        url = f"{self.base_url}/formulas/{self.formula_uri}/fibers"
        response = requests.post(
            url, headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=self.timeout,
        )
        response.raise_for_status()
        fiber = response.json()
        if fiber.get("status") != "succeeded":
            raise RuntimeError(f"搜索执行失败，状态：{fiber.get('status')}")
        context = fiber.get("context") or {}
        output_field = "output"
        content = context.get(output_field)
        if content in (None, ""):
            output_field = "encrypted_output"
            content = context.get(output_field)
        if content in (None, ""):
            raise RuntimeError("搜索状态为succeeded，但返回内容为空。")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        self.api_turns.append({
            "kind": "formula_fiber", "model_turn": self.model_turns, "request": body,
            "http_status": response.status_code, "status": fiber["status"],
            "fiber_id": fiber.get("id"), "output_field": output_field, "output_length": len(content),
        })
        if self.verbose:
            print(f"工具状态：succeeded；{output_field}：{len(content)}字符")
        return content

    def search_and_answer(self, question, max_iterations=5):
        """从新问题开始，执行完整循环；每次提问都重置历史。"""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题必须是非空字符串。")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("模型轮数上限必须是正整数。")
        self.question = question
        self.conversation_history = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": question},
        ]
        self.tools = None
        self.trace = []
        self.api_turns = []
        self.model_turns = 0
        self.finish_reason = None
        self.status = "running"
        self.answer_complete = False
        self.answer = ""

        # 对应06的循环；这次从第1轮开始，不读取03至06的旧问题记录。
        for iteration in range(1, max_iterations + 1):
            if self.verbose:
                print(f"\n第{iteration}/{max_iterations}轮模型请求：{len(self.conversation_history)}条消息", flush=True)
            choice = self._chat()
            self.model_turns = iteration
            self.finish_reason = choice.finish_reason
            calls = []
            for call in choice.message.tool_calls or []:
                calls.append({
                    "id": call.id, "type": call.type,
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                })
            message = {"role": "assistant", "content": choice.message.content or ""}
            if calls:
                message["tool_calls"] = calls
            self.conversation_history.append(message)
            if self.verbose:
                print("本轮结束原因：", choice.finish_reason)

            # 对应06的结束分支：有完整回答时return，同时退出方法内的循环。
            if choice.finish_reason == "stop" and message["content"].strip() and not calls:
                self.answer = message["content"]
                self.answer_complete = True
                self.status = "answered"
                self.trace.append({"iteration": iteration, "type": "answer", "content": self.answer})
                return self.answer
            if choice.finish_reason != "tool_calls" or not calls:
                self.status = "truncated" if choice.finish_reason == "length" else "invalid_response"
                raise RuntimeError("模型回复被截断或无效，尚未得到完整答案。")
            if iteration == max_iterations:
                self.status = "max_turns_reached"
                raise RuntimeError("模型请求轮数已达到上限，仍有新调用，尚未完成问答。")
            ids = [call["id"] for call in calls]
            if len(set(ids)) != len(ids):
                raise RuntimeError("本轮调用ID重复，无法匹配结果。")
            for call in calls:
                function = call["function"]
                if function["name"] != "web_search" or not isinstance(function["arguments"], str):
                    raise RuntimeError("当前学习版只执行web_search，参数必须保留为原始字符串。")

            # 对应04执行工具及05追加tool消息：所有结果返回后进入下一轮模型请求。
            for call in calls:
                function = call["function"]
                self.trace.append({
                    "iteration": iteration, "type": "action", "tool_call_id": call["id"],
                    "tool": function["name"], "raw_arguments": function["arguments"],
                })
                if self.verbose:
                    print("调用ID：", call["id"])
                    print("搜索参数：", function["arguments"], flush=True)
                content = self._execute_formula(function["name"], function["arguments"])
                self.conversation_history.append({"role": "tool", "tool_call_id": call["id"], "content": content})
                self.trace.append({
                    "iteration": iteration, "type": "observation", "tool_call_id": call["id"], "content": content,
                })

    def clear_history(self):
        """对应官方clear_history，供交互入口的clear命令调用。"""
        self.conversation_history = []

    def get_record(self):
        """只提取学习记录字段；不把对象中的API密钥写入结果。"""
        record = {
            "question": self.question, "model": self.model, "base_url": self.base_url,
            "status": self.status, "model_turns": self.model_turns, "finish_reason": self.finish_reason,
            "answer_complete": self.answer_complete, "answer": self.answer, "tools": self.tools,
            "conversation_history": self.conversation_history, "trace": self.trace, "api_turns": self.api_turns,
        }
        return json.loads(json.dumps(record, ensure_ascii=False))

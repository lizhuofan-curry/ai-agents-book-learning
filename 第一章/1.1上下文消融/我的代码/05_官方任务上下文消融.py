"""实验1.1：用DeepSeek运行官方多币种任务的上下文消融。

不传参数时只运行full基线，节省首次验证成本。
可用模式：full、no_history、no_tool_calls、no_tool_results。
当前DeepSeek思考模式要求工具请求回传reasoning_content，因此不能把
no_reasoning伪装成与官方定义等价的有效组。
"""

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from finance_tools import FINANCE_TOOLS, execute_finance_tool


MODES = ("full", "no_history", "no_tool_calls", "no_tool_results")
TASK = """根据公司的季度收入：
- 第一季度：2,500,000美元（USD）
- 第二季度：2,100,000欧元（EUR）
- 第三季度：1,800,000英镑（GBP）
- 第四季度：380,000,000日元（JPY）

请使用可用的货币换算工具，把所有非美元收入换算为美元，再使用计算工具算出全年总收入和季度平均收入。两个结果均保留两位小数。不得自行估计汇率，必须使用工具返回的观测结果。"""
EXPECTED_NUMBERS = ("9602895.73", "2400723.93")


def tool_calls_to_dict(tool_calls):
    return [
        {
            "id": call.id,
            "type": call.type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in tool_calls
    ]


class AblationAgent:
    def __init__(self, client, model, mode, max_turns=5):
        self.client = client
        self.model = model
        self.mode = mode
        self.max_turns = max_turns
        self.history = []
        self.api_turns = []
        self.executed_tools = []

    def _messages_for_request(self):
        if self.mode == "no_history":
            return copy.deepcopy(self.history[:2])
        return copy.deepcopy(self.history)

    def run(self):
        self.history = [
            {
                "role": "system",
                "content": (
                    "你是财务计算Agent。请使用提供的工具，让每个数值结果都有工具观测依据，"
                    "并用简洁中文回答。"
                ),
            },
            {"role": "user", "content": TASK},
        ]
        final_answer = ""
        status = "running"

        for turn_number in range(1, self.max_turns + 1):
            request_messages = self._messages_for_request()
            request = {
                "model": self.model,
                "messages": request_messages,
                "max_tokens": 4096,
                "reasoning_effort": "high",
                "thinking": {"type": "enabled"},
            }
            create_kwargs = {
                "model": self.model,
                "messages": request_messages,
                "max_tokens": 4096,
                "reasoning_effort": "high",
                "extra_body": {"thinking": {"type": "enabled"}},
            }
            if self.mode != "no_tool_calls":
                request["tools"] = FINANCE_TOOLS
                create_kwargs["tools"] = FINANCE_TOOLS

            print(
                f"  第{turn_number}/{self.max_turns}轮："
                f"发送{len(request_messages)}条消息，"
                f"工具定义={'有' if 'tools' in request else '无'}"
            )
            response = self.client.chat.completions.create(**create_kwargs)
            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls or []
            call_dicts = tool_calls_to_dict(tool_calls)
            response_dict = response.model_dump()
            self.api_turns.append(
                {
                    "turn": turn_number,
                    "request": request,
                    "response": response_dict,
                }
            )
            print("  结束原因：", choice.finish_reason)

            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "reasoning_content": getattr(message, "reasoning_content", None),
                "tool_calls": call_dicts,
            }
            self.history.append(assistant_message)

            if tool_calls:
                for call in tool_calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError as exc:
                        arguments = {}
                        result = {"error": f"工具参数不是有效JSON：{exc}"}
                    else:
                        result = execute_finance_tool(call.function.name, arguments)
                    visible_content = (
                        ""
                        if self.mode == "no_tool_results"
                        else json.dumps(result, ensure_ascii=False)
                    )
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": visible_content,
                        }
                    )
                    self.executed_tools.append(
                        {
                            "turn": turn_number,
                            "id": call.id,
                            "name": call.function.name,
                            "arguments": arguments,
                            "actual_result": result,
                            "content_sent_to_model": visible_content,
                        }
                    )
                    print("  执行工具：", call.function.name, arguments)
                continue

            final_answer = (message.content or "").strip()
            if choice.finish_reason == "stop" and final_answer:
                status = "answered"
            else:
                status = "invalid_terminal_response"
            break
        else:
            status = "max_turns_reached"

        normalized = final_answer.replace(",", "").replace("$", "").replace(" ", "")
        correct = bool(final_answer) and all(n in normalized for n in EXPECTED_NUMBERS)
        signatures = [
            item["name"] + ":" + json.dumps(item["arguments"], sort_keys=True)
            for item in self.executed_tools
        ]
        return {
            "mode": self.mode,
            "provider": "deepseek",
            "model": self.model,
            "status": status,
            "completed": status == "answered",
            "task_success": correct,
            "turn_count": len(self.api_turns),
            "tool_action_count": len(self.executed_tools),
            "repeated_tool_actions": len(signatures) - len(set(signatures)),
            "final_answer": final_answer,
            "executed_tools": self.executed_tools,
            "api_turns": self.api_turns,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="实验1.1：DeepSeek官方任务上下文消融；默认只跑full基线。"
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["full"])
    parser.add_argument("--max-turns", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com"
    model = os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-flash"
    timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在项目根目录.env中填写DEEPSEEK_API_KEY。")
    if args.max_turns < 2:
        raise SystemExit("--max-turns至少为2。")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )
    print("实验1.1：官方多币种任务上下文消融。")
    print("模型：", model)
    print("本次模式：", args.modes)
    print("正确答案：年总收入9602895.73美元，季度平均2400723.93美元。")

    arms = []
    for mode in args.modes:
        print(f"\n运行模式：{mode}")
        agent = AblationAgent(client, model, mode, args.max_turns)
        try:
            arm = agent.run()
        except Exception as exc:
            arm = {
                "mode": mode,
                "provider": "deepseek",
                "model": model,
                "status": "error",
                "completed": False,
                "task_success": False,
                "turn_count": len(agent.api_turns),
                "tool_action_count": len(agent.executed_tools),
                "executed_tools": agent.executed_tools,
                "api_turns": agent.api_turns,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            print("  模式运行失败：", exc)
        arms.append(arm)
        print(
            "  结果：",
            f"status={arm['status']}, completed={arm['completed']}, "
            f"task_success={arm['task_success']}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "运行结果"
        / "上下文消融"
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "experiment_id": "1-1",
        "provider": "deepseek",
        "model": model,
        "task": TASK,
        "expected_numbers": list(EXPECTED_NUMBERS),
        "requested_modes": args.modes,
        "exact_official_five_arms": False,
        "limitation": (
            "DeepSeek思考模式的工具调用要求保留reasoning_content，因此这里没有把"
            "官方no_reasoning历史消融伪装成有效实验组。"
        ),
        "arms": arms,
    }
    output_path = output_dir / "evidence.json"
    output_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n证据已保存到：", output_path)
    if args.modes == ["full"] and not arms[0].get("task_success"):
        raise RuntimeError("full基线未得到官方正确答案，暂不运行其他组。")


if __name__ == "__main__":
    main()

"""实验1.1第四步：从用户问题开始，自动完成完整工具调用循环。"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calculator_tool import CALCULATOR_TOOLS, execute_tool


class CalculatorAgent:
    def __init__(self, api_key, base_url, model, timeout=180, verbose=True):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.verbose = verbose
        self.messages = []
        self.turns = []
        self.answer = ""
        self.status = "not_started"

    # 表示放在 CalculatorAgent 类中的工具方法，但它不需要访问当前的Agent对象
    # 因此参数中没有 self
    @staticmethod
    def _tool_calls_to_dict(tool_calls):
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

    def run(self, question, max_turns=3):
        self.messages = [
            {
                "role": "system",
                "content": "你是计算助手。需要计算时调用calculate，依据实际工具结果回答。",
            },
            {"role": "user", "content": question},
        ]
        self.turns = []
        self.answer = ""
        self.status = "running"

        for turn_number in range(1, max_turns + 1):
            if self.verbose:
                print(f"\n第{turn_number}/{max_turns}轮模型请求：{len(self.messages)}条消息")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=CALCULATOR_TOOLS,
                max_tokens=2048,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls or []
            call_dicts = self._tool_calls_to_dict(tool_calls)

            # DeepSeek工具对话必须保留reasoning_content，但不在终端展示。
            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "reasoning_content": getattr(message, "reasoning_content", None),
                "tool_calls": call_dicts,
            }
            self.messages.append(assistant_message)
            turn_record = {
                "turn": turn_number,
                "sent_message_count": len(self.messages) - 1,
                "finish_reason": choice.finish_reason,
                "tool_calls": call_dicts,
                "tool_results": [],
                "usage": response.usage.model_dump() if response.usage else None,
            }
            self.turns.append(turn_record)
            if self.verbose:
                print("本轮结束原因：", choice.finish_reason)

            if tool_calls:
                for call in tool_calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("模型返回的工具参数不是有效JSON。") from exc
                    result = execute_tool(call.function.name, arguments)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                    self.messages.append(tool_message)
                    turn_record["tool_results"].append(
                        {
                            "tool_call_id": call.id,
                            "name": call.function.name,
                            "arguments": arguments,
                            "result": result,
                        }
                    )
                    if self.verbose:
                        print("执行工具：", call.function.name, arguments)
                        print("工具结果：", result)
                continue

            answer = (message.content or "").strip()
            if choice.finish_reason == "stop" and answer:
                self.answer = answer
                self.status = "answered"
                return answer
            raise RuntimeError(f"模型未正常结束：{choice.finish_reason}")

        self.status = "max_turns_reached"
        raise RuntimeError("达到模型请求轮数上限，仍未得到最终回答。")

    def get_record(self, question, max_turns):
        return {
            "provider": "deepseek",
            "model": self.model,
            "question": question,
            "max_turns": max_turns,
            "status": self.status,
            "answer_complete": self.status == "answered",
            "answer": self.answer,
            "turns": self.turns,
            "messages": self.messages,
        }


def main():
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com"
    model = os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-flash"
    timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在项目根目录.env中填写DEEPSEEK_API_KEY。")

    question = "请使用calculate工具计算(246 + 754) * 6，依据工具结果回答。"
    max_turns = 3
    agent = CalculatorAgent(api_key, base_url, model, timeout, verbose=True)
    error = None
    print("实验1.1：从问题开始运行完整计算Agent循环。")
    print("请求模型：", model)
    print("用户问题：", question)
    try:
        answer = agent.run(question, max_turns=max_turns)
        print("\nDeepSeek最终回答：", answer)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        print("本次Agent运行未完成：", exc)
    finally:
        record = agent.get_record(question, max_turns)
        if error:
            record["error"] = error
        output = Path(__file__).resolve().parents[1] / "运行结果" / "deepseek_full_loop.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("记录已保存到：", output)
    if error:
        raise RuntimeError(error["message"])


if __name__ == "__main__":
    main()

"""实验1.1第二步：把计算器说明交给DeepSeek，观察它提出调用。

本步只发送一次模型请求并保存调用；不执行计算器，不回传工具结果。
DeepSeek使用OpenAI兼容接口：https://api-docs.deepseek.com/guides/tool_calls/
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 这是给模型看的工具说明，并不是calculate函数本身。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算由数字、括号及加减乘除组成的算式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的算式，例如(125 + 375) * 8。",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def main():
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com"
    model = os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-flash"
    timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在项目根目录.env中填写DEEPSEEK_API_KEY。")

    # DeepSeek兼容OpenAI接口，所以仍使用同一个OpenAI客户端类。
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )

    question = "请使用calculate工具计算(125 + 375) * 8，获得工具结果后再回答。"
    messages = [
        {
            "role": "system",
            "content": "你是计算助手。需要计算时调用calculate，依据实际工具结果回答。",
        },
        {"role": "user", "content": question},
    ]

    print("实验1.1：让DeepSeek提出计算调用，尚未执行本地工具。")
    print("请求模型：", model)
    print("用户问题：", question)
    print("把问题与工具说明交给DeepSeek……", flush=True)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        max_tokens=2048,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    choice = response.choices[0]
    print("本轮结束原因：", choice.finish_reason)
    calls = []
    for tool_call in choice.message.tool_calls or []:
        call = {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        calls.append(call)
        print("调用ID：", tool_call.id)
        print("工具名称：", tool_call.function.name)
        print("参数原文：", tool_call.function.arguments)
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            print("参数不是有效JSON，需要检查；本步不执行工具。")
        else:
            print("解析后的参数：", arguments)

    # DeepSeek思考模式可能返回reasoning_content，后续回传工具结果时需要保留。
    reasoning_content = getattr(choice.message, "reasoning_content", None)
    record = {
        "provider": "deepseek",
        "model": response.model,
        "question": question,
        "tools": TOOLS,
        "messages": messages,
        "finish_reason": choice.finish_reason,
        "tool_calls": calls,
        "assistant_content": choice.message.content,
        "reasoning_content": reasoning_content,
        "calculator_executed": False,
    }
    output = (
        Path(__file__).resolve().parents[1]
        / "运行结果"
        / "deepseek_calculator_request.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("记录已保存到：", output)

    if choice.finish_reason != "tool_calls" or not calls:
        raise RuntimeError("本轮未观察到工具调用请求，请检查保存的响应。")
    print("已观察到调用请求；下一步由Python执行计算器，再把结果交回DeepSeek。")


if __name__ == "__main__":
    main()

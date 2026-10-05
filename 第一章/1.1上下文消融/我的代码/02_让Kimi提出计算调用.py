"""实验1.1第二步：自己定义计算器说明，观察Kimi提出调用。

本步只发送一次模型请求、保存调用；不执行计算器，不回传工具结果。
参考官方context/agent.py中calculate工具定义与工具调用流程。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 工具说明由我们编写，名称对应下一步会执行的本地calculate函数。
# 这份字典只描述函数，并没有把Python函数发送给模型。
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
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在根目录.env填写MOONSHOT_API_KEY。")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    question = "请使用calculate工具计算(125 + 375) * 8，获得工具结果后再回答。"
    messages = [
        {"role": "system", "content": "你是计算助手。需要计算时调用calculate，依据实际工具结果回答。"},
        {"role": "user", "content": question},
    ]
    print("实验1.1：让Kimi提出计算调用，尚未执行本地工具。")
    print("用户问题：", question)
    print("工具说明：")
    print(json.dumps(TOOLS, ensure_ascii=False, indent=2))
    print("把问题与工具说明交给Kimi……", flush=True)
    response = client.chat.completions.create(
        model="kimi-k3",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=1,
        max_tokens=32768,
        reasoning_effort="max",
    )
    choice = response.choices[0]
    print("本轮结束原因：", choice.finish_reason)
    calls = []
    for tool_call in choice.message.tool_calls or []:
        calls.append({
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        })
        print("调用ID：", tool_call.id)
        print("工具名称：", tool_call.function.name)
        print("参数原文：", tool_call.function.arguments)
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            print("参数不是有效JSON，需要检查；本步不执行工具。")
        else:
            print("解析后的参数：", arguments)

    record = {
        "model": response.model,
        "question": question,
        "tools": TOOLS,
        "messages": messages,
        "finish_reason": choice.finish_reason,
        "tool_calls": calls,
        "assistant_content": choice.message.content,
        "calculator_executed": False,
    }
    output = Path(__file__).resolve().parents[1] / "运行结果" / "kimi_calculator_request.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("记录已保存到：", output)
    if choice.finish_reason != "tool_calls" or not calls:
        raise RuntimeError("本轮未观察到完整的工具调用请求，请检查保存的响应。")
    print("已观察到调用请求；下一步由Python执行计算器，再把结果交回Kimi。")


if __name__ == "__main__":
    main()

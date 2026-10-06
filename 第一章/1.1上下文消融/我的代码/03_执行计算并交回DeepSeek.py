"""实验1.1第三步：执行02保存的调用，并把工具结果交回DeepSeek。

本步只处理02已经提出的一个calculate调用，再发送一次模型请求。
尚未实现可处理任意轮数的完整Agent循环。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from calculator_tool import calculate


def main():
    experiment_dir = Path(__file__).resolve().parents[1]
    input_path = experiment_dir / "运行结果" / "deepseek_calculator_request.json"
    if not input_path.exists():
        raise SystemExit("请先成功运行DeepSeek版02，生成调用记录。")

    record = json.loads(input_path.read_text(encoding="utf-8"))
    calls = record.get("tool_calls") or []
    if record.get("provider") != "deepseek":
        raise RuntimeError("02记录不是DeepSeek生成的，停止回传。")
    if record.get("calculator_executed") is not False:
        raise RuntimeError("02记录不处于尚未执行工具的状态。")
    if len(calls) != 1:
        raise RuntimeError("本学习步骤只处理一个工具调用。")

    call = calls[0]
    if call.get("function", {}).get("name") != "calculate":
        raise RuntimeError("模型提出的不是calculate调用。")
    try:
        arguments = json.loads(call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("工具参数不是有效JSON。") from exc
    if set(arguments) != {"expression"} or not isinstance(arguments["expression"], str):
        raise RuntimeError("calculate参数必须只有一个字符串expression。")

    # 到这一行，Python才真正执行本地工具。
    tool_result = calculate(arguments["expression"])
    tool_content = json.dumps(tool_result, ensure_ascii=False)
    print("实验1.1：执行02提出的本地计算调用，再把结果交回DeepSeek。")
    print("调用ID：", call["id"])
    print("执行函数： calculate")
    print("传入参数：", arguments)
    print("本地工具结果：", tool_result)

    # assistant消息说明“模型刚才提出了什么调用”。
    # reasoning_content由DeepSeek返回；工具对话的下一次请求必须原样保留。
    assistant_message = {
        "role": "assistant",
        "content": record.get("assistant_content"),
        "reasoning_content": record.get("reasoning_content"),
        "tool_calls": calls,
    }
    # tool_call_id把结果与前面的调用一一对应。
    tool_message = {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": tool_content,
    }
    messages = list(record["messages"])
    messages.append(assistant_message)
    messages.append(tool_message)

    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com"
    model = os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-flash"
    timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在项目根目录.env中填写DEEPSEEK_API_KEY。")
    if model != record.get("model"):
        raise RuntimeError(
            f"02使用{record.get('model')}，当前配置为{model}；请保持同一模型。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )
    print("发送给DeepSeek的消息数：", len(messages))
    print("等待DeepSeek依据工具结果回答……", flush=True)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=record["tools"],
        max_tokens=2048,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    choice = response.choices[0]
    answer = (choice.message.content or "").strip()
    new_calls = choice.message.tool_calls or []
    print("本轮结束原因：", choice.finish_reason)
    if answer:
        print("DeepSeek回答：", answer)
    if new_calls:
        print("DeepSeek又提出了", len(new_calls), "个工具调用。")

    output_record = {
        "provider": "deepseek",
        "model": response.model,
        "question": record["question"],
        "source_call": call,
        "tool_result": tool_result,
        "sent_messages": messages,
        "finish_reason": choice.finish_reason,
        "answer": answer,
        "new_tool_calls": [
            {
                "id": item.id,
                "type": item.type,
                "function": {
                    "name": item.function.name,
                    "arguments": item.function.arguments,
                },
            }
            for item in new_calls
        ],
        "answer_complete": choice.finish_reason == "stop" and bool(answer) and not new_calls,
    }
    output_path = experiment_dir / "运行结果" / "deepseek_calculator_reply.json"
    output_path.write_text(
        json.dumps(output_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("记录已保存到：", output_path)

    if not output_record["answer_complete"]:
        raise RuntimeError("本轮尚未得到完整最终回答，请检查保存记录。")


if __name__ == "__main__":
    main()

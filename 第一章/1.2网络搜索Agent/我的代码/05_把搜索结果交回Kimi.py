"""实验1.2：把04保存的工具结果交回Kimi，观察模型的下一轮回复。

对应官方search_and_answer中追加assistant/tool消息、再次调用_chat的环节。
学习改动：复用03、04的真实记录，显示消息对应关系，只发送一次模型请求。
来源：https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main():
    # 1. 配置与03相同。这次用OpenAI客户端请求Kimi继续处理对话。
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在项目根目录的.env中填写MOONSHOT_API_KEY。")

    # 2. 03记录了模型要求搜什么；04记录了每个调用实际返回什么。
    results_dir = Path(__file__).resolve().parents[1] / "运行结果"
    request_path = results_dir / "kimi_tool_request.json"
    search_path = results_dir / "kimi_search_results.json"
    if not request_path.is_file() or not search_path.is_file():
        raise SystemExit("请先完成03和04，保留各自生成的JSON文件。")
    previous = json.loads(request_path.read_text(encoding="utf-8"))
    searches = json.loads(search_path.read_text(encoding="utf-8"))
    tool_calls = previous.get("tool_calls") or []
    search_results = searches.get("search_results") or []

    # 检查两份记录属于同一轮，并且每个调用都有一个成功结果。
    # 如果重新运行了03，就必须重新执行对应的04，不能混用旧结果。
    if previous.get("finish_reason") != "tool_calls" or not tool_calls:
        raise RuntimeError("03的记录中没有有效工具调用。")
    if searches.get("question") != previous["question"] or searches.get("tool_calls") != tool_calls:
        raise RuntimeError("03和04的记录不匹配，不能把旧搜索结果交给新的调用。")
    if searches.get("all_searches_succeeded") is not True or searches.get("completed_searches") != len(tool_calls):
        raise RuntimeError("04尚未完成本轮全部搜索，请先检查04的输出。")
    call_ids = [call["id"] for call in tool_calls]
    result_ids = [item["tool_call_id"] for item in search_results]
    if len(set(call_ids)) != len(call_ids) or sorted(call_ids) != sorted(result_ids):
        raise RuntimeError("调用ID与结果ID必须一一对应，不能遗漏或重复。")
    for call in tool_calls:
        item = next(item for item in search_results if item["tool_call_id"] == call["id"])
        if item.get("request") != call["function"] or item.get("status") != "succeeded":
            raise RuntimeError("搜索结果的参数或成功状态与调用不匹配。")
        if not isinstance(item.get("content"), str) or not item["content"]:
            raise RuntimeError("工具结果必须是非空字符串。")

    # 3. 保留原来的system和user消息，再补上Kimi提出调用的assistant消息。
    # list(...)复制外层列表；append(...)把一条新消息加入列表末尾。
    # previous是03保存的记录
    messages = list(previous["messages"])
    messages.append({
        "role": "assistant",
        "content": previous.get("assistant_content") or "",
        "tool_calls": tool_calls,
    })

    # 4. 每份工具结果各是一条tool消息，用tool_call_id说明它回答的是哪个调用。
    # encrypted_output已经保存在content中；原样传回，不解析、不修改。
    for item in search_results:
        # 补上KImi 之前提出搜索调用的记录
        messages.append({
            "role": "tool",
            "tool_call_id": item["tool_call_id"],
            "content": item["content"],
        })

    print("实验1.2：把真实搜索结果交回Kimi，发送一次后续模型请求。")
    print("用户问题：", previous["question"])
    print("本次发送的消息数：", len(messages))
    for number, message in enumerate(messages, start=1):
        if message["role"] == "tool":
            print(f"[{number}] tool → {message['tool_call_id']}，结果长度：{len(message['content'])}字符")
        elif message["role"] == "assistant":
            print(f"[{number}] assistant → 提出了{len(message['tool_calls'])}个工具调用")
        else:
            print(f"[{number}] {message['role']} → {message['content']}")

    # 5. 仍然提供tools，让Kimi能够自行决定回答还是继续搜索。
    # 三份结果已经在messages里，一次create(...)把完整列表交给模型。
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    print("等待Kimi读取结果并继续处理……", flush=True)
    response = client.chat.completions.create(
        model="kimi-k3",
        messages=messages,
        tools=previous["tools"],
        tool_choice="auto",
        temperature=1,
        max_tokens=32768,
        reasoning_effort="max",
    )

    # 6. 与03一样，提取本轮结束原因、回答内容和可能出现的新调用。
    choice = response.choices[0]
    new_calls = []
    for call in choice.message.tool_calls or []:
        new_calls.append({
            "id": call.id,
            "type": call.type,
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        })
    assistant_message = {"role": "assistant", "content": choice.message.content or ""}
    if new_calls:
        assistant_message["tool_calls"] = new_calls
    answer_complete = choice.finish_reason == "stop" and bool(assistant_message["content"].strip()) and not new_calls
    result = {
        "question": previous["question"],
        "model": response.model,
        "tools": previous["tools"],
        "messages": messages,
        "tool_results_returned": len(search_results),
        "finish_reason": choice.finish_reason,
        "assistant_content": assistant_message["content"],
        "tool_calls": new_calls,
        "answer_complete": answer_complete,
        "conversation_history": messages + [assistant_message],
    }
    output = results_dir / "kimi_search_reply.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("本轮结束原因：", choice.finish_reason)
    print("结果已保存到：", output)

    if answer_complete:
        print("\nKimi基于搜索结果给出的回答：")
        print(assistant_message["content"])
        print("下一步对照来源检查答案，再把这些步骤连接成自动循环。")
    elif choice.finish_reason == "tool_calls" and new_calls:
        print("Kimi提出了新的搜索调用，本次还没有得到最终答案：")
        for call in new_calls:
            print("调用ID：", call["id"])
            print("工具名称：", call["function"]["name"])
            print("参数原文：", call["function"]["arguments"])
        print("新调用已经保存，下一步继续执行工具并回传结果。")
    elif choice.finish_reason == "length":
        raise RuntimeError("模型输出达到max_tokens上限，已保存响应，不能把它视为完整答案。")
    else:
        raise RuntimeError("本轮没有得到有效完整回答或有效工具调用，请检查保存的响应。")


if __name__ == "__main__":
    main()

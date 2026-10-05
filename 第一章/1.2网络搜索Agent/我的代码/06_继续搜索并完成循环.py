"""实验1.2：接着05的真实记录，循环执行搜索、回传结果、请求模型。

对应官方search_and_answer的多轮工具调用循环；模型请求总上限为5轮。
学习改动：从05继续，逐次保存进度，重启时跳过本轮已经成功的搜索。
来源：https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI


MAX_MODEL_TURNS = 5  # 包含03和05已经完成的2轮模型请求；不是搜索次数。


def save_progress(output, state):
    """把当前进度写入文件，便于查看及继续运行。"""
    output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute_search(call, execute_url, api_key, timeout):
    """把04中的执行步骤提取成函数：输入一个调用，返回一份搜索结果。"""
    body = call["function"]
    print("调用ID：", call["id"], flush=True)
    print("搜索参数：", body["arguments"], flush=True)
    response = requests.post(
        execute_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    fiber = response.json()
    if fiber.get("status") != "succeeded":
        raise RuntimeError(f"调用{call['id']}执行失败，状态：{fiber.get('status')}")
    context = fiber.get("context") or {}
    output_field = "output"
    content = context.get(output_field)
    if content in (None, ""):
        output_field = "encrypted_output"
        content = context.get(output_field)
    if content in (None, ""):
        raise RuntimeError("工具状态为succeeded，但没有返回结果内容。")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    print(f"HTTP状态：{response.status_code}；工具状态：succeeded；{output_field}：{len(content)}字符")
    return {
        "tool_call_id": call["id"],
        "request": body,
        "http_status": response.status_code,
        "fiber_id": fiber.get("id"),
        "status": "succeeded",
        "output_field": output_field,
        "content": content,
    }


def main():
    # 1. 沿用之前的配置、工具接口和模型客户端。
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在项目根目录的.env中填写MOONSHOT_API_KEY。")
    execute_url = f"{base_url.rstrip('/')}/formulas/moonshot/web-search:latest/fibers"
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    # 2. 05保存的conversation_history包括它收到的消息和最新的assistant回复。
    results_dir = Path(__file__).resolve().parents[1] / "运行结果"
    input_path = results_dir / "kimi_search_reply.json"
    if not input_path.is_file():
        raise SystemExit("请先运行05，保留kimi_search_reply.json。")
    previous = json.loads(input_path.read_text(encoding="utf-8"))
    if previous.get("answer_complete"):
        print("05已经得到回答，先对照来源检查答案，再继续学习完整实现。")
        return
    calls = previous.get("tool_calls") or []
    if previous.get("finish_reason") != "tool_calls" or not calls:
        raise RuntimeError("05没有有效的待执行调用，请先检查05的输出。")
    initial_history = previous["conversation_history"]
    if initial_history != previous["messages"] + [{
        "role": "assistant", "content": previous.get("assistant_content") or "", "tool_calls": calls,
    }]:
        raise RuntimeError("05的历史消息与最新回复不一致，请检查保存文件。")

    # 3. 第一次从05继续；已有06记录时，读取其中保存的进度。
    output = results_dir / "kimi_agent_loop.json"
    if output.is_file():
        state = json.loads(output.read_text(encoding="utf-8"))
        if state.get("source_history") != initial_history or state.get("tools") != previous["tools"]:
            raise RuntimeError("06的记录属于另一次05运行。请先另存旧06记录，再继续新问题。")
        if state.get("answer_complete"):
            print("本问题已经完成，读取已保存的回答：")
            print(state["answer"])
            return
        if state.get("status") in ("truncated", "invalid_response"):
            raise RuntimeError("已有记录包含截断或无效回复，请先检查该记录。")
        print("读取06已保存的进度，继续处理待完成的调用。")
    else:
        state = {
            "question": previous["question"],
            "tools": previous["tools"],
            "source_history": initial_history,
            "conversation_history": list(initial_history),
            "model_turns": sum(message["role"] == "assistant" for message in initial_history),
            "pending_calls": calls,
            "search_results": [],
            "finish_reason": "tool_calls",
            "status": "needs_tools",
            "answer_complete": False,
            "answer": "",
        }
        save_progress(output, state)
    messages = state["conversation_history"]  # 两个名字指向同一个列表，append也会更新state。
    print("实验1.2：接着05，循环执行工具并请求Kimi。")
    print("已完成的模型请求轮数：", state["model_turns"])

    # 4. 循环上限计算的是模型请求；每轮可以包含多个搜索调用。
    while state["model_turns"] < MAX_MODEL_TURNS:
        calls = state["pending_calls"]
        call_ids = [call["id"] for call in calls]
        if not calls or len(call_ids) != len(set(call_ids)):
            raise RuntimeError("本轮必须有有效且ID不重复的调用。")
        for call in calls:
            function = call.get("function") or {}
            if function.get("name") != "web_search" or not isinstance(function.get("arguments"), str):
                raise RuntimeError("当前脚本只执行web_search，参数必须保留为原始字符串。")
        round_number = state["model_turns"]
        print(f"\n处理第{round_number}轮模型提出的{len(calls)}个搜索调用：")

        # 5. 执行本轮所有工具，并把各自的结果作为tool消息加入历史。
        for call in calls:
            # 只匹配当前轮次：调用ID的唯一性由服务端保证在同一轮内成立。
            completed = any(
                item["model_turn"] == round_number and item["tool_call_id"] == call["id"]
                for item in state["search_results"]
            )
            if completed:
                print("复用本轮已经保存的成功结果：", call["id"])
                continue
            item = execute_search(call, execute_url, api_key, timeout)
            item["model_turn"] = round_number
            state["search_results"].append(item)
            messages.append({
                "role": "tool", "tool_call_id": call["id"], "content": item["content"],
            })
            save_progress(output, state)

        # 6. 全部工具结果加入后，再发送完整历史；这就是05的模型请求步骤。
        state["status"] = "ready_for_model"
        save_progress(output, state)
        print(f"发送第{round_number + 1}轮模型请求，共{len(messages)}条消息……", flush=True)
        response = client.chat.completions.create(
            model="kimi-k3", messages=messages, tools=state["tools"], tool_choice="auto",
            temperature=1, max_tokens=32768, reasoning_effort="max",
        )
        choice = response.choices[0]
        state["model_turns"] += 1
        state["model"] = response.model
        state["finish_reason"] = choice.finish_reason
        new_calls = []
        for call in choice.message.tool_calls or []:
            new_calls.append({
                "id": call.id, "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            })
        assistant_message = {"role": "assistant", "content": choice.message.content or ""}
        if new_calls:
            assistant_message["tool_calls"] = new_calls
        messages.append(assistant_message)
        state["pending_calls"] = new_calls
        print("本轮结束原因：", choice.finish_reason)

        # 7. stop得到回答后break退出；tool_calls回到while开头继续下一轮。
        if choice.finish_reason == "stop" and assistant_message["content"].strip() and not new_calls:
            state["answer"] = assistant_message["content"]
            state["answer_complete"] = True
            state["status"] = "answered"
            save_progress(output, state)
            print("\nKimi基于搜索结果给出的回答：")
            print(state["answer"])
            break
        if choice.finish_reason == "tool_calls" and new_calls:
            state["status"] = "needs_tools"
            save_progress(output, state)
            for call in new_calls:
                print("新的调用：", call["id"], call["function"]["arguments"])
        else:
            state["status"] = "truncated" if choice.finish_reason == "length" else "invalid_response"
            save_progress(output, state)
            raise RuntimeError("本轮回复被截断或无效，已保存响应，尚未得到完整答案。")

    if not state["answer_complete"]:
        state["status"] = "max_turns_reached"
        save_progress(output, state)
        raise RuntimeError(f"已达到{MAX_MODEL_TURNS}轮模型请求上限，仍有待处理调用；不能视为任务完成。")
    print("\n循环结果已保存到：", output)
    print("下一步对照来源检查答案，继续完成官方README的结果分析与用法练习。")


if __name__ == "__main__":
    main()

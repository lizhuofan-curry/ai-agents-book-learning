"""实验1.2：执行03保存的搜索调用，观察并保存工具的真实返回。

按官方_execute_formula拆出执行环节；下一步再把结果交回Kimi。
学习改动：读取本地03的记录、逐条显示执行状态、每次成功后保存结果。
来源：https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def main():
    # 1. 沿用根目录配置；这一步只执行搜索工具，不发送新的模型对话。
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在项目根目录的.env中填写MOONSHOT_API_KEY。")

    # 2. 读取你刚才真实运行03保存的记录，接着执行其中的tool_calls。
    results_dir = Path(__file__).resolve().parents[1] / "运行结果"
    input_path = results_dir / "kimi_tool_request.json"
    if not input_path.is_file():
        raise SystemExit("请先运行03脚本，生成kimi_tool_request.json。")
    previous = json.loads(input_path.read_text(encoding="utf-8"))
    tool_calls = previous.get("tool_calls") or []
    if previous.get("finish_reason") != "tool_calls" or not tool_calls:
        raise RuntimeError("03的记录没有有效工具调用，请先检查03的输出。")
    for call in tool_calls:
        if call.get("function", {}).get("name") != "web_search":
            raise RuntimeError("当前学习脚本只执行web_search，请检查保存的工具名称。")
        if not isinstance(call["function"].get("arguments"), str):
            raise RuntimeError("工具参数必须保留为模型返回的原始JSON字符串。")

    # 3. 与03相同的工具标识；地址末尾改为/fibers，表示提交一次工具执行。
    formula_uri = "moonshot/web-search:latest"
    execute_url = f"{base_url.rstrip('/')}/formulas/{formula_uri}/fibers"
    output = results_dir / "kimi_search_results.json"
    search_results = []
    result = {
        "question": previous["question"],
        "formula_uri": formula_uri,
        "tool_calls": tool_calls,
        "completed_searches": 0,
        "all_searches_succeeded": False,
        "search_results": search_results,
    }
    # 初始化本次记录，避免把上次运行的文件误认为本次全部成功。
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("实验1.2：执行03保存的真实搜索请求。")
    print("用户问题：", previous["question"])
    print("待执行的搜索数：", len(tool_calls))

    # 4. 逐个处理模型提出的调用，每个ID对应一份工具结果。
    # name 告诉服务端执行哪个工具
    # arguments 提供搜索参数，保留模型返回的原始字符串
    for number, call in enumerate(tool_calls, start=1):
        # body是Python字典；其中arguments的值仍是原始字符串。
        body = {
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"],
        }
        print(f"\n[{number}/{len(tool_calls)}] 调用ID：{call['id']}", flush=True)
        print("参数原文：", body["arguments"], flush=True)

        # POST把工具名和参数提交给服务端；json=body由requests负责JSON编码。
        # 把工具名和参数提交给Moonshot,让服务端实际执行搜索
        fiber_response = requests.post(
            # 访问以 /fibers 结尾的工具执行接口
            execute_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
        # 收到 HTTP响应后，还要检查其中的 status确认工具确实执行成功
        fiber_response.raise_for_status()
        fiber = fiber_response.json()  # Fiber是服务端返回的这次执行记录。
        if fiber.get("status") != "succeeded":
            raise RuntimeError(f"调用{call['id']}执行失败，状态：{fiber.get('status')}")

        # 5. 读取结果。官方搜索工具可能返回供下一轮模型读取的加密输出。
        '''
        context.output 普通工具结果
        context.encrypted_output 供下一轮Kimi读取的加密工具结果
        '''
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

        search_results.append({
            "tool_call_id": call["id"],
            "request": body,
            "http_status": fiber_response.status_code,
            "fiber_id": fiber.get("id"),
            "status": fiber["status"],
            "output_field": output_field,
            "content": content,
        })
        print("HTTP状态：", fiber_response.status_code)
        print("工具执行状态：", fiber["status"])
        print("结果字段：", output_field)
        print("结果长度：", len(content), "字符")
        if output_field == "encrypted_output":
            print("结果是加密工具输出，按原文保存，供下一步交回Kimi。")
        else:
            print("结果预览：", content[:300])

        # 6. 每次成功后立即保存；后续请求失败时，已完成的记录仍在文件中。
        result["completed_searches"] = len(search_results)
        result["all_searches_succeeded"] = len(search_results) == len(tool_calls)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n全部搜索执行成功，结果已保存到：", output)
    print("下一步按调用ID把结果放入tool消息，再请求Kimi继续处理。")


if __name__ == "__main__":
    main()

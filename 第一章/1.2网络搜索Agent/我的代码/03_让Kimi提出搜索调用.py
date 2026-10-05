"""实验1.2：获取官方搜索工具定义，观察Kimi提出的真实工具调用。

按官方_get_tools和_chat拆出搜索循环的第一轮。
本脚本获取工具说明并发送一次模型请求；下一步继续执行搜索与回传结果。
来源：https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent
"""

import json
import os
from pathlib import Path

# 这里的requests是第三方 HTTP 库,负责通过网络向服务器发送请求，接收返回内容
# 平时在浏览器中输入网址，浏览器会向网站服务器请求页面，requests就是帮我们完成这件事
# 02_接入kimi api代码中的client.chat.completions.create(...)
# 是由 openai SDK帮我们发送模型对话请求，现在我们要访问Moonshot的工具接口，request直接请求这个接口
import requests
from dotenv import load_dotenv
from openai import OpenAI


def main():
    # 1. 沿用02脚本的根目录配置与客户端。
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在项目根目录的.env中填写MOONSHOT_API_KEY。")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    # 2. GET请求获取官方工具定义：它说明工具叫什么、做什么、接受什么参数。
    # 获取说明不会执行搜索。Authorization用于向服务端证明身份。

    # 这是一个普通的变量赋值
    # 只是 "moonshot/web-search:latest" 是 Moonshot 官方规定的工具标识
    # 可以先把Formula 理解成 Moonshot 托管的工具服务
    # 把 formula_uri 理解成“我要使用哪个工具”
    formula_uri = "moonshot/web-search:latest"
    # 组装成程序要访问的完整网络地址：https://api.moonshot.cn/v1/formulas/moonshot/web-search:latest/tools
    '''
    https://api.moonshot.cn/v1 Moonshot API的基础地址
    /formulas/ 是访问 Formula 工具接口
    moonshot/web-search:latest 指定搜索工具
    /tools 获取这个工具的定义
    '''
    tools_url = f"{base_url.rstrip('/')}/formulas/{formula_uri}/tools"
    print("获取Moonshot官方搜索工具定义……", flush=True)
    # requests.get 向Moonshot 请求工具定义
    # GET 是一种HTTP请求方式，用来请求读取某个地址上的资源
    # HTTP可以先理解成程序与服务器交流时遵循的一套规则
    tools_response = requests.get(
        tools_url,
        # Authorization 携带密钥，供服务端认证
        # headers 叫做请求头，用来请求携带一些附加信息，这里携带的是身份凭据
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    tools_response.raise_for_status()  # HTTP请求失败时，停止并显示错误。
    # .json 把返回的JSON解析成Python对象，再取出其中的tools
    tools = tools_response.json()["tools"]
    if not isinstance(tools, list) or not tools:
        raise RuntimeError("官方接口没有返回有效的工具定义列表。")
    if not any(tool.get("function", {}).get("name") == "web_search" for tool in tools):
        raise RuntimeError("官方返回的工具定义中没有web_search。")
    print("官方返回的工具定义：")
    print(json.dumps(tools, ensure_ascii=False, indent=2))

    # 3. 使用与离线演示相同的主题，这次让真实模型自己生成搜索参数。
    question = "请搜索Moonshot AI的Context Caching官方说明，解释这是什么技术，并提供来源链接。"
    messages = [
        {"role": "system", "content": "你是Kimi搜索助手。请使用web_search查询官方资料，再依据搜索结果回答。"},
        {"role": "user", "content": question},
    ]
    print("\n用户问题：", question)
    print("把问题和工具定义发送给Kimi……", flush=True)
    response = client.chat.completions.create(
        model="kimi-k3",
        messages=messages,
        # 告诉模型有哪些工具可以用
        tools=tools,  # 新增：把官方返回的工具定义原样交给模型。
        tool_choice="auto",  # 与接口默认行为一致，由模型决定是否调用工具。
        temperature=1,
        max_tokens=32768,
        reasoning_effort="max",
    )

    # 4. 工具调用包含ID、工具名和参数；这里先观察并保存它。
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
        # arguments本身是JSON字符串；解析后方便在Python中查看字段。
        # 这里只解析来展示，保存与后续执行时仍使用模型返回的原始字符串。
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            print("参数不是合法JSON，保留原文，后续执行前需要排查。")
        else:
            print("解析后的参数：", arguments)
            print("参数的类型变化：", type(tool_call.function.arguments).__name__, "→", type(arguments).__name__)

    output = Path(__file__).resolve().parents[1] / "运行结果" / "kimi_tool_request.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "question": question,
        "tools": tools,
        "messages": messages,
        "finish_reason": choice.finish_reason,
        "tool_calls": calls,
        "assistant_content": choice.message.content,
        "search_executed": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("结果已保存到：", output)
    if choice.finish_reason != "tool_calls" or not calls:
        raise RuntimeError("本轮没有提出工具调用，请查看保存的响应；尚未观察到搜索行动。")
    print("本轮已观察到真实工具调用请求；下一步执行搜索，并将结果交回模型。")


if __name__ == "__main__":
    main()

"""实验1.2：先理解一次真实Kimi请求，再给它接入搜索工具。

从官方config.py、WebSearchAgent.__init__和_chat中拆出配置与模型调用。
这个学习步骤只发送一次普通对话；后续继续完成真实搜索与结果分析。
来源：https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main():
    # 1. 找到根目录的.env，并把其中的配置载入当前进程的环境变量。
    # 这里的 parents[3] 找到 E:\AI_Agents_Book
    project_root = Path(__file__).resolve().parents[3]
    # 把.env 中的配置加载入当前Python进程的环境变量中
    load_dotenv(project_root / ".env")
    # os.getenv 读取指定变量    .strip() 去掉字符串两端的空白
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    model = "kimi-k3"  # 与当前官方实验的默认模型一致。
    if not api_key:
        raise SystemExit("请先在项目根目录的.env中填写MOONSHOT_API_KEY。")

    # 2. 创建SDK客户端。地址指向Moonshot，这里使用的是Kimi模型服务。
    # 创建客户端本身不发送问题；max_retries=0便于学习时直接看到首次错误。
    # OpenAI 作用是 openai 第三方提供的客户端类，负责组织和发送请求
    # base_url 指向 Moonshot 的服务地址
    # model 指定实际使用的kimi模型
    # max_retries=0 表示请求失败时直接显示错误，便于我们观察和排查
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    # 3. 给模型准备消息。messages是列表，每条消息是字典。
    question = "请用一句话解释ReAct中的行动和观察有什么区别。"
    messages = [
        {"role": "system", "content": "你是AI Agent学习助手，用简洁中文回答。"},
        {"role": "user", "content": question},
    ]
    print("实验1.2：一次真实Kimi对话，尚未接入搜索工具。")
    print("请求模型：", model)
    print("用户问题：", question)
    print("等待Kimi返回回答……", flush=True)

    # 4. 这一行才会向Kimi服务发送真实请求。参数与官方K3调用一致。
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # 控制 top_t
        temperature=1,
        # 设置输出token上限
        max_tokens=32768,
        # 设置推理强度
        reasoning_effort="max",
    )

    # 5. response是SDK响应对象，从第一个候选中取出助手消息的文本。
    choice = response.choices[0]
    print("本轮结束原因：", choice.finish_reason)
    if choice.finish_reason == "length":
        raise RuntimeError("回答被输出上限截断，不能把这次结果当作完整回答。")
    answer = (choice.message.content or "").strip()
    if choice.finish_reason != "stop" or not answer:
        raise RuntimeError("本轮未返回完整的普通对话答案，请检查响应与模型配置。")
    print("Kimi的实际回答：", answer)

    # 6. 保存本次问题与真实回答，沿用你已经理解的JSON保存方式。
    output = Path(__file__).resolve().parents[1] / "运行结果" / "kimi_chat.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "question": question,
        "messages": messages,
        "model": response.model,
        "finish_reason": choice.finish_reason,
        "answer": answer,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("结果已保存到：", output)


if __name__ == "__main__":
    main()

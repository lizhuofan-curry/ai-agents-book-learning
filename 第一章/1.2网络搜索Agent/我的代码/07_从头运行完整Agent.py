"""实验1.2：对应官方基础用法，从一个入口完成新的搜索问答。

先阅读07_阅读指南.md和web_search_agent.py，再运行本文件。
来源：https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from web_search_agent import WebSearchAgent


def main():
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在根目录.env中填写MOONSHOT_API_KEY。")

    # 创建我们自己定义的Agent对象，__init__会准备客户端和记录容器。
    agent = WebSearchAgent(api_key=api_key, base_url=base_url, timeout=timeout, verbose=True)
    # 沿用官方README中的Python版本主题，观察一个新的独立问题。
    question = "Python 3.12有哪些新特性？请搜索Python官方文档，概括3个变化并提供来源链接。"
    output = Path(__file__).resolve().parents[1] / "运行结果" / "kimi_agent_python312.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    error = None
    print("实验1.2：从一个入口运行完整搜索Agent。")
    print("用户问题：", question)
    try:
        # 调用自己的方法；方法内部获取定义、请求模型、执行工具、继续循环。
        answer = agent.search_and_answer(question, max_iterations=5)
        print("\nKimi的最终回答：")
        print(answer)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # 成功或失败都会保存已取得的记录；失败仍抛出异常，不冒充正常完成。
        record = agent.get_record()
        if error:
            record["error"] = error
            if record["status"] == "running":
                record["status"] = "failed"
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("记录已保存到：", output)


if __name__ == "__main__":
    main()

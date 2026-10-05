"""实验1.2：对应官方quickstart.py，用菜单选择真实搜索或交互模式。

保留官方的模式选择、三个演示问题、自定义问题与继续交互流程。
沿用本项目的.env和WebSearchAgent，每次问答单独保存记录。
来源：https://github.com/bojieli/ai-agent-book/blob/main/chapter1/web-search-agent/quickstart.py
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from web_search_agent import WebSearchAgent


def run_question(agent, question):
    """两个菜单分支共用的问答与保存过程，来自前面已学过的入口。"""
    results_dir = Path(__file__).resolve().parents[1] / "运行结果" / "引导式体验"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = results_dir / f"{timestamp}.json"
    error = None
    print("\n用户问题：", question)
    try:
        answer = agent.search_and_answer(question, max_iterations=5)
        print("\nKimi的最终回答：")
        print(answer)
    except KeyboardInterrupt:
        error = {"type": "KeyboardInterrupt", "message": "当前问答被用户中断。"}
        raise
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        print("本次问答未完成：", exc)
    finally:
        record = agent.get_record()
        if error:
            record["error"] = error
            if error["type"] == "KeyboardInterrupt":
                record["status"] = "interrupted"
            elif record["status"] == "running":
                record["status"] = "failed"
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("记录已保存到：", output)


def demo_search(agent):
    """对应官方demo_search：从列表选一个问题，或输入自己的问题。"""
    demo_questions = [
        "OpenAI 最新发布了什么产品？",
        "2024年有哪些重要的AI突破？",
        "如何开始学习机器学习？",
    ]
    print("\n选择一个演示问题，或输入自己的问题：")
    # enumerate把列表变为“显示编号、问题”；1表示显示编号从1开始。
    for i, question in enumerate(demo_questions, 1):
        print(f"{i}. {question}")
    print("0. 输入自定义问题")
    try:
        choice = int(input("请选择（0-3）：").strip())
    except ValueError:
        print("请输入数字。")
        return
    if choice == 0:
        question = input("请输入你的问题：").strip()
        if not question:
            print("问题不能为空。")
            return
    elif 1 <= choice <= len(demo_questions):
        # 菜单编号从1开始，列表索引从0开始，所以用choice - 1。
        question = demo_questions[choice - 1]
    else:
        print("无效的选项。")
        return
    run_question(agent, question)


def interactive_mode(agent):
    """对应官方interactive_mode：每个问题独立，输入退出命令结束。"""
    print("\n进入交互模式，每个问题独立处理；输入exit、quit或q退出。")
    while True:
        question = input("\n你的问题：").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        run_question(agent, question)


def main():
    # 新增的重点：先接收一个模式编号，再调用对应函数。
    print("实验1.2：官方引导式快速体验。")
    print("1. 真实搜索演示（选择一个问题）")
    print("2. 交互模式（连续输入独立问题）")
    print("3. 退出")
    mode = input("请选择（1-3）：").strip()
    if mode == "3":
        print("结束快速体验。")
        return
    if mode not in ("1", "2"):
        print("无效的模式。")
        return

    # 配置与07、08相同；输入3时已返回，不会进入这一部分。
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在根目录.env中填写MOONSHOT_API_KEY。")
    agent = WebSearchAgent(api_key=api_key, base_url=base_url, timeout=timeout, verbose=True)

    if mode == "1":
        demo_search(agent)
        # 与官方相同：一次演示后，可选择进入交互模式。
        cont = input("\n是否进入交互模式？（y/n）：").strip().lower()
        if cont == "y":
            interactive_mode(agent)
    elif mode == "2":
        interactive_mode(agent)
    print("结束快速体验。")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n结束快速体验。")

"""实验1.2：对应官方main.py的run_interactive_mode，在终端连续输入问题。

沿用已经跑通的WebSearchAgent，每个问题独立，每题单独保存运行记录。
来源：https://github.com/bojieli/ai-agent-book/blob/main/chapter1/web-search-agent/main.py
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from web_search_agent import WebSearchAgent


def main():
    # 1. 配置与07相同，创建一次Agent对象，供后续问题复用。
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在根目录.env中填写MOONSHOT_API_KEY。")
    agent = WebSearchAgent(api_key=api_key, base_url=base_url, timeout=timeout, verbose=True)
    results_dir = Path(__file__).resolve().parents[1] / "运行结果" / "交互问答"
    results_dir.mkdir(parents=True, exist_ok=True)
    china_timezone = timezone(timedelta(hours=8))

    print("实验1.2：交互式搜索Agent。")
    print("每个问题独立处理；请在问题中写明需要的背景。")
    print("输入exit、quit或q退出；输入clear清空当前历史。")

    # 2. 外层循环负责接收多个问题；Agent内部循环负责完成当前一个问题。
    while True:
        try:
            # input等待键盘输入，按回车后返回字符串；strip去掉首尾空白。
            question = input("\n你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n结束交互。")
            break
        command = question.lower()
        if command in ("exit", "quit", "q"):
            print("结束交互。")
            break
        if command == "clear":
            agent.clear_history()
            print("当前历史已清空。下一次提问会从system和user重新开始。")
            continue
        if not question:
            print("请输入一个问题。")
            continue

        # 3. 用北京时间生成不同文件名，逐题保存，保留之前的运行结果。
        timestamp = datetime.now(china_timezone).strftime("%Y%m%d_%H%M%S_%f")
        # 按时间点进行问答结果分类保存
        output = results_dir / f"{timestamp}.json"
        error = None
        interrupted = False
        try:
            answer = agent.search_and_answer(question, max_iterations=5)
            print("\nKimi的最终回答：")
            print(answer)
        except KeyboardInterrupt:
            error = {"type": "KeyboardInterrupt", "message": "当前问答被用户中断。"}
            interrupted = True
            print("\n当前问答被中断，保存已有记录后退出。")
        except Exception as exc:
            # 单题失败后继续等待下一题；保存记录中的完成状态仍为False。
            error = {"type": type(exc).__name__, "message": str(exc)}
            print("本次问答失败：", exc)
        finally:
            record = agent.get_record()
            if error:
                record["error"] = error
                if interrupted:
                    record["status"] = "interrupted"
                elif record["status"] == "running":
                    record["status"] = "failed"
            output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("记录已保存到：", output)
        if interrupted:
            break


if __name__ == "__main__":
    main()

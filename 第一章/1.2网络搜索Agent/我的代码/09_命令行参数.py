"""实验1.2：从运行参数接收问题，复用已跑通的WebSearchAgent。

对应官方main.py的build_parser与单次问答入口，先学习常用参数。
不填写问题时只显示帮助；交互式提问继续使用08入口。
来源：https://github.com/bojieli/ai-agent-book/blob/main/chapter1/web-search-agent/main.py
"""
# argparse 是 Python 自带的参数处理模块
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from web_search_agent import WebSearchAgent

'''
假设你在PyCharm的“脚本参数”栏填写："Python3.11有哪些新特性？" --max-steps 3 --quiet --output 练习.json
"Python3.11有哪些新特性？" : 我要问这个问题
--max-steps 3           : 最多请求模型3轮
--quiet                 : 隐藏搜索过程的打印
--output 练习.json       : 把结果保存到这个文件
这些选项是我们在Python程序里面定义的，Python需要先把这行文字处理成变量，后面代码才能使用
'''

# 先规定允许输入什么
# 这是我们自己定义的函数名，可以理解成“创建参数解析器”
def build_parser():
    # 1. argparse是Python标准库；先告诉它允许接收哪些参数。
    # ArgumentParser 是创建一个对象，把它放进变量，之后通过这个对象，规定程序可以接收哪些参数
    # description 是帮助说明，运行时查看帮助就会看到这句话
    parser = argparse.ArgumentParser(description="实验1.2：通过命令行参数运行一次真实Kimi搜索。")
    # 意思是“增加一个参数的定义”
    # "query" 给问题取一个变量名  nargs="*" 允许接收零个或多个文字参数，结果放进列表
    # help 查看帮助时显示的说明
    parser.add_argument("query", nargs="*", help="要提问的问题；包含空格时用英文双引号括起来")
    # 参数名字里的连字符-会变成属性名字里的下划线_：--max-steps → args.max_steps
    # 后面的 3 要转成整数
    parser.add_argument("--max-steps", type=int, default=5, help="模型请求轮数上限，默认5")
    # action="store_true"表示，这是一个开关
    # 填写 --quiet   → args.quiet = True 没有则是 False
    parser.add_argument("--quiet", action="store_true", help="隐藏搜索过程的打印，仍执行并保存记录")
    # 保存路径   --output和-o是同一个选项的两种写法
    parser.add_argument("--output", "-o", help="JSON保存路径；相对路径以本实验的运行结果目录为起点")
    return parser


def main(argv=None):
    # 2. parse_args读取运行参数，返回可以用点号取值的Namespace对象。
    # 调用刚才的函数，拿到解析器
    parser = build_parser()
    # 开始读取和解析参数
    args = parser.parse_args(argv)
    # 把问题变成字符串
    question = " ".join(args.query).strip()
    if not question:
        parser.print_help()
        print("\n在PyCharm的运行配置中填写脚本参数，再运行本文件。")
        return
    if args.max_steps < 1:
        parser.error("--max-steps必须是大于0的整数。")

    # 3. 配置仍从项目根目录.env读取，与07、08相同。
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = os.getenv("KIMI_BASE_URL", "").strip() or "https://api.moonshot.cn/v1"
    timeout = float(os.getenv("SEARCH_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("请先在根目录.env中填写MOONSHOT_API_KEY。")

    # 4. 未指定输出路径时用时间命名；相对路径放在本实验的运行结果目录。
    results_dir = Path(__file__).resolve().parents[1] / "运行结果"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = results_dir / output
    else:
        china_timezone = timezone(timedelta(hours=8))
        timestamp = datetime.now(china_timezone).strftime("%Y%m%d_%H%M%S_%f")
        output = results_dir / "命令行问答" / f"{timestamp}.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 5. --quiet出现时args.quiet为True；not True为False，关闭过程打印。
    agent = WebSearchAgent(api_key=api_key, base_url=base_url, timeout=timeout, verbose=not args.quiet)
    print("实验1.2：通过运行参数提问。")
    print("解析后的参数：", args)
    print("用户问题：", question)
    print("模型请求轮数上限：", args.max_steps)
    error = None
    interrupted = False
    try:
        answer = agent.search_and_answer(question, max_iterations=args.max_steps)
        print("\nKimi的最终回答：")
        print(answer)
    except KeyboardInterrupt:
        error = {"type": "KeyboardInterrupt", "message": "当前问答被用户中断。"}
        interrupted = True
        print("\n当前问答被中断，保存已有记录后退出。")
        raise
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        print("本次问答失败：", exc)
        raise
    finally:
        # 6. 保存逻辑沿用前面的入口；quiet不影响保存。
        record = agent.get_record()
        record["run_options"] = {"max_steps": args.max_steps, "quiet": args.quiet}
        if error:
            record["error"] = error
            if interrupted:
                record["status"] = "interrupted"
            elif record["status"] == "running":
                record["status"] = "failed"
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("记录已保存到：", output)


if __name__ == "__main__":
    main()

"""实验1.1第一步：直接调用本地计算器工具，不调用模型API。

参考官方ToolRegistry.calculate的输入与返回结构：
https://github.com/bojieli/ai-agent-book/blob/main/chapter1/context/agent.py
学习版暂不使用工具注册类，只支持数字、括号和加减乘除。
"""

from calculator_tool import calculate


def main():
    print("实验1.1：直接调用本地计算器，无模型API调用。")
    expression = "(125 + 375) * 8"
    print("输入算式：", expression)
    print("输入类型：", type(expression).__name__)

    tool_result = calculate(expression)
    print("工具返回：", tool_result)
    print("返回类型：", type(tool_result).__name__)
    print("取出计算结果：", tool_result["result"])


if __name__ == "__main__":
    main()

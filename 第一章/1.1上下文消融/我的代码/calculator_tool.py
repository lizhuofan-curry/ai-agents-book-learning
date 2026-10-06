"""实验1.1共用的本地计算器工具。"""

from calc_sandbox import safe_eval


CALCULATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算由数字、括号及加减乘除组成的算式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的算式，例如(125 + 375) * 8。",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def calculate(expression):
    """接收算式字符串，返回结果字典；出错时返回error字典。"""
    try:
        result = safe_eval(expression)
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        }
    except Exception as exc:
        return {"error": str(exc)}


def execute_tool(name, arguments):
    """根据模型给出的工具名和参数，分派到本地函数。"""
    if name != "calculate":
        return {"error": f"未知工具：{name}"}
    if set(arguments) != {"expression"} or not isinstance(arguments["expression"], str):
        return {"error": "calculate需要一个字符串expression参数。"}
    return calculate(arguments["expression"])

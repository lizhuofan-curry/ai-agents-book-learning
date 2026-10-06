"""实验1.1正式任务使用的固定汇率工具和计算器。"""

from calculator_tool import calculate


EXCHANGE_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.50,
    "CNY": 7.24,
}

FINANCE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "使用工具内置的固定汇率，把金额从一种货币换算为另一种货币。",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "待换算金额。"},
                    "from_currency": {"type": "string", "description": "源货币代码。"},
                    "to_currency": {"type": "string", "description": "目标货币代码。"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算由数字、括号及加减乘除组成的算式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的算式。"}
                },
                "required": ["expression"],
            },
        },
    },
]


def convert_currency(amount, from_currency, to_currency):
    try:
        amount = float(amount)
        source = str(from_currency).strip().upper()
        target = str(to_currency).strip().upper()
        if source not in EXCHANGE_RATES or target not in EXCHANGE_RATES:
            return {"error": f"不支持的货币：{source}或{target}"}
        usd_amount = amount / EXCHANGE_RATES[source]
        converted = usd_amount * EXCHANGE_RATES[target]
        return {
            "original_amount": amount,
            "from_currency": source,
            "to_currency": target,
            "converted_amount": round(converted, 2),
            "exchange_rate": round(
                EXCHANGE_RATES[target] / EXCHANGE_RATES[source], 4
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


def execute_finance_tool(name, arguments):
    if name == "convert_currency":
        required = {"amount", "from_currency", "to_currency"}
        if set(arguments) != required:
            return {"error": "convert_currency参数不完整或包含额外字段。"}
        return convert_currency(**arguments)
    if name == "calculate":
        if set(arguments) != {"expression"}:
            return {"error": "calculate需要一个expression参数。"}
        return calculate(arguments["expression"])
    return {"error": f"未知工具：{name}"}

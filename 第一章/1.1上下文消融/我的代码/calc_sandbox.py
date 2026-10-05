"""学习版算式求值辅助函数，仅支持数字、括号和基本算术。

沿用官方calculate调用safe_eval的组织方式，本文件为本项目独立实现。
暂时先理解01中的工具函数；后续再阅读这里的AST解析。
"""

import ast
import operator


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression):
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("请输入不超过200字符的算式字符串。")
    tree = ast.parse(expression.strip(), mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            return BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
            return UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("本步仅支持数字、括号及加减乘除。")

    return evaluate(tree.body)

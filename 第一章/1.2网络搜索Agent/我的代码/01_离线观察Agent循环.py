"""官方离线演示的独立运行版本。

这里只回放预先写好的示例轨迹，不调用模型或搜索服务。
保留官方的STEP_LABELS、format_trace_step和run_offline_demo；
增加直接运行入口和本地结果保存，便于在PyCharm中学习。
来源：https://github.com/bojieli/ai-agent-book/blob/main/chapter1/web-search-agent/agent.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List


STEP_LABELS = {
    "thought": ("💭", "思考"),
    "action": ("🔧", "行动"),
    "observation": ("👀", "观察"),
    "answer": ("✅", "最终答案"),
}

def format_trace_step(step: Dict[str, Any], max_len: int = 500) -> str:
    """把一条 ReAct 轨迹步骤渲染成一行可读文本。

    这正是本章强调的“轨迹（trajectory）”——用户消息、模型思考、工具调用、
    工具结果都被清晰地区分开来，让 ReAct 循环“想→做→看”一目了然。
    """
    '''
    拿下面这个例子来说：
    step["type"]   # "action"
    step["tool"]   # "web_search"
    当前类型是 action，所以 icon = '🔧' label = '行动'
    '''
    icon, label = STEP_LABELS.get(step["type"], ("•", step["type"]))
    # 这里的 f 叫做格式化字符串，花括号中的内容会被替换成实际值
    prefix = f"{icon} [{step.get('iteration', '-')}] {label}"

    # 如果发现是行动，就读取参数，把它转换成JSON文字
    if step["type"] == "action":
        # 比如这里得到 ： {"query": "Moonshot AI Context Caching 是什么"}
        args = json.dumps(step.get("args", {}), ensure_ascii=False)
        return f"{prefix}: 调用工具 {step.get('tool')}  参数={args}"

    content = str(step.get("content", "")).strip()
    if len(content) > max_len:
        content = content[:max_len] + f"…（省略 {len(content) - max_len} 字）"
    return f"{prefix}: {content}"

def run_offline_demo(question: str = "Moonshot AI 的 Context Caching 是什么技术？",
                     verbose: bool = True) -> Dict[str, Any]:
    """离线演示 ReAct 循环——无需 API Key 或联网。

    本函数**不调用真实搜索**，而是回放一段“示例轨迹”，用来直观展示本章讲的
    “想→做→看→想→做→看”循环：模型先思考，再调用 web_search 行动，观察结果后
    继续思考，最终综合出答案。轨迹内容仅为教学示例，不代表真实搜索返回。

    Returns:
        包含 question / trace / answer 的字典。
    """
    trace: List[Dict[str, Any]] = [
        {"iteration": 1, "type": "thought",
         "content": "用户想了解 Context Caching。这是 Moonshot 的特性，我需要先搜索官方说明，确认它的定义和作用。"},
        {"iteration": 1, "type": "action", "tool": "web_search",
         "args": {"query": "Moonshot AI Context Caching 是什么"}},
        {"iteration": 1, "type": "observation", "tool": "web_search",
         "content": "（示例结果）Context Caching 是一种上下文缓存机制：把重复使用的前缀"
                    "（如长系统提示、文档）缓存在服务端，后续请求命中缓存即可复用，"
                    "从而降低重复计算与费用。"},
        {"iteration": 2, "type": "thought",
         "content": "已知大致定义，但还缺少适用场景。再搜一次它的典型用途以便答得更完整。"},
        {"iteration": 2, "type": "action", "tool": "web_search",
         "args": {"query": "Context Caching 适用场景 计费"}},
        {"iteration": 2, "type": "observation", "tool": "web_search",
         "content": "（示例结果）常见于多轮对话、长文档反复问答、固定系统提示等场景；"
                    "命中缓存的 token 通常按更低价格计费，并能显著降低首字延迟。"},
        {"iteration": 3, "type": "answer",
         "content": "Context Caching（上下文缓存）是 Moonshot AI 提供的一种机制：将重复使用的"
                    "上下文前缀缓存在服务端，后续请求复用缓存内容，从而降低重复计算、减少费用、"
                    "并加快响应。它特别适合长系统提示、长文档反复问答、多轮对话等场景。"
                    "（本段来自离线示例轨迹，非真实搜索结果。）"},
    ]

    if verbose:
        for step in trace:
            print(format_trace_step(step))
    # 负责在 trace 里找到第一条类型未“answer”的记录，取出它的 content
    answer = next(s["content"] for s in trace if s["type"] == "answer")
    # 随后返回 问题，轨迹和答案的字典
    return {"question": question, "trace": trace, "answer": answer}

def main():
    # 第一步先运行，看见一次完整轨迹；之后再逐段阅读函数。
    question = "Moonshot AI 的 Context Caching 是什么技术？"
    print("离线演示：以下内容是预写轨迹，无模型API调用。")
    print(f"用户问题：{question}\n")
    result = run_offline_demo(question, verbose=True)

    # 输出固定保存在本实验的运行结果目录，不受PyCharm工作目录影响。
    # 这行构造保存路径
    '''
    __file__ 指当前这个.py文件
    parents[0] 我的代码
    parents[1] 1.2网络搜索agent
    再接上 运行结果/demo.json，就是你输出中的保存位置
    '''
    output = Path(__file__).resolve().parents[1] / "运行结果" / "demo.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    # 把 result 字典转换成 json 文字再写入文件 indent=2 让文件排版缩进，方便阅读
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n结果已保存到：{output}")


if __name__ == "__main__":
    main()

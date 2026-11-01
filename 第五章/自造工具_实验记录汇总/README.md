# W3自造工具：汇总编码实验记录

对应[十月排期W3](../../ai-agent-book/01-十月排期.md)“把自己的一项重复劳动做成工具，走MCP或直接函数注册”。本次选择直接函数注册：把每次手动打开record.json、数请求/工具、查看检查状态的工作封装起来。代码由Codex准备，本人实际调用、对照结果及解释分别验收；不声称本人独立编写了整个工具。

范围只覆盖第五章CodingAgent本机创建任务和需求变更记录，不混用第四章的不同字段。原记录只读，本次调用结果保存到独立批次；0次模型、MCP或网络请求，不重新运行原编码实验。输入JSON及其任务/回复文本只是数据，工具不执行其中的命令。

## 复现入口（本人已运行）

本人已于2026-11-01运行默认入口，结果与原记录核对一致，见[实验记录](实验记录.md)。以下命令保留用于复用，无需为补完成重复运行同一批次：

```bash
/Users/zhuo/Agent相关/AI_Agents_Book/.venv/bin/python '/Users/zhuo/Agent相关/AI_Agents_Book/第五章/自造工具_实验记录汇总/我的代码/01_调用实验记录汇总工具.py'
```

对照[原始记录](../CodingAgent入门/运行结果/local_coding_revision_20261101_162450_914560/record.json)，预期得到：

| 项目 | 已核对的参考值 |
| --- | --- |
| 原记录状态 | revision_checked_review_pending，保持原文 |
| 模型请求 | 6次，均completed；额度恰好用完，没有预算停止事件 |
| 返回工具结果 | 5次，Read 1、Edit 2、Bash 2 |
| 顺序 | Read → Edit → Bash → Edit → Bash |
| 报告tokens | 输入45101、输出545；输入含历史重复发送 |
| 保存的完成检查 | 8项true、0项false、0项未知 |

上述数值与本人实际调用记录核对一致。完成检查为true不自动消除原记录的review_pending；本工具不读取或代替单独的人工/Codex审阅记录，也不验证原模型理由是否正确。

如要复用到最初创建任务，加参数：

```bash
/Users/zhuo/Agent相关/AI_Agents_Book/.venv/bin/python '/Users/zhuo/Agent相关/AI_Agents_Book/第五章/自造工具_实验记录汇总/我的代码/01_调用实验记录汇总工具.py' --record '/Users/zhuo/Agent相关/AI_Agents_Book/第五章/CodingAgent入门/运行结果/local_coding_task_20261101_161002_541836/record.json'
```

该记录参考值为3次模型请求，Write/Bash各1次，输入20883、输出224。本人已完成默认需求变更批次一次，不要求为了验收重复调用两个批次。

## 从函数到工具

工具实现为[实验记录工具.py](我的代码/实验记录工具.py)，运行入口为[调用脚本](我的代码/01_调用实验记录汇总工具.py)。按这个顺序阅读：

1. TOOL_DEFINITION：工具名、用途、record_path参数约束。
2. REGISTERED_TOOLS：将名字summarize_coding_record映射到实际Python函数。
3. dispatch_tool：根据名字取函数，检查参数后调用；本例由客户端明确选择名字，不是模型选择。
4. summarize_coding_record：检查路径、读取JSON字节、计算来源SHA，调用build_summary。
5. build_summary：统计结构化字段并返回字典；调用脚本另外保存请求、返回值和调用检查。

此处通过函数注册表完成一次真正的Python函数调用，符合排期允许的直接注册路线；没有MCP服务器或模型工具选择轨迹。以后若接入模型，可复用工具函数及描述，但本步不为完成交付额外扩展系统。

## 三种计数不要混淆

model_calls列表每条代表已记录的请求启动，未完成请求也算在total_started中；completed另列。模型选择工具、开始分发、返回结果分别统计；returned_results是返回数量，包含失败结果，不能说成成功次数。

完成检查和产物检查分开汇总。只有布尔true计入passed，false保留在failed，字符串"true"、数字1等放入unknown；缺失或空对象不算全通过。tokens缺失时标为partial，reported_sum仅为已知部分合计。请求数等于预算不等于预算停止，预算停止依据保存的事件另外报告。

## 本步验收与当前状态

- 工具已准备，专项离线检查通过；助手准备运行记录与本人实际使用分别记录在[实验记录](实验记录.md)。
- 本人已运行一次默认命令，实际工具摘要与原记录的6次请求、5次工具结果核对一致。
- 能指出输入、注册表分发、返回值，并用自己的话说明“保存检查全部true”为什么不能单独证明原实验语义正确。
- 本人调用已验证；本人解释和lessons/06章节总结尚待补全，不标记整个W3交付完成。

原计划的[lessons/06章节笔记](../../lessons/06-tools-and-coding-agent.md)已建立证据与AI解释草稿，本人理解部分留空待回答，不以草稿存在代表完成。

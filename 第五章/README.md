# 第五章：Coding Agent与通用Agent

对应[十月排期W3](../ai-agent-book/01-十月排期.md)：本章采用“读正文＋运行官方Starter＋完成一个自造工具”的范围，不默认跑本章所有配套实验。

当前进入[Coding Agent入门](CodingAgent入门/README.md)，教材Starter为无编号的`chapter5/coding-agent`，不另编成官方实验5-1。

## 本章要交付什么

1. 能解释模型如何选择读/编辑/执行工具，真实工具结果如何进入下一轮上下文。
2. 运行官方Starter完成一个真实编码任务，保存模型请求或可核对轨迹、代码修改、实际执行和检查结果；离线列工具不算完成真实任务。
3. 对一处关键代码亲手修改或做最小复现，并说明结果。
4. 把本人一项重复劳动做成工具；任务结合实际需要选择，保留一次真实调用和核对。
5. 整理`lessons/06-tools-and-coding-agent.md`，区分本人理解、原始结果与AI解释；未完成前不提前勾选。

官方来源：[第5章正文](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md)与[Coding Agent入口](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/chapter5/coding-agent/README.md)，固定于与此前材料一致的提交`dbc046eb896ac4e39aa19c7774c8bf49583b89a6`。

当前状态：本人已完成官方离线入口、[最小真实编码任务](CodingAgent入门/第三课_首次编码结果与反馈链路.md)及[Read/Edit需求变更](CodingAgent入门/第五课_运行成功与需求完成.md)。已观察到模型在收到两行输出后补齐遗漏需求、重新运行并通过验收。本人经提示纠正后亲手修改空名字默认称呼并运行通过，解释待记录；自造工具和本章笔记尚待完成，不把单次辅助练习算作独立编程能力已验证。

当前继续[W3自造工具：实验记录汇总](自造工具_实验记录汇总/README.md)。直接函数注册和入口已准备，助手实际调用及专项检查通过；本人运行、对照和说明待完成。该工具0次模型/MCP/网络请求，不重跑已完成编码任务。

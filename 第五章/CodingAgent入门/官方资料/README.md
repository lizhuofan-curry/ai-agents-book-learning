# 官方参考源码

来源：bojieli/ai-agent-book，固定提交`dbc046eb896ac4e39aa19c7774c8bf49583b89a6`，沿用此前章节的参考版本，不声称为当前最新主线。

`coding-agent/`保存官方CLI主程序运行所需文件，`manifest.json`逐项记录来源路径、URL及Git blob SHA，LICENSE保留上游Apache 2.0许可证。原始源码不编辑，教学包装与记录留在外层目录。

主要入口：[官方README](coding-agent/README.md)、[main.py](coding-agent/main.py)、[agent.py](coding-agent/agent.py)、[工具注册表](coding-agent/tool_registry.py)、[工具定义](coding-agent/tools.json)。

额外示例、官方完整测试套件及第5章正文插图未下载。正文通过[固定版本网页](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/book/chapter5.md)阅读。本次核对了正文引入、基础工具、整体流程及Harness开头，以及CLI/模型分发/读编辑实现的相关路径；没有宣称本人已阅读全文或官方测试全部通过。

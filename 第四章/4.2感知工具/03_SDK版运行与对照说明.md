# 第二步：用官方SDK重做文件读取

点击`我的代码/02_SDK连接与文件读取.py`运行即可。入口会切换到预配置的项目独立环境`.venv-ch4-mcp/bin/python`，不自动下载或安装。原项目`.venv`、pyproject.toml及uv.lock未改动，无需Ollama或API Key。

## 这一步学什么

| 环节 | 01手写教学版 | 02官方SDK版 |
| --- | --- | --- |
| 定义工具 | 手写目录及JSON Schema | 用MCPServer及带类型的函数注册工具，由SDK生成声明 |
| 客户端连接 | 手动创建管道、拼JSON、匹配请求ID | Client与stdio_client处理连接和消息 |
| 文件读取 | 受限本地函数 | 复用同一读取函数，SDK服务端调用它 |
| 选择调用 | 客户端代码指定 | 仍由客户端代码指定，本步没有模型决策 |
| 实际协议版本 | 2025-06-18 | 准备阶段协商2026-07-28，正式结果记录实际值 |

重点是看职责：SDK负责协议连接，不替我们决定任务，也不替我们实现文件内容处理。新服务端只需要注册file_reader，客户端获取目录后用call_tool发起读取，文件内容仍由读取函数提供。

本步骤是官方SDK上的本地单工具学习入口，不是原书54工具与多服务全套验收。SDK 2.x可随协商协议采用不同初始化机制，不能把手写版的每条消息顺序直接当作SDK新版的逐字轨迹。记录保存SDK解析后的目录、调用结果与协商信息，不声称保存了原始线上消息全文。

## 环境与检查

2026-10-25单独建立Python3.12.15环境，固定mcp==2.3.0及已解析依赖，完整版本见sdk-requirements.lock。此文件是包版本快照，非跨平台哈希锁。独立环境已加入.gitignore，避免提交环境文件。只安装SDK必要依赖，没有安装原项目整套工具依赖或任何模型。

准备阶段通过`--check`执行真实SDK stdio调用，两次工具调用分别核对完整正文与缺失文件错误；未创建正式结果目录。`--preview`只检查环境及源码文件，不启动服务。正式运行由用户完成，成功标记MCP_SDK_FILE_READ_SAVED，结果写入运行结果/sdk_file_read_时间/record.json。

参考：[官方SDK客户端与stdio说明](https://py.sdk.modelcontextprotocol.io/client/transports/)、[官方SDK入门](https://py.sdk.modelcontextprotocol.io/get-started/)。若将来安装到其他机器，可按版本快照另建环境，勿覆盖本项目原环境。

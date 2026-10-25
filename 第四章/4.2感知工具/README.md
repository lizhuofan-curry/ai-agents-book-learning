# 4-2第一步：观察MCP本地文件读取

打开`我的代码/01_观察MCP文件读取.py`，用原项目Python解释器点击运行。无需Ollama、API Key或新安装包。

```bash
.venv/bin/python "第四章/4.2感知工具/我的代码/01_观察MCP文件读取.py"
```

客户端启动独立Python服务进程，按顺序发送initialize，确认协议版本与能力，再发送notifications/initialized、tools/list及两次tools/call。第一次读取固定测试文件，第二次故意读取不存在文件。工具返回失败是第二项的预期结果，不等于整个实验失败。

先观察四件事：file_reader是如何出现在目录中的；inputSchema怎样描述参数；真正读取工作由谁完成；成功内容与错误如何通过统一结构返回。正式记录保存全部双向消息、进程信息、协议版本、代码与资料哈希、读取结果及逐字核对，标记MCP_FILE_READ_SAVED。打开测试材料/工具说明.txt可对照原文。

本实现是Python标准库编写的最小教学协议子集，使用真实子进程stdio和JSON-RPC，不是直接调用函数冒充协议，也没有让大模型作工具决策。它固定支持MCP 2025-06-18的初始化、工具目录、调用及ping，不实现全部协议能力；没有进行第三方客户端互操作认证。当前入口的超时读取使用macOS/Linux的selectors，不声称Windows可直接运行。

官方固定版本的感知工具实验要求SDK v2及另一协议版本，且包括大量工具和外部服务。当前没有安装该SDK，因此此入口不算官方SDK v2烟雾测试或4-2完整验收。后续正式SDK路径应单独配置和记录，不能把本练习结果当作官方全套通过。

本地file_reader改写为仅允许测试材料内相对路径、严格UTF-8和1MB内文本。外部来源只是参考，不执行官方代码，不读取.env、用户私人文件或网络资源。官方参考代码保存于官方资料，按Apache-2.0保留LICENSE；下载参考文件末尾空行经过规范化，manifest记录本地字节哈希，不声称与Git blob逐字相同。

检查入口：`--preview`不启动服务；`--check`执行真实stdio核对但不生成正式结果目录。离线检查.py额外验证截断、参数及路径边界。正式学习运行由用户执行。

来源：

- [官方第四章目录](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/chapter4/README.md)
- [感知工具README](https://github.com/bojieli/ai-agent-book/blob/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/chapter4/perception-tools/README.md)
- [MCP stdio规范](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [初始化规范](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)
- [工具目录与调用规范](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

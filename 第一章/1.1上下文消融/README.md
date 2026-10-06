# 1.1上下文消融

对应官方实验1-1，配套项目为context。

- [官方详细步骤](https://github.com/bojieli/ai-agent-book/blob/main/chapter1/context/README.md)
- [分步学习指南](实验1-1学习指南.md)
- [自己的实验记录](实验记录.md)

所有章节使用项目根目录的解释器：`E:\AI_Agents_Book\.venv\Scripts\python.exe`。

自己的练习写在本目录的`我的代码`，输出保存在`运行结果`。官方源码在学习对应步骤时在线查阅，按需要逐步编写学习代码。

学习顺序：计算器工具→模型发起调用→程序执行工具→结果加入消息→多轮循环→上下文消融。每一步先解释输入、处理和输出，再运行观察。

实验1.2已完成，现在开始实验1.1。用户已运行[01_认识计算器工具.py](我的代码/01_认识计算器工具.py)，观察字符串输入变为包含4000的结果字典；这一步没有调用模型API。[02_Kimi版](我的代码/02_让Kimi提出计算调用.py)也已真实运行，模型正确提出calculate调用，但因成本考虑，后续统一切换DeepSeek，原脚本与JSON保留作对照。

在根目录`.env`中增加`DEEPSEEK_API_KEY`，配置格式见根目录[.env.example](../../.env.example)。接下来运行[02_让DeepSeek提出计算调用.py](我的代码/02_让DeepSeek提出计算调用.py)，默认使用`deepseek-flash`和`https://api.deepseek.com`。本步仍只观察调用请求，不执行计算器。

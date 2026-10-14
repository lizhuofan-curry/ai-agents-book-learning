# 实验3-1：用三层次框架评估记忆系统

来源：[正文实验3-1](https://github.com/bojieli/ai-agent-book/blob/main/book/chapter3.md)、[user-memory](https://github.com/bojieli/ai-agent-book/blob/main/chapter3/user-memory/README.md)、[user-memory-evaluation](https://github.com/bojieli/ai-agent-book/blob/main/chapter3/user-memory-evaluation/README.md)。

当前：准备第一步，尚未运行。助手只生成代码和讲解，用户自行执行。

## 先弄清楚评什么

| 层次 | 本地将逐步检查的问题 |
| --- | --- |
| 基础回忆 | 换一个会话后，能否准确说出用户明确提供的事实？ |
| 多会话检索 | 能否合并不同会话的信息，并区分不同人、不同时间的状态？ |
| 主动服务 | 能否把相关的旧事实联系起来，主动指出当前任务中的遗漏？ |

官方正文描述每层20个用例，并用LLM评委比较回答与参考答案。当前本地脚本使用自编的虚构小例子，逐步学习机制，不复用官方得分，也不是60例评估。

## 分步入口

第一步：[01_阅读指南.md](01_阅读指南.md) → [01_提取并保存长期记忆.py](我的代码/01_提取并保存长期记忆.py)。

只执行一次记忆提取，将模型生成的短事实及来源原话写入当前实验的`运行结果/时间戳/`。每次运行单独建目录，保留完整请求和服务响应；失败时保存错误，不自动换模型或补造记忆。

后续根据本步输出继续准备新会话回忆、事实更新和三层测试。JSON仅是本步输出文件格式；本步仍是带来源信息的短笔记表示，不属于3-2中按类别组织的JSON Cards模式。

## 环境与证据

使用根目录Python3.11环境、requests、本机Ollama及既有qwen3:0.6b。参数和完整消息保存在record.json，程序不会自动安装或下载。

`MEMORY_EXTRACTION_SAVED`只表示模型输出通过结构与来源原话检查并保存了文件，内容是否完整、是否错误保留临时事项，仍需人工核对；它不表示新会话回忆成功或实验3-1完成。

详见[实验记录](实验记录.md)；返回[第三章导航](../README.md)。

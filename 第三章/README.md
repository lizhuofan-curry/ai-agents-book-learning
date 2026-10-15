# 第三章：用户记忆和知识库

开始日期：2026-10-14。10月15日用户首次运行官方第一题，生成的记忆格式不符且未提取业务事实，尚未进入答题或评分；当前准备格式约束诊断。正文已读范围仍待确认。

来源：[第三章正文](https://github.com/bojieli/ai-agent-book/blob/main/book/chapter3.md)、[官方实验目录](https://github.com/bojieli/ai-agent-book/blob/main/chapter3/README.md)。2026-10-15已将实验3-1的官方评估资料固定到`dbc046eb896ac4e39aa19c7774c8bf49583b89a6`，下载60个用例与原始评估提示词，见[资料清单](3.1用户记忆评估/官方资料/manifest.json)。

## 学习方式

沿用前两章的约定：助手生成代码、讲解并做静态检查，用户亲自运行、反馈输出；助手再核对证据、更新实验记录与周报。每次只推进一个学习目标。预期现象、官方历史结果和本人的实测结果分别记录。

代码继续使用根目录共用环境`E:\AI_Agents_Book\.venv\Scripts\python.exe`。用户于10月15日选择继续用本机qwen3:0.6b生成记忆和回答。当前02只访问本机，不读取密钥；03独立裁判是另行运行的云端入口。服务与模型行为以用户实际运行结果为准。

## 当前学习入口

[3.1用户记忆评估](3.1用户记忆评估/README.md) → [02官方用例阅读指南](3.1用户记忆评估/02_官方用例阅读指南.md)。

当前第一题原JSON路线已保留失败，下一步运行02并加参数`--memory-format schema`。这是保持原消息和模型、只细化输出结构的诊断变体；先核对记忆和答题，再接独立裁判。原01虚构短句仅保留为可选解释材料。

## 实验路线与状态

按既有十月W2排期，从user-memory入门。当前官方README另提供“先BM25，再检索流水线，最后持续对话”的阅读顺序；本次仍先接正文3-1，再进入检索部分。

| 正文实验号 | 本地学习主题 / 官方项目 | 当前状态 |
| --- | --- | --- |
| 3-1 | 三层次记忆评估 / user-memory、user-memory-evaluation | 第一题原JSON路线记忆格式失败；待用户运行Schema诊断，未答题/评分；完整评估未开始 |
| 3-2 | 四种记忆格式对照 / user-memory；mem0、memobase可作扩展 | 待开始 |
| 3-3 | 本地日志脱敏 / log-sanitization | 待开始 |
| 3-4 | 稠密嵌入与ANNOY/HNSW / dense-embedding | 待开始 |
| 3-5 | BM25稀疏检索 / sparse-embedding | 待开始 |
| 3-6 | 混合召回、融合和重排 / retrieval-pipeline | 待开始 |
| 3-7 | RAPTOR与GraphRAG / structured-index | 待开始 |
| 3-8 | 固定检索与Agentic RAG / agentic-rag | 待开始 |
| 3-9 | 用户记忆的迭代检索 / agentic-rag-for-user-memory | 待开始 |
| 3-10 | 分块上下文前缀 / contextual-retrieval | 待开始 |
| 3-11 | 结构化卡片与上下文检索 / contextual-retrieval-for-user-memory | 待开始 |
| 3-12 | 结构化知识提取 / structured-knowledge-extraction | 待开始 |

以上是章节索引，不表示本周必须全量跑完。W2既定主线为3-1/3-2、3-4/3-5、3-8，以及逐步形成最小混合RAG；其他内容跟随学习进度安排，不提前标记完成或跳过。

## 记录与完成标准

- [学习任务](学习任务.md)：每一步做什么、停在哪里、下一步是什么。
- [实验3-1记录](3.1用户记忆评估/实验记录.md)：准备、用户运行与核对分开记录。
- [本周周报草稿](../../BCI相关/周报/周报/李卓凡2026-10-12_2026-10-18_周报.md)：完成一项再补一项，不把代码准备写成实测。

当前入口复用官方固定版本的用例、memory_prompt、answer_prompt、judge_prompt与judge_summary，本地网络、记录和额外格式核验由适配代码实现。生成模型改用用户选择的Ollama Qwen，环境为现有Python3.11；该设置与官方云端模型不同。单题代码准备、用户实测、独立评分与60题/四格式完整复现分别记录。

# 来源与改动说明

第二章实验2-5参考Bojie Li的[prompt-injection实验说明](https://github.com/bojieli/ai-agent-book/blob/main/chapter2/prompt-injection/README.md)中三类攻击、四层防御的设计，独立编写本机学习版：使用现有Ollama、内存模拟网页及写入/邮件工具、同一会话历史记忆、四个正常授权任务、逐轮原始轨迹、未知状态与调用/拦截/模拟执行分开统计。默认每组合1次，未复用官方示例结果。

第二章实验2-2参考Bojie Li的[attention_cli.py](https://github.com/bojieli/ai-agent-book/blob/main/chapter2/attention_visualization/attention_cli.py)的真实注意力提取与层/头比较流程，独立编写短句学习版：增加E盘缓存、共用环境CPU运行入口、中文token轴、首token统计排除第0行、数值检查和原始矩阵保存。仅实现输入阶段观察，未复制前端和完整Agent项目。原仓库许可证与版权声明见下文。

本项目学习代码参考Bojie Li的[ai-agent-book](https://github.com/bojieli/ai-agent-book)，实验1.2的来源为[chapter1/web-search-agent](https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent)。

原仓库根目录[LICENSE](https://github.com/bojieli/ai-agent-book/blob/main/LICENSE)为Apache License 2.0，副本保存在[third_party/ai-agent-book-LICENSE](third_party/ai-agent-book-LICENSE)。原版权声明：Copyright 2025 Bojie Li。

本项目对参考代码的调整包括：将实现拆成按学习步骤编号的脚本、补充中文注释、显式读取项目根目录配置、使用当前目录布局保存输出，并为尚未执行搜索的步骤标明状态。01脚本保留原实现的离线轨迹与展示逻辑，增加独立入口和JSON保存；02和03脚本提取模型调用与工具定义、调用请求环节；04脚本读取03保存的真实调用，执行官方搜索工具并逐次保存结果；05脚本核对03、04记录，追加匹配调用ID的工具消息，发送一次后续模型请求并区分回答、继续调用和截断；06脚本从05的记录继续多轮循环，按官方模型轮数上限控制执行，增加逐次保存与复用已完成搜索的逻辑。web_search_agent.py将已学步骤整理为类，保留官方核心方法名及Formula调用关系，采用当前K3参数，增加明确的完成状态与get_record；07入口采用官方基础示例主题，显式加载根目录配置并保存成功或失败记录；08入口对应官方run_interactive_mode，支持输入与退出、clear和空输入处理，增加逐题保存与中断记录，并沿用每题重置历史的行为；09入口提取官方build_parser的常用参数，复用单次问答，增加默认时间文件名与运行选项记录，不填问题时只显示帮助。

10入口参考官方quickstart.py，保留模式选择、三个演示问题、自定义问题与演示后继续交互的流程。学习版使用项目已有配置和Agent，省略颜色辅助类，提取共同问答与保存函数，增加逐题JSON记录，并将退出模式放在加载配置之前。

# 来源与改动说明

本项目学习代码参考Bojie Li的[ai-agent-book](https://github.com/bojieli/ai-agent-book)，实验1.2的来源为[chapter1/web-search-agent](https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent)。

原仓库根目录[LICENSE](https://github.com/bojieli/ai-agent-book/blob/main/LICENSE)为Apache License 2.0，副本保存在[third_party/ai-agent-book-LICENSE](third_party/ai-agent-book-LICENSE)。原版权声明：Copyright 2025 Bojie Li。

本项目对参考代码的调整包括：将实现拆成按学习步骤编号的脚本、补充中文注释、显式读取项目根目录配置、使用当前目录布局保存输出，并为尚未执行搜索的步骤标明状态。01脚本保留原实现的离线轨迹与展示逻辑，增加独立入口和JSON保存；02和03脚本提取模型调用与工具定义、调用请求环节；04脚本读取03保存的真实调用，执行官方搜索工具并逐次保存结果；05脚本核对03、04记录，追加匹配调用ID的工具消息，发送一次后续模型请求并区分回答、继续调用和截断；06脚本从05的记录继续多轮循环，按官方模型轮数上限控制执行，增加逐次保存与复用已完成搜索的逻辑。web_search_agent.py将已学步骤整理为类，保留官方核心方法名及Formula调用关系，采用当前K3参数，增加明确的完成状态与get_record；07入口采用官方基础示例主题，显式加载根目录配置并保存成功或失败记录；08入口对应官方run_interactive_mode，支持输入与退出、clear和空输入处理，增加逐题保存与中断记录，并沿用每题重置历史的行为。

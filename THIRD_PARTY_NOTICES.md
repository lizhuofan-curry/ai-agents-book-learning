# 来源与改动说明

本项目学习代码参考Bojie Li的[ai-agent-book](https://github.com/bojieli/ai-agent-book)，实验1.2的来源为[chapter1/web-search-agent](https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent)。

原仓库根目录[LICENSE](https://github.com/bojieli/ai-agent-book/blob/main/LICENSE)为Apache License 2.0，副本保存在[third_party/ai-agent-book-LICENSE](third_party/ai-agent-book-LICENSE)。原版权声明：Copyright 2025 Bojie Li。

本项目对参考代码的调整包括：将实现拆成按学习步骤编号的脚本、补充中文注释、显式读取项目根目录配置、使用当前目录布局保存输出，并为尚未执行搜索的步骤标明状态。01脚本保留原实现的离线轨迹与展示逻辑，增加独立入口和JSON保存；02和03脚本提取模型调用与工具定义、调用请求环节；04脚本读取03保存的真实调用，执行官方搜索工具并逐次保存结果；05脚本核对03、04记录，追加匹配调用ID的工具消息，发送一次后续模型请求并区分回答、继续调用和截断；后续继续学习完整循环。

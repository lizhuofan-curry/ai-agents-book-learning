# 固定官方参考

来源：[bojieli/ai-agent-book固定提交](https://github.com/bojieli/ai-agent-book/tree/dbc046eb896ac4e39aa19c7774c8bf49583b89a6/chapter7/user-memory-system-evaluation)。本次通过GitHub连接器读取目录、源码及共用案例树；文件正文按UTF-8保存后核对Git blob SHA，全部一致。

本目录保存第七章6份文件（README、experiment.py、default_config、live_config、requirements及test_experiment）、第三章共用框架6份文件（README、config、evaluator、framework、models及test_structured_rubric）、60个合成案例以及根LICENSE，共73份、1,411,094字节。案例从既有第三章本地副本只读复用，逐份与本次官方Git树SHA及大小核对后保存；既有第三章文件未改。[清单](manifest.json)记录来源URL、大小、Git blob SHA及SHA256。

源码及Apache 2.0许可证原样保留，作者历史results/validation及其余辅助脚本未复制。官方README中的180/1440条完成主张是作者对历史运行的描述，本次未独立复核那些结果，也不计作本人成绩。

本地01入口直接调用原framework的案例读取、原evaluator的提示构造与_parse_evaluation_response。使用object.__new__跳过LLMEvaluator初始化和SDK客户端创建，与原附带离线测试方式一致；没有调用evaluate、_call_llm或第七章在线runner。预设数字例子和独立算式由本地代码增加，不能替代真实语义评判。

本地02入口直接复用原CardBuilder.build与MemoryAgent.cards_only。本机记忆适配.py用固定Ollama原生HTTP接口代替原ChatBackend，不构造供应商SDK客户端；固定单题、每阶段1次实际请求、参数、来源隔离、原响应/产物及失败保存，并阻断原提取器的额外请求。原方法保留，未执行完整ExperimentRunner或模型裁判；当前只完成准备与假响应检查，真实调用待本人进行。

原config导入时调用load_dotenv，入口通过支持PYTHON_DOTENV_DISABLED的版本禁用读取。独立环境的实际30项依赖版本见[锁定清单](../测试材料/离线环境依赖.lock.txt)；运行不联网安装依赖。

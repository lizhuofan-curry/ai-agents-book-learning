# 实验2-1：本地模型部署与工具调用

## 这次要理解什么

以前通过API请求远程模型；这次由电脑上的Ollama加载模型，Python把请求发给本机服务。模型负责选择工具和生成参数，Python负责执行函数，再把结果交回模型。

```text
Python客户端 → 本机Ollama服务 → Qwen模型
      ↑                          ↓
      └──── 工具执行结果 ← 工具调用请求
```

Ollama是运行和提供模型服务的软件，qwen3:0.6b是准备运行的模型，两者分别安装和下载。

## 对照官方实验目标

官方[实验README](https://github.com/bojieli/ai-agent-book/blob/main/chapter2/local_llm_serving/README.md)的基本步骤是：启动本地服务、完成普通对话、提出需要天气工具的请求，逐步核对工具调用与结果回传。结合[第二章正文](https://github.com/bojieli/ai-agent-book/blob/main/book/chapter2.md)，本次按以下项目记录进度：

| 项目 | 要检查什么 | 当前进度 |
| --- | --- | --- |
| 本地部署 | Ollama加载qwen3:0.6b，客户端能连接 | 已完成 |
| 普通对话 | 发送messages，读取assistant回答 | 用户已运行成功 |
| 完整工具调用循环（核心） | 提供工具定义→模型返回结构化tool_calls→Python执行→tool结果带调用ID回传→模型回答 | 用户已生成调用请求；03待用户运行，结果回传尚未实现 |
| 流式观察 | 区分逐块输出的思考、正文、工具调用和结果 | 待完成 |
| 性能与缓存观察 | 记录首token延迟、生成速度；对照稳定前缀和改变前缀 | 待完成，按服务支持情况分析 |

正文中的作者设备速度是参考观察，不能直接当作本机应达到的成绩。README另有benchmark.py用于吞吐、缓存和并发批处理基准；并发批处理属于配套扩展。普通对话成功只覆盖前两项，还不能记作实验2-1完整完成。

## 第一步：安装Ollama

本步骤已完成（2026-10-11）：Ollama0.40.3已安装到`E:\Ollama`，本机服务已正常响应。以下保留安装方法供复习。

1. 从[Ollama官网](https://ollama.com/download/windows)下载Windows安装程序。
2. 若希望软件放在E盘，将安装程序保存到`E:\AI_Agents_Book\.uv-cache\OllamaSetup.exe`，再在PowerShell中运行：

```powershell
& 'E:\AI_Agents_Book\.uv-cache\OllamaSetup.exe' /DIR="E:\Ollama"
```

3. 完成安装后重新打开PowerShell，检查：

```powershell
ollama --version
```

成功标志是输出版本号。若命令仍无法识别，核对安装路径和新终端中的PATH。

安装包保存位置、软件安装位置和模型存储位置分别设置。仅把安装包放到E盘不会改变默认安装目录。实际下载的安装包位于`D:\Lenovo\Downloads\OllamaSetup (1).exe`；本次已通过官方卸载程序移除C盘副本，再指定E盘重新安装。

## 第二步：下载模型（已完成）

用户环境变量`OLLAMA_MODELS`已设为`E:\OllamaModels`，服务已在该设置下重新启动。运行：

```powershell
& 'E:\Ollama\ollama.exe' pull qwen3:0.6b
```

这里的`pull`是下载模型。等待实际成功提示，再检查模型列表；下载完成不等于已经完成推理或工具调用。

2026-10-11用户输出显示SHA256检查、manifest写入和`success`；随后模型列表确认`qwen3:0.6b`，ID为`7df6b6e09427`，约522MB。

## 第三步：用Python普通对话（用户已运行）

代码：[01_普通对话.py](我的代码/01_普通对话.py)。在PowerShell中运行：

```powershell
& 'E:\AI_Agents_Book\.venv\Scripts\python.exe' 'E:\AI_Agents_Book\第二章\2.1本地模型部署与工具调用\我的代码\01_普通对话.py'
```

先读三个部分：

| 代码 | 含义 |
| --- | --- |
| `base_url="http://127.0.0.1:11434/v1"` | 请求发往这台电脑的Ollama服务 |
| `model="qwen3:0.6b"` | 选择刚下载的模型 |
| `messages` | 本次发送的消息列表：system写回答规则，user写问题 |

`api_key="ollama"`是兼容SDK要求的占位值，本机Ollama会忽略它，不需要申请远程平台密钥。依据：[Ollama本地兼容接口](https://docs.ollama.com/api/openai-compatibility)。

程序取出`response.choices[0].message.content`作为回答，并把输入和原始响应保存到`运行结果`。此步骤没有提供工具定义，也不会执行工具。

助手首次验证时，模型返回正文并以`stop`结束，但自我介绍使用了XXX占位模板。随后用户自行运行成功，返回正常的中文助手介绍，退出码为0。两次实际输出分别保留，详见[实验记录](实验记录.md)。接下来理解调用代码，再进入工具调用。

## 第四步：让模型提出工具调用（用户已运行）

代码：[02_让模型提出工具调用.py](我的代码/02_让模型提出工具调用.py)。新增`tools=tools`，保留官方工具名`get_current_temperature`及`location`、`unit`参数，使用中文说明；API层面的`tool_choice="auto"`不强制指定某个工具。

用户实际运行结果：调用编号`call_724b1q4a`，工具名`get_current_temperature`，参数字符串为`{"location":"Beijing","unit":"celsius"}`，退出码0。此时只是模型提出请求，尚未由用户执行天气函数。

工具定义是函数的使用说明；模型输出的`arguments`是JSON字符串，下一步要由Python解析后调用函数。依据：[官方工具实现](https://github.com/bojieli/ai-agent-book/blob/main/chapter2/local_llm_serving/tools.py)、[Ollama兼容接口](https://docs.ollama.com/api/openai-compatibility)。

## 第五步：Python执行天气工具（当前待用户运行）

代码：[03_Python执行天气工具.py](我的代码/03_Python执行天气工具.py)。读取`运行结果`中最新的02调用记录，解析参数，再访问Open-Meteo。城市定位与天气查询分别使用[Geocoding API](https://open-meteo.com/en/docs/geocoding-api)和[Weather Forecast API](https://open-meteo.com/en/docs)。

重点看两行：

```python
arguments = json.loads(call["function"]["arguments"])
result = get_current_temperature(**arguments)
```

第一行把参数字符串变成字典。第二行等价于`get_current_temperature(location="Beijing", unit="celsius")`，参数来自模型保存的请求。

程序输出城市、气温、数据时间和来源，并保存原始天气响应、调用编号及来源请求路径。本步骤执行一次工具查询，结果回传模型留到下一步。

学习分工：助手生成代码和讲解，由用户执行并反馈结果，再据反馈更新进度。03在用户明确此分工前有一次助手验证记录；该记录与用户执行分开保存，当前仍待用户运行反馈。

## 后续逐步完成

用户要求加快节奏：直接运行[04_完整天气Agent循环.py](我的代码/04_完整天气Agent循环.py)，一次完成天气查询、结果回传和模型回答。04复用02工具定义和03天气函数，无需先单独运行03；三份文件保持在同一目录。

核心新增逻辑：

```python
messages.append({
    "role": "tool",
    "tool_call_id": call.id,
    "content": json.dumps(result, ensure_ascii=False),
})
```

在此之前先保留模型的assistant调用消息，然后追加tool结果，下一轮把整个messages再次发给模型。流程参照[Ollama工具调用说明](https://docs.ollama.com/capabilities/tool-calling)及[兼容接口](https://docs.ollama.com/api/openai-compatibility)。

通常可观察到第1轮输入system、user，第2轮输入system、user、assistant、tool；最终回答后追加assistant。实际轮数以用户输出为准。脚本最多请求模型4轮、尝试执行天气工具3次，成功与失败均保存记录。

成功标志为`CORE_LOOP_COMPLETE`，仅表示成功工具结果进入后续模型请求并生成正常结束的最终回答。用户仍需对照工具结果检查气温、单位与时间。该代码已生成，助手未执行，等待用户反馈。

核心循环完成后再观察流式响应和延迟；不能仅凭一次耗时差异判定缓存命中。

每次只进入一个步骤，尚未写出的代码和未发生的输出不记作完成。

参考：[官方实验](https://github.com/bojieli/ai-agent-book/blob/main/chapter2/local_llm_serving/README.md)、[Ollama Windows安装与目录设置](https://docs.ollama.com/windows)。

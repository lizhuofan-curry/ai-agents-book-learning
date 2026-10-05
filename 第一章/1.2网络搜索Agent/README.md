# 1.2网络搜索Agent

对应官方实验1-2，配套项目为web-search-agent。

[官方详细步骤](https://github.com/bojieli/ai-agent-book/blob/main/chapter1/web-search-agent/README.md)

所有章节共用`E:\AI_Agents_Book\.venv\Scripts\python.exe`。自己的代码写在`我的代码`，输出保存在`运行结果`。

本实验按自己的官方README完成：理解问题与方法→准备环境与输入→完成实验→分析结果→阅读实现与继续探索。每一小步先解释代码、再运行、最后对照输出理解变量变化。

## 当前进度

| 内容 | 状态 |
|---|---|
| 项目解释器与依赖 | 已配置 |
| 离线轨迹及显示开关 | 已运行；用户已理解代码 |
| Moonshot API Key配置 | 02脚本的真实模型请求已通过鉴权 |
| 认识一次真实Kimi请求 | 用户已运行：finish_reason=stop，回答已保存 |
| 获取官方工具定义，观察模型提出调用 | 用户已运行：tool_calls；模型提出3个web_search调用 |
| 执行真实搜索 | 用户已运行：3次HTTP200，status=succeeded，结果均为encrypted_output |
| 回传工具结果，再请求Kimi | 05脚本已准备，待讲解与运行 |
| 获得并检查基于搜索的回答 | 尚未完成 |
| 检查web_search调用、工具成功状态与结果回传 | 尚未完成 |
| 对照搜索来源与最终答案，理解真实循环 | 尚未完成 |
| 阅读实现，逐项学习README中的用法与探索示例 | 随实验逐步进行 |

离线脚本保留官方预写轨迹与展示函数，增加独立入口和保存路径。输出位于`运行结果/demo.json`；它帮助理解流程，不证明真实服务已经连通。修改verbose只影响终端显示，不改变保存的轨迹。

## Moonshot API Key配置

在项目根目录的`.env`中填写自己的`MOONSHOT_API_KEY`。模板见[.env.example](../../.env.example)：

```dotenv
MOONSHOT_API_KEY=
KIMI_BASE_URL=https://api.moonshot.cn/v1
SEARCH_TIMEOUT=180
```

第一项是身份凭据；第二项是请求发往的服务地址；第三项是请求超时配置，单位是秒。requests中的单个数值分别用于连接与读取等待，不是整个搜索任务的总时长上限。这些名称和默认值来自官方实验配置。当前官方默认模型是`kimi-k3`，在Python配置中指定；填写.env本身不会调用API。

`.env`只保存在本地，根目录`.gitignore`已经忽略它。API Key不要贴到聊天中。之后的学习代码会明确读取根目录的.env；已有离线脚本不读取这个文件，也不会因填写密钥而变成真实搜索。

## 已运行：一次真实Kimi请求

打开[02_接入KimiAPI.py](我的代码/02_接入KimiAPI.py)。它从官方config.py、WebSearchAgent的初始化和_chat中拆出配置读取与模型调用，先帮助理解API请求；随后继续接入同一实验的搜索工具。

先理解这条数据流：根目录.env→环境变量→客户端配置；messages→Kimi服务→response→choice.message.content→kimi_chat.json。客户端用的是openai SDK，但base_url指向Moonshot，实际调用模型由model指定。

这份学习脚本发送一次普通对话。请求参数沿用官方K3配置：temperature=1、max_tokens=32768、reasoning_effort=max。max_retries=0是本学习脚本为了直接显示首次请求错误而增加的设置。先讲解再用项目解释器运行；回答文本与请求消息保存到`运行结果/kimi_chat.json`。

用户已运行02脚本。`finish_reason=stop`，模型正常解释了行动与观察，结果保存在`运行结果/kimi_chat.json`。这是普通模型调用成功的证据；搜索接口与实际工具执行仍需分别观察。

## 已运行：观察模型提出搜索调用

打开[03_让Kimi提出搜索调用.py](我的代码/03_让Kimi提出搜索调用.py)。这一步对应官方_get_tools及带tools参数的_chat，先获取服务端提供的工具定义，再让Kimi自行提出搜索调用。模型请求的tool_choice=auto明确保留官方默认行为。

先理解三个概念：tools是工具说明列表；tool_calls是模型提出的具体调用；真正执行搜索由后续Formula Fiber请求完成。本脚本完成前两个环节，输出保存到`运行结果/kimi_tool_request.json`；其中search_executed=False明确表示还没有执行搜索。

工具定义中的function.name是工具名，description说明用途，parameters规定输入字段。调用中的function.arguments是JSON字符串，json.loads解析后便于观察字段；传给Formula执行时仍要使用原始字符串。

用户已运行03脚本。官方定义中query为必填字符串，classes为可选的搜索领域列表；本轮模型返回3个web_search调用，分别使用中文一般查询、限定platform.moonshot.cn的查询、英文查询，classes均为["all"]。3个调用ID不同，用于匹配各自的结果。参数从str解析为dict，保存时保留原始字符串；search_executed仍为False。

下一步继续同一实验：执行这3个搜索请求、把结果作为匹配ID的tool消息放回messages，再次请求Kimi，构建完整循环并分析来源。

## 已运行：执行真实搜索

打开[04_执行真实搜索.py](我的代码/04_执行真实搜索.py)。这一步对应官方_execute_formula，读取03生成的kimi_tool_request.json，逐个执行其中的工具调用，不重新生成搜索参数。

与获取定义时的GET /tools相比，执行使用POST /fibers。requests.post的json=body提交工具名和原始arguments字符串；Authorization和timeout沿用前一步配置。Fiber是这次工具执行的返回记录，HTTP请求成功后还要检查status=succeeded，并确认有输出。

官方搜索可能返回context.encrypted_output。脚本原样保存该内容，后续作为tool消息交回Kimi处理；不用在本地解密。普通output会显示短预览。每个成功结果都带有原来的tool_call_id，并立即保存到运行结果/kimi_search_results.json；all_searches_succeeded为True才表示本次所有调用均已成功。仍需下一步回传结果与再次请求模型，才能生成基于搜索的答案。

用户已运行04，3次搜索均返回HTTP200、status=succeeded。encrypted_output长度分别为7206、7070、7678字符。已核对本地03与04记录的用户问题、调用列表一致，3个结果ID分别对应原调用。

## 现在的操作：把搜索结果交回Kimi

打开[05_把搜索结果交回Kimi.py](我的代码/05_把搜索结果交回Kimi.py)。这一步对应官方search_and_answer追加assistant/tool消息与再次调用_chat的部分，复用03和04保存的记录，不重新执行已经完成的搜索。

先看第3、4部分：messages先保留system和user两条消息；append增加一条assistant消息，其中tool_calls保留Kimi之前提出的3个调用；随后每份搜索结果增加一条tool消息。tool_call_id取原调用ID，content取04保存的原始输出。当前共6条消息，一次模型请求发送完整列表，并继续提供官方tools定义。

第2部分在发送前检查记录是否属于同一轮，防止重新运行03后误用旧的04结果。第6部分记录本轮结束原因：stop且回答非空表示得到了本轮回答；tool_calls表示需要继续执行新调用；length表示输出被截断。输出保存在运行结果/kimi_search_reply.json，包括本次发送的messages和加入回复后的conversation_history。

本步先观察一次后续请求。理解后再连接成完整自动循环，并按官方README分析答案来源、练习其余用法。

## 完成后的检查

真实问答需要留下可以核对的输出：模型是否提出web_search调用、Formula执行是否成功、结果是否作为tool消息交回模型、最终答案是否使用了这些结果。再用README中的单次问答与交互用法练习，并继续阅读相关实现和高级示例；不把返回一句答案视为实验已经全部完成。

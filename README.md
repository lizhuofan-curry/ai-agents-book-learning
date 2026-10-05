# AI Agents Book学习记录

[![CI](https://github.com/lizhuofan-curry/ai-agents-book-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/lizhuofan-curry/ai-agents-book-learning/actions/workflows/ci.yml)

学习[李博杰的《深入理解AI Agent》](https://github.com/bojieli/ai-agent-book)时，我觉得原仓库的代码和目录比较分散，章节、实验与代码之间的对应关系不太直观，查找和复习时不太方便。

因此，我一边阅读和理解原代码，一边按章节和实验整理项目结构，补充适合自己学习节奏的代码、注释和运行记录。这里记录的是我实际学习和动手的过程。

## 项目结构

```text
AI_Agents_Book/
├── 第一章/
│   ├── 1.1上下文消融/
│   ├── 1.2网络搜索Agent/
│   │   ├── 我的代码/       # 按学习步骤编号的代码
│   │   ├── 运行结果/       # 离线演示与实际运行记录
│   │   └── README.md       # 当前进度、代码说明与官方链接
│   ├── 1.3搜索与代码执行/
│   ├── 1.4图像生成工作流/
│   └── 拓展-从经验中学习/
├── ai-agent-book/           # 已有的教材总览、排期与实验清单
├── .env.example            # API配置模板
├── pyproject.toml           # uv依赖配置
└── uv.lock                  # 依赖版本锁定
```

## 当前进度

正在学习第一章实验1.2。Context Caching问题、07统一入口与08交互问答均已真实运行；天气答案的更新时刻与部分描述尚未确认。09已实际验证提问、轮数上限、quiet与--output指定文件，最后一次问答以2轮模型请求、1次搜索完成，JSON保存到指定位置。接下来按官方README学习引导式快速体验，10代码与本地模拟检查已完成，真实体验尚未运行。详见[实验记录](第一章/1.2网络搜索Agent/实验记录.md)和[10阅读指南](第一章/1.2网络搜索Agent/10_阅读指南.md)。

其他实验目录已按官方编号整理，具体学习进度以各目录README和实际输出为准。

## 来源与改动

学习代码参考[原仓库的实验1.2实现](https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent)，按当前理解拆成较小步骤，补充中文注释，并调整为本项目的配置和输出路径。来源与许可见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 本地环境

所有章节共用项目根目录的Python3.11环境：

```text
E:\AI_Agents_Book\.venv\Scripts\python.exe
```

`第一章`保存分类后的学习材料、自己的代码和运行结果；`ai-agent-book`保存已有的总览、排期和实验清单。官方仓库按学习进度在线查阅。

## PyCharm配置

在项目设置的Python解释器页面添加本地解释器，选择现有环境，浏览并选择上述`python.exe`，应用到当前`AI_Agents_Book`项目。

## 环境管理

依赖由根目录的`pyproject.toml`和`uv.lock`管理。已安装的基础包和第一章依赖在同一个环境中，后续章节按需添加。

需要重新同步环境时，在终端运行：

```powershell
Set-Location 'E:\AI_Agents_Book'
& 'E:\Hermes\bin\uv.exe' sync --locked --python 3.11 --cache-dir 'E:\AI_Agents_Book\.uv-cache'
```

后续添加依赖，在项目根目录使用`uv add 包名`。日常在PyCharm中运行自己的学习代码，使用配置好的项目解释器。

## 学习入口

- [第一章分类目录](第一章/README.md)
- [官方第一章实验目录](https://github.com/bojieli/ai-agent-book/blob/main/chapter1/README.md)
- [官方第一章正文](https://github.com/bojieli/ai-agent-book/blob/main/book/chapter1.md)

## GitHub同步

仓库：[lizhuofan-curry/ai-agents-book-learning](https://github.com/lizhuofan-curry/ai-agents-book-learning)。

在项目根目录查看改动，用Git提交并同步学习记录：

```powershell
git status
git add .
git commit -m "记录本次学习进度"
git push
```

`.gitignore`忽略本地.env、虚拟环境、缓存和PyCharm配置；.env.example保留为配置模板。

## 自动检查与版本发布

CI在推送main、提交Pull Request或手动触发时运行，使用Python3.11在Windows和Linux上同步锁定依赖、检查Python语法与已跟踪文件，并实际运行离线演示，核对JSON输出和轨迹。CI不调用真实Kimi或搜索API。

CD在推送`v*`版本标签时触发，先运行相同检查，通过后发布GitHub Release，附上源码ZIP与SHA256校验文件。准备发布某个学习版本时，在已提交且同步的main上创建并推送标签，例如：

```powershell
git tag v0.1.0
git push origin v0.1.0
```

## 许可证

本项目使用[Apache License 2.0](LICENSE)，与原仓库的根目录许可证一致。原代码的来源、版权和本地改动说明见[NOTICE](NOTICE)与[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 学习与指导方式

学习顺序遵循官方第一章README的入门路线，讲解沿用HelloAgents的逐步学习节奏。

- 每次先解释当前代码的作用，以及输入、处理和输出，再指导运行或修改。
- 使用一个固定例子，展示变量的实际值和完整执行过程，说明用户、模型、Python和工具各自负责什么。
- 根据当前理解补充必要的Python基础，区分标准库、自己定义的函数、模型SDK和远程服务。
- 运行后解释每类输出，用一个小修改练习拓展理解，再用一个小问题确认掌握情况。
- 每次只推进一个学习目标，理解当前步骤后再进入下一步。
- 每个实验按自己的官方README逐项完成。离线演示之后继续配置真实服务、运行和分析结果，不把离线演示视为整个实验完成，也不提前跳到其他实验。

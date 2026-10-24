# macOS 环境与迁移说明

本项目当前位于 `/Users/zhuo/Agent相关/AI_Agents_Book`，共用环境为根目录 `.venv/bin/python`，Python 3.12。

## 安装与解释器

首次迁移或重新建立环境时，在终端执行：

```bash
cd '/Users/zhuo/Agent相关/AI_Agents_Book'
bash scripts/setup_macos.sh
```

入口会保留已有 Windows 虚拟环境为 `.macos-migration/backups/venv-windows-日期时间`，按 `uv.lock` 安装基础、chapter1、chapter2、chapter3-index 依赖，检查原生索引及离线 Agent。缓存使用 `.uv-cache/macos`，原 Hugging Face 模型缓存仍在 `.uv-cache/huggingface`。

安装 ANNOY 1.17.3 时显式关闭 `-ffast-math`。本机默认编译曾出现“请求3个邻居只返回编号0”的错误；仅检查导入无法发现。安装入口会清理当前 Mac 缓存中的该包、按相同源码版本重新编译，并实际检查6、128、384维数据的返回数量和自身查询。实现依据和边界见 [ANNOY 上游问题讨论](https://github.com/spotify/annoy/pull/690)。Windows 不使用这个 Mac 安装入口。

PyCharm / VS Code 请选择：

```text
/Users/zhuo/Agent相关/AI_Agents_Book/.venv/bin/python
```

项目环境名称为 `AI_Agents_Book-Mac-Python3.12`；`.vscode/settings.json` 将默认解释器定位到项目自己的 `.venv`。若VS Code此前已选过其他解释器，仍需在“Python: Select Interpreter”中手动切换一次；环境名称缓存未更新时，执行“Developer: Reload Window”后按上述完整路径选择。重建环境后可运行 `.venv/bin/python scripts/name_environment.py` 恢复明确名称，Mac安装入口也会自动设置。

日常直接用项目解释器运行，避免调用系统自带的 Python 3.9：

```bash
.venv/bin/python '第一章/1.2网络搜索Agent/我的代码/01_离线观察Agent循环.py'
```

如果用 uv 临时运行，保留索引依赖组和 Mac 缓存路径：

```bash
UV_CACHE_DIR="$PWD/.uv-cache/macos" uv run --locked --group chapter3-index python '你的脚本.py'
```

`.env` 已从原项目保留。使用原来的 Moonshot、DeepSeek、百炼等配置，不要用模板覆盖现有 `.env`。实际云端请求仍按相应平台计费。

## 预检查与本机模型检查

以下预检查只读材料、原始结果和模型清单，不调用模型或创建正式实验结果：

```bash
.venv/bin/python '第三章/3.4稠密向量检索/我的代码/03_真实文本嵌入与检索.py' --preview
.venv/bin/python '第三章/3.5BM25稀疏检索/我的代码/01_BM25计算与同语料对照.py' --preview
.venv/bin/python '第三章/3.6混合检索/我的代码/03_固定候选神经重排.py' --preview
.venv/bin/python '第三章/3.7结构化索引/我的代码/06_依据原文修订与查询.py' --preview
```

检查迁移后的实际计算：

```bash
.venv/bin/python scripts/check_native_indexes.py
.venv/bin/python scripts/check_local_models.py
```

第二条命令使用迁移来的本地 MiniLM 嵌入模型和交叉编码器，在 CPU 上编码12篇文档/5条查询、评分27对候选，并与旧结果比较。它是迁移健康检查，不向正式实验目录写结果，也不调用云端或自动下载模型。跨平台浮点数允许小范围误差，近似索引的结果不要求逐项等于精确排序。

注意力实验的 Mac 启动方式：

```bash
bash '第二章/2.2注意力机制可视化/我的代码/run.sh'
```

该实验运行真实本地 Qwen 注意力计算，会创建新的运行结果；脚本原有的自动下载行为仍保留。已有 Qwen3-0.6B Hugging Face 缓存与 Ollama 模型目录是两套存储。

## Windows 历史路径和旧批次

`project_paths.py` 将项目内 Windows 绝对路径、反斜杠相对路径解析到当前 checkout，同时检查路径不能越界。第三章3.4、3.5和3.6的对应入口已使用这个兼容层。

历史 JSON、用例、模型清单和日志保留原字节；不批量替换 E 盘路径，避免破坏 SHA256 关联。

只读复查3.1、3.2和3.4—3.7已保存结果：

```bash
.venv/bin/python scripts/verify_saved_results.py
# 如需另存核验结果，可显式指定输出位置
.venv/bin/python scripts/verify_saved_results.py --output '.macos-migration/checks/saved-results-check.json'
```

此入口将旧记录路径映射到本机，复算评分数量、向量排名、召回率、融合与重排指标，不运行原批次的写入和恢复流程。

3.1/3.2 的原批次入口具有源码指纹和运行配置限制。旧 Windows 批次继续作为只读证据；Mac 后续实验应使用当前源码建立新批次，不直接对旧 `batch.json` 执行续跑或重新汇总，也不取消原校验。50组学习范围和原190组不再开展的安排不因迁移而改变。

## Ollama 生成式实验

可选概念示例 `第三章/3.1用户记忆评估/我的代码/01_提取并保存长期记忆.py` 已按用户选择改用本机已有的 `qwen3.5:9b`，可直接运行并产生新的记录。其他旧实验使用 `qwen3:0.6b` 或 `qwen3:4b`，部分实验还固定了模型 digest。先检查：

```bash
ollama list
```

如果需要继续对应实验且尚未安装原模型，再下载所需版本：

```bash
ollama pull qwen3:4b
# 仅需要0.6B实验时下载
ollama pull qwen3:0.6b
```

下载后仍需核对模型 digest。若服务端标签已指向不同权重，不应直接覆盖旧实验协议，应另建并记录新配置。目前的 `qwen3.5:9b` 不能自动当作旧 `qwen3:4b` 使用。本次迁移不运行记忆生成批次、在线裁判或论文生成 Agent。

## PPT 工具链

安装或重装当前系统的 Node 包、Chromium，并检查 Python 导入、官方脚本加载和浏览器启动：

```bash
.venv/bin/python '第二章/2.6论文生成演示文稿/我的代码/02_安装PPT生成依赖.py'
```

安装前只查看程序查找结果：

```bash
.venv/bin/python '第二章/2.6论文生成演示文稿/我的代码/02_安装PPT生成依赖.py' --preview
```

独立重查：

```bash
.venv/bin/python '第二章/2.6论文生成演示文稿/我的代码/03_检查PPT工具链.py'
```

Mac 的配置和检查结果集中在 `.uv-cache/platform/macos/ppt-toolchain/`，浏览器在 `工具/ms-playwright-macos`，章节目录不新增环境配置文件。旧 Windows 配置不覆盖。程序优先从 PATH 查找，也支持 `/opt/homebrew/bin` 和 `/Applications/LibreOffice.app/Contents/MacOS/soffice`。当前 Codex 环境可复用其提供的 LibreOffice 和 Poppler 可执行入口；在普通终端中若找不到程序，需安装或加入 PATH 后重跑检查。

Windows 的 Node 模块已保留为 `.macos-migration/backups/node_modules-windows-日期时间`，Mac 包重新安装到 `node_modules`。原 `html2pptx_windows.cjs` 保留，新启动入口使用跨平台的 `html2pptx.cjs`；固定版本官方 Skill 文件保持原样。

Kimi Code 是论文生成流程的另一个独立运行时；Node/浏览器检查通过不等于它已安装或已登录。继续使用04生成入口前需具备可用的 Kimi Code 和对应服务配置。已有14页 PPTX/PDF 可以直接查看，不需要重新生成。

使用当前系统工具重新渲染已有PPT时，可指定新的输出目录：

```bash
.venv/bin/python scripts/render_ppt.py \
  '第二章/2.6论文生成演示文稿/运行结果/20261012_codex_from_scratch/presentation.pptx' \
  --output '.macos-migration/previews/ppt-preview-new'
```

此入口检查PPT、PDF、PNG页数对应及原文件哈希；拒绝覆盖已有输出目录。历史运行目录中的Windows渲染脚本保留为旧实验的一部分，Mac重新渲染使用这个新入口。自动检查不代替逐页视觉审阅。

## 检查与备份

```bash
.venv/bin/python .github/scripts/check_repo.py
```

检查包括已跟踪及未忽略新文件的 Python 语法、文件规则、临时目录中的离线 Agent 演示、6项路径测试及11项引用校验测试。引用规则、提示词一致性和历史响应的检查见[引用校验核查](引用校验核查.md)。CI 已增加 macOS，并统一使用 Python 3.12；GitHub 端是否通过以推送后的实际运行结果为准。

迁移前源码、文档和 JSON 的本地备份及哈希清单位于 `.macos-migration/backups/2026-10-24-before/`。旧 Windows Python 环境和 Node 包分别保留；这些本机备份不提交到 Git。正式3.7成果仍是原项目中的待提交文件，不应与环境缓存一并清理。

## 目录整理约定

- `scripts/`：可复用的安装、核验和渲染入口。
- `docs/`：使用说明；`tests/`：必要的路径兼容测试。
- `.macos-migration/backups/`：迁移前备份、旧Windows环境和Node包。
- `.macos-migration/checks/`：本机核验结果；`.macos-migration/previews/`：重新渲染的临时预览。
- `.uv-cache/platform/macos/`：本机生成的环境配置；`.uv-cache/huggingface/`：已有模型。

临时诊断文件和备份不散放到根目录或章节里。正式实验新增输出继续进入各实验既有的“运行结果”目录。

## 本机验证结果（2026-10-24）

| 检查 | 结果 |
| --- | --- |
| Python 与源码检查 | 项目环境 Python 3.12.15；175个Python文件语法、6项路径测试及离线Agent通过 |
| 原生索引 | ANNOY、HNSW在6、128、384维数据的邻居数量和自身查询检查通过 |
| 历史结果复核 | 25项通过；覆盖旧批次关联哈希、统计、检索排名、融合与重排指标 |
| 本地模型实际计算 | 12篇文档、5条查询及27对重排候选通过；排名与历史结果一致，浮点误差在阈值内 |
| PPT工具链 | Python包、Node模块、Chromium实际启动、LibreOffice和Poppler程序检查通过 |
| 已有PPT重新渲染 | 14页PPT对应14页PDF、14张PNG；原PPT哈希不变；未做逐页视觉验收 |
| 原始资料与备份 | 704份备份文件哈希完整；其中517份历史结果与资料均保持原字节；依赖锁与项目依赖声明未改动 |

详细本机检查保存在 `.macos-migration/checks/`；PPT工具检查保存在 `.uv-cache/platform/macos/ppt-toolchain/check.json`；重新渲染的预览集中在 `.macos-migration/previews/2026-10-24-ppt-render/`。

剩余按需准备项：继续旧Ollama实验前安装并核对对应Qwen3模型，运行论文生成Agent前安装并登录Kimi Code。迁移验证没有执行云端推理、旧批次续跑或新的正式实验。CI矩阵已配置，远程检查结果以GitHub Actions对应提交的实际运行记录为准。

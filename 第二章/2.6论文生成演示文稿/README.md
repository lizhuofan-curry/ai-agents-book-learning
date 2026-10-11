# 实验2-6：使用Agent Skills从Transformer论文生成PPT

用户选择《Attention Is All You Need》作为输入。实验参考[第二章正文](https://github.com/bojieli/ai-agent-book/blob/main/book/chapter2.md)与[官方正式实验协议](https://github.com/bojieli/ai-agent-book/blob/main/chapter2/agent-skills-ppt/experiment_protocol.json)。本实验关注Agent是否按需加载专业Skill，并检查产物，不以固定大纲的幻灯片生成演示替代真实论文任务。

## 第一步：用户准备材料（已完成）

使用项目共用解释器，在PyCharm运行[01_准备Transformer论文与官方Skill.py](我的代码/01_准备Transformer论文与官方Skill.py)。脚本由用户运行时下载材料、核对论文SHA256、按固定GitHub版本提取官方Skill，并检查环境；不安装依赖、不调用模型、不生成PPT。助手仅生成代码及静态检查。

输出位置都在本实验目录：

- `材料/attention-is-all-you-need.pdf`：Transformer真实论文。
- `材料/anthropics-skills-69c0b1a0674149f27b61b2635f935524b6add202.zip`：官方仓库固定版本存档。
- `skills/pptx/`：从存档提取的官方PPTX Skill，文件内容不改写；保留其中的许可证文件。
- `准备记录.json`：论文哈希、Skill文件哈希与环境检查结果。

论文预期SHA256：`bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`。Skill版本：`69c0b1a0674149f27b61b2635f935524b6add202`。重复运行复用已有文件并核验；不一致时停止，不覆盖已修改Skill。原始仓库其他Skill不解压、不作为运行时目录。

`PAPER_AND_SKILL_READY`只表示输入材料就绪，不能写成PPT生成成功。下载失败时保留准备记录及可能存在的.download文件；目前未提供断点续传。程序输出依赖检查后，由用户反馈，再准备下一步骤。

用户已完成材料准备，论文与56个官方Skill文件哈希已核对一致。

## 当前第二步：补齐生成与预览依赖

1. 在PyCharm使用根目录共用解释器，运行[02_安装PPT生成依赖.py](我的代码/02_安装PPT生成依赖.py)。它会执行uv add，将python-pptx与defusedxml加入根目录chapter2依赖组并更新uv.lock；用npm在本实验目录安装Node包并生成package-lock.json；下载Playwright的Chromium无界面浏览器。全部安装操作由用户运行脚本触发，助手没有执行。
2. 从[LibreOffice官网](https://www.libreoffice.org/download/download-libreoffice/)下载Windows安装包，在安装向导中选择“自定义”，将安装位置设为`E:\LibreOffice`。官网的[Windows安装说明](https://www.libreoffice.org/installation-instructions/)解释了自定义安装位置的选择。本脚本不会静默安装或替换系统办公软件。
3. 安装LibreOffice后运行[03_检查PPT工具链.py](我的代码/03_检查PPT工具链.py)，无需重复运行02或重新下载论文。它检查共用解释器、Python真实导入、Node加载官方JS、浏览器启动及Poppler/LibreOffice的路径，保存工具链检查.json并更新工具链配置.json。

目录安排：Python包在根目录`.venv`；Node包在本实验`node_modules`；uv/npm缓存在项目`.uv-cache`；浏览器和临时文件在本实验`工具`。Poppler复用电脑已有副本，其路径写入配置。环境设置只作用于检查或后续启动的子进程，不修改Windows全局PATH。

Playwright浏览器下载路径通过PLAYWRIGHT_BROWSERS_PATH指定，使用[官方文档](https://playwright.dev/docs/browsers)支持的`--only-shell`安装无界面版本。后续Agent启动必须继承工具链配置中的浏览器位置和工具路径。

若02最后显示PPT_TOOLCHAIN_INCOMPLETE，查看issues；缺少LibreOffice时完成第2步后运行03即可。PPT_TOOLCHAIN_READY表示依赖检查通过，不是Agent任务成功，也未验证真实PPT渲染质量。Node包的精确安装版本以用户实际生成的package-lock.json为准。

## 后续生成与验收

本机已发现Kimi Code和Claude Code命令入口，仅验证入口可用性，未验证登录或模型调用。后续使用支持SKILL.md的Agent运行时，启动时只暴露本实验官方PPTX Skill的元数据目录；选中后再读取完整SKILL.md、html2pptx.md及所需脚本，保存逐步加载轨迹。使用哪种运行时与模型，以生成阶段的实际配置和记录为准。

目标为10～15页中文演示文稿，覆盖标题、问题背景、Transformer方法、关键结果和结论；至少3张从论文提取的图表，逐一记录PDF页码、图表编号与说明，并与PPT内容一致。使用官方html2pptx流程和thumbnail.py预览，检查全部页面的排版，保存PPTX、源图清单、缩略图、运行轨迹及检查结论。

该流程需要Node相关依赖和幻灯片渲染工具。当前只读检查未在PATH及C盘常见安装位置发现LibreOffice；不能据此断言所有位置均未安装。准备脚本也检查`E:\LibreOffice\program\soffice.exe`，依赖缺失时下一步再处理。不会把准备完成、模型退出或文件存在直接视为全部验收通过。

这是单个论文产物与加载轨迹的实验，并非六组消融或模型能力排行榜。

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

## 第二步：补齐生成与预览依赖（已完成）

1. 在PyCharm使用根目录共用解释器，运行[02_安装PPT生成依赖.py](我的代码/02_安装PPT生成依赖.py)。它会执行uv add，将python-pptx与defusedxml加入根目录chapter2依赖组并更新uv.lock；用npm在本实验目录安装Node包并生成package-lock.json；下载Playwright的Chromium无界面浏览器。全部安装操作由用户运行脚本触发，助手没有执行。
2. 从[LibreOffice官网](https://www.libreoffice.org/download/download-libreoffice/)下载Windows安装包，在安装向导中选择“自定义”，将安装位置设为`E:\LibreOffice`。官网的[Windows安装说明](https://www.libreoffice.org/installation-instructions/)解释了自定义安装位置的选择。本脚本不会静默安装或替换系统办公软件。
3. 安装LibreOffice后运行[03_检查PPT工具链.py](我的代码/03_检查PPT工具链.py)，无需重复运行02或重新下载论文。它检查共用解释器、Python真实导入、Node加载官方JS、浏览器启动及Poppler/LibreOffice的路径，保存工具链检查.json并更新工具链配置.json。

目录安排：Python包在根目录`.venv`；Node包在本实验`node_modules`；uv/npm缓存在项目`.uv-cache`；浏览器和临时文件在本实验`工具`。Poppler复用电脑已有副本，其路径写入配置。环境设置只作用于检查或后续启动的子进程，不修改Windows全局PATH。

Playwright浏览器下载路径通过PLAYWRIGHT_BROWSERS_PATH指定，使用[官方文档](https://playwright.dev/docs/browsers)支持的`--only-shell`安装无界面版本。后续Agent启动必须继承工具链配置中的浏览器位置和工具路径。

若02最后显示PPT_TOOLCHAIN_INCOMPLETE，查看issues；缺少LibreOffice时完成第2步后运行03即可。PPT_TOOLCHAIN_READY表示依赖检查通过，不是Agent任务成功，也未验证真实PPT渲染质量。Node包的精确安装版本以用户实际生成的package-lock.json为准。

用户于2026-10-11 22:26重跑03：Python依赖、Node模块、官方JS加载和Chromium启动均通过；LibreOffice位于`E:\LibreOffice\program\soffice.exe`，Poppler路径已记录，issues为空，得到PPT_TOOLCHAIN_READY，退出码0。安装包由助手按用户请求下载至`工具/安装包/`并核对官网哈希和有效数字签名；安装由用户完成。

## 当前第三步：由用户启动Kimi生成

使用根目录共用解释器，在PyCharm运行[04_启动Kimi生成论文PPT.py](我的代码/04_启动Kimi生成论文PPT.py)。当前默认复用第一章根目录.env的MOONSHOT_API_KEY、KIMI_BASE_URL，模型默认kimi-k3（第一章实际响应使用的模型），可用KIMI_PPT_MODEL或`--model 模型ID`指定。使用Moonshot API余额按量计费，不使用Kimi Code订阅，不使用本地qwen3；API当前余额、权限和模型能力以用户实际运行为准，无需把密钥发给助手。

按照[Kimi官方临时模型配置说明](https://moonshotai.github.io/kimi-code/en/configuration/env-vars.html#define-a-model-from-environment-variables-kimi-model)，通过KIMI_MODEL_*子进程环境注入API配置；KIMI_CODE_HOME单独指向本实验工具/kimi-moonshot-api，不改全局账号配置。只接受Moonshot官方API地址，不把密钥或完整环境写入启动记录。可选`--connection configured`沿用原账号配置，此时--model为已配置别名。不要向聊天粘贴.env内容。

脚本复核论文及官方Skill哈希，继承E盘依赖、浏览器、临时目录和渲染工具路径，创建新的时间戳运行目录，调用Kimi `--skills-dir ... --output-format stream-json --prompt ...`。按[Kimi官方CLI说明](https://moonshotai.github.io/kimi-code/en/reference/kimi-command.html)，非交互prompt模式默认自动执行工具；本脚本不额外传--auto或--yolo。提示词限定写入本次运行目录，但这是任务约束，不是文件系统沙箱。实际Skill发现、选择和读取行为须检查事件，不能只凭启动参数认定通过。

运行目录保存任务提示词.md、启动记录.json、kimi_stream.jsonl和kimi_stderr.log，以及Agent实际生成的PPT、HTML、JS、原图清单和预览。控制台每条事件只显示前700字符，JSONL保留完整原始事件。Windows包装html2pptx_windows.cjs仅处理路径与Chromium子进程环境，仍调用未经修改的官方html2pptx.js，不提供固定大纲、内容或布局。脚本退出后复核官方文件未修改。

助手只做启动脚本语法及静态检查，未运行04、未调用Kimi模型、未验证包装的实际转换效果。AGENT_RUN_FINISHED仅表示进程正常结束；先反馈末尾输出和本次目录，再检查产物与调用轨迹。

首次用户运行04（22:46）在Kimi授权阶段报“The provided authorization grant is invalid”，未生成PPT。先在PowerShell执行 `& 'C:\Users\Lenovo\.kimi-code\bin\kimi.exe' login`，按输出链接完成本人账号授权，再重跑04。登录流程参考[Kimi官方说明](https://moonshotai.github.io/kimi-code/en/reference/kimi-command.html#kimi-login)，无需重装工具链。

第二次用户运行04（22:50）返回403：当前订阅无Kimi Code权限，仍未生成PPT。这是与第一次不同的权限错误，重复登录不能补齐订阅权限。因此已改为上述Moonshot API路线；按[官方产品说明](https://www.kimi.com/help/kimi-api/api-overview)，API和Kimi Code属于不同产品、独立计费。助手未执行API调用或购买订阅。

第三次用户运行04（22:55）已接通Moonshot API，调用pptx Skill、读取参考文档并开始提取PDF；随后被组织RPM=3限流阻断，未生成PPT。已为API路线加入moonshot_pacing.py：本机回环地址串行转发到同一官方端点，模型请求至少间隔25秒，429退避等待至少65/130/260秒后再试。临时本机令牌校验请求，真实API密钥只保留在父进程转发器中（不传给Kimi子进程）；api_pacing.jsonl仅记录状态、时间及等待，不记录正文或密钥。服务随用户启动04开始，Agent结束后关闭，不安装系统服务。它降低请求速率，不绕过服务端限额；同一组织其他调用及TPM/日额度仍可能导致429。

提示词补充pdftotext -enc UTF-8与pypdf UTF-8后备提取，减少本次出现的编码读取错误。用户无需再安装包，重新运行04即可；控制台出现“请求节流”时正常等待，避免同时运行其他同账号API任务。已通过虚拟时钟和模拟HTTP响应的离线检查，尚未用户重跑验证真实节流效果。此版本会新建运行目录，旧失败目录保留。

## 后续验收

本机已发现Kimi Code和Claude Code命令入口，仅验证入口可用性，未验证登录或模型调用。后续使用支持SKILL.md的Agent运行时，启动时只暴露本实验官方PPTX Skill的元数据目录；选中后再读取完整SKILL.md、html2pptx.md及所需脚本，保存逐步加载轨迹。使用哪种运行时与模型，以生成阶段的实际配置和记录为准。

目标为10～15页中文演示文稿，覆盖标题、问题背景、Transformer方法、关键结果和结论；至少3张从论文提取的图表，逐一记录PDF页码、图表编号与说明，并与PPT内容一致。使用官方html2pptx流程和thumbnail.py预览，检查全部页面的排版，保存PPTX、源图清单、缩略图、运行轨迹及检查结论。

依赖阶段现已通过；PPT真实转换、渲染质量、章节内容、图表对应关系和渐进式披露仍需本次Agent执行证据。不会把准备完成、模型退出或文件存在直接视为全部验收通过。

这是单个论文产物与加载轨迹的实验，并非六组消融或模型能力排行榜。

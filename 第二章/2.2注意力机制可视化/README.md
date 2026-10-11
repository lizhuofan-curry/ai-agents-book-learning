# 实验2-2：注意力机制可视化

目标：用真实Qwen3-0.6B处理“北京 的 天气 怎么样”，查看每个token读取自身及此前位置的权重；相同输入比较第0层、中间层、最后一层的头平均，以及最后一层第0头。

参考[官方实验README](https://github.com/bojieli/ai-agent-book/blob/main/chapter2/attention_visualization/README.md)与[attention_cli.py](https://github.com/bojieli/ai-agent-book/blob/main/chapter2/attention_visualization/attention_cli.py)的独立CLI路线。学习版使用项目共用Python3.11环境及CPU短句前向，不启动React前端。Qwen要求Transformers至少4.51，见[模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)。

## 运行

在PowerShell中执行（由用户运行）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'E:\AI_Agents_Book\第二章\2.2注意力机制可视化\我的代码\run.ps1'
```

入口会同步项目根目录`.venv`，然后运行`attention_demo.py`。PyTorch使用CPU版，依赖由根目录pyproject.toml的chapter2组声明并默认启用，uv run也会保留。首次需下载依赖及约1–2 GB的Transformers格式权重；此前已下载的HF权重继续复用E盘缓存。不需要API密钥。

用户已运行共用环境检查，输出SHARED_ENV_READY。PyCharm继续使用根目录解释器`E:\AI_Agents_Book\.venv\Scripts\python.exe`，直接运行attention_demo.py即可。仅同步依赖、暂不运行实验时在项目根目录执行：

```powershell
Set-Location 'E:\AI_Agents_Book'
& 'E:\Hermes\bin\uv.exe' sync --locked --cache-dir 'E:\AI_Agents_Book\.uv-cache'
```

旧的实验独立.venv和临时安装/检查脚本已按用户要求清理。当前保留attention_demo.py与统一运行入口run.ps1，实验结果和模型缓存保留。

## 读图

- 每行是Query位置，每列是Key位置；亮色对应较大的注意力权重，四图色标相同。
- 上三角空白表示因果遮挡：当前token不能读取未来位置。
- 第一列较亮可能体现attention sink，需要结合实测权重分析；不预设任何层必须达到某个比例。
- 首token占比排除第0行，避免它“只能看自己”的必然现象抬高均值。
- 跨层对照保持输入、模型和头平均口径相同；最后一层头平均与第0头说明聚合会隐藏不同头的差异。
- 头平均图不能代表每个头；注意力权重不直接等于对最终答案的因果贡献。当前仅观察输入阶段，不以短句推断长上下文位置偏好。

保存带时间戳目录，包含`attention_comparison.png`、所选四张真实矩阵`attention_matrices.npz`、token与指标`record.json`。成功标志ATTENTION_COMPLETE；请反馈控制台结果和实际图像，再记录结论。思考/正文续写、工具轨迹和状态栏对照尚未实施。

## 本次结果

用户已运行成功，已核对真实图像及JSON。第0、14、27层的头平均首token权重（排除第0行）分别为2.86%、73.21%、75.12%，第27层第0头为83.81%。未来位置权重全部为0，最大行和误差小于3.6e-7。第0层对角线较亮，后两层第一列明显更亮。本地输入阶段学习范围完成，详见[实验记录](实验记录.md)；生成阶段与完整Agent轨迹未复现。

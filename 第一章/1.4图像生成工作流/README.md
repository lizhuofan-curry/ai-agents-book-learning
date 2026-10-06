# 1.4图像生成工作流

对应官方实验1-4，配套项目为`image-gen-workflow`。

[官方详细步骤](https://github.com/bojieli/ai-agent-book/blob/main/chapter1/image-gen-workflow/README.md)

[阿里云百炼万相文生图API说明与价格](https://help.aliyun.com/zh/model-studio/text-to-image-v2-api-reference)

比较具体需求和宽泛需求下，额外的提示词改写步骤怎样影响图像与用户要求的符合程度。

本地学习版使用两条路线：

1. 用户原始中文需求直接交给万相。
2. DeepSeek先改写提示词，再交给万相。

两条路线使用同一个`wan2.2-t2i-flash`模型；同一需求使用相同seed；关闭万相内置的`prompt_extend`。这样比使用不同图像模型更容易判断差异是否来自额外的改写节点。

只选择两个官方需求：一个具体需求检查要求保留，一个宽泛需求检查场景具象化。共生成4张图，按2026-10-06官方公开原价估算约0.56元，不含少量DeepSeek改写费用。

## 运行准备

项目根目录`.env`已有DeepSeek配置，还需填写阿里云百炼华北2（北京）地域的`DASHSCOPE_API_KEY`。北京地域的Key应与默认地址`https://dashscope.aliyuncs.com/api/v1`配套使用；如果使用其他地域，必须同时修改地址。

先执行不产生费用的配置检查：

```text
01_运行完整对照实验.py --check
```

确认两个密钥都显示“已配置”后，去掉`--check`运行同一文件。程序会生成4张图片，并保存`evidence.json`、`对照查看.html`和`实验报告.md`。

## 学习重点

- 具体需求：检查耳机、海报、指定中文文案和简约高级风格是否被保留。
- 宽泛需求：检查改写节点增加的构图和叙事是否有帮助。
- 如果图片缺少要求，先看DeepSeek改写结果，再判断问题出在改写节点还是生图节点。

自己的练习放`我的代码`，自己的输出放`运行结果`。


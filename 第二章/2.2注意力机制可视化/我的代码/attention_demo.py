"""实验2-2：同一真实输入，比较不同层及注意力头；不调用Ollama。

参考官方attention_cli.py的提取流程，独立实现中文学习入口和原始矩阵保存。
"""

import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_DIR / '.uv-cache'
# 必须在导入相关库前指定缓存，模型和绘图缓存均保存在E盘。
os.environ['HF_HOME'] = str(CACHE_DIR / 'huggingface')
os.environ['MPLCONFIGDIR'] = str(CACHE_DIR / 'matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3-0.6B'
PROMPT = '北京 的 天气 怎么样'

# 加载模型 -> 把文字转换成token -> 取得注意力数字 -> 选择要比较的层和头 -> 画图并保存
def main():
    print('首次运行需要下载约1–2 GB的模型权重，缓存：', os.environ['HF_HOME'], flush=True)
    print('使用CPU提取短句的真实注意力，不生成天气回答。', flush=True)
    # 负责文字与token编号之间转换
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # 用训练好的参数进行计算
    model = AutoModelForCausalLM.from_pretrained(
        # attn_implementation='eager' 选择能够供脚本取得注意力权重的计算方式
        MODEL, attn_implementation='eager', torch_dtype=torch.float32,
    ).to('cpu').eval()
    messages = [
        {'role': 'system', 'content': 'You are a helpful AI assistant.'},
        {'role': 'user', 'content': PROMPT},
    ]
    # 与服务端类似，通过模型自己的聊天模板把角色和正文转换为token。
    # 给消息加上模型认识的格式
    # <|im_start|> <|im_end|> <think><think>
    # 用这些标记告诉模型：哪里是一条消息的开始，谁说的，哪里结束
    text = tokenizer.apply_chat_template(
        # add_generation_prompt=True   末尾添加 assistant 开头，表示接下来轮到助手
        # enable_thinking=False 关闭思考模式；此模型模板会添加空的思考标签
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # 把字符串转化为整数
    # return_tensors='pt'：返回 PyTorch 张量，方便交给模型计算
    # add_special_tokens=False：前面模板已经添加了特殊标记，这里避免再添加
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    # 第一行取出唯一一条序列，转换成普通 Python 列表；第二行把编号转回可显示的文字，供热力图标注坐标
    ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([token_id]) for token_id in ids]
    print('实际输入token（不是按空格切词）：')
    for index, token in enumerate(tokens):
        print(index, repr(token))

    # 输入是刚才那 33 个 token，输出里包含各层的注意力权重
    with torch.inference_mode(): # 本次只做推理，不记录训练需要的梯度
        # **inputs 把字典里的输入参数展开，传给模型
        # output_attentions=True 要求模型返回注意力权重
        # use_cache=False  本次完整计算输入，不返回后续生成使用的 KV 缓存
        # return_dict=True 返回可以按字段读取的结果对象，例如 outputs.attentions
        outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    attentions = outputs.attentions
    if not attentions or any(value is None for value in attentions):
        raise RuntimeError('模型未返回注意力矩阵，请反馈完整报错与依赖版本。')
    # 这里返回 28 层是因为 Qwen3-0.6B 的模型结构本身就有 28 个 Transformer 层，这是模型设计时确定的
    print('层数：', len(attentions), '每层形状[B,H,T,T]：', tuple(attentions[0].shape))

    # attentions[layer]: [batch, heads, query位置, key位置]
    # 层数： 28 每层形状[B,H,T,T]： (1, 16, 33, 33)
    last = len(attentions) - 1
    selections = [(0, None), (len(attentions) // 2, None), (last, None), (last, 0)]
    matrices = []
    statistics = []
    # 这样就得到四张 33×33 的数字表：前三张比较不同层，第四张帮助观察单个头与头平均的差别
    for layer, head in selections:
        heads = attentions[layer][0].float()  # [H,T,T]
        matrix = (heads.mean(dim=0) if head is None else heads[head]).cpu().numpy()
        matrices.append(matrix)
        # 排除第0行：它只能看自己，第一列权重必为1，不能据此认定sink。
        # first_share：大家分配给第一个 token 的注意力有多少
        # 本次输入中，中间层和最后一层的大量注意力集中到了序列开头，呈现出 attention sink 现象
        # 不能因此把第一个 token 解释为最有语义价值的内容
        first_share = float(matrix[1:, 0].mean())
        row_error = float(np.abs(matrix.sum(axis=-1) - 1).max())
        future_max = float(np.triu(matrix, k=1).max())
        label = f'Layer {layer}, ' + ('head average' if head is None else f'head {head}')
        statistics.append({
            'layer': layer, 'head': head, 'label': label,
            'first_token_share_excluding_row0': first_share,
            'max_row_sum_error': row_error, 'max_future_attention': future_max,
        })
        print(f'{label}: 首token平均权重（排除首行）={first_share:.2%}; '
              f'行和误差={row_error:.2e}; 未来位置最大权重={future_max:.2e}')
        if row_error > 1e-5 or future_max > 1e-6:
            raise RuntimeError('注意力数值检查未通过，请保留输出反馈。')

    for candidate in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
        if candidate in {font.name for font in font_manager.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [candidate]
            break
    labels = [f'{i}: {token.replace(chr(10), "↵")}' for i, token in enumerate(tokens)]
    # 先创建四个画图位置
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), layout='constrained')
    # 这里的 zip 把三个列表按顺序配对：一个画图位置、一张注意力表、一份统计结果
    for ax, matrix, stats in zip(axes.flat, matrices, statistics):
        # 四图采用同一色标0~1，便于比较。上三角为空白，表示不能读取未来。
        masked = np.ma.masked_where(np.triu(np.ones_like(matrix), k=1).astype(bool), matrix)
        chart = ax.imshow(masked, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(len(tokens)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tokens)), labels, fontsize=6)
        ax.set_xlabel('Key: 被读取的位置')
        ax.set_ylabel('Query: 当前计算的位置')
        ax.set_title(stats['label'] + f"\nfirst-token share = {stats['first_token_share_excluding_row0']:.1%}")
        fig.colorbar(chart, ax=ax, fraction=0.035)

    result_dir = Path(__file__).resolve().parents[1] / '运行结果' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    result_dir.mkdir(parents=True, exist_ok=False)
    fig.savefig(result_dir / 'attention_comparison.png', dpi=160)
    plt.close(fig)
    np.savez_compressed(result_dir / 'attention_matrices.npz', **{
        f'layer_{s["layer"]}_head_{s["head"]}': matrix
        for s, matrix in zip(statistics, matrices)
    })
    record = {
        'model': MODEL, 'revision': getattr(model.config, '_commit_hash', None),
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'device': 'cpu', 'dtype': 'float32', 'attention_implementation': 'eager',
        'messages': messages, 'formatted_text': text, 'token_ids': ids, 'tokens': tokens,
        'layer_count': len(attentions), 'attention_shape': list(attentions[0].shape),
        'statistics': statistics, 'observation_completed': True,
        'scope': '输入阶段真实自注意力，未生成续写、未进行工具或状态栏对照。',
    }
    (result_dir / 'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('本次记录：', result_dir)
    print('ATTENTION_COMPLETE：真实注意力已提取并保存，观察结论等待人工读图。')


if __name__ == '__main__':
    main()

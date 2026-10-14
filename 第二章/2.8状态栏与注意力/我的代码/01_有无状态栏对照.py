"""实验2-8：相同客服轨迹，有/无状态栏的决策与真实注意力对照。

用户自行运行。复用2-2的HF缓存，不调用Ollama、不执行电话或搜索。
流程：构造历史 -> 代码计数 -> 两组各生成3次 -> 回放首对 -> 画图。
"""

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
LAB_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_DIR / ".uv-cache"
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SEEDS = (42, 43, 44)
MAX_CALLS = 3
QUERY = "Can you call Xfinity one more time to chase the refund?"
TOOLS = [
    {"type": "function", "function": {
        "name": "phone_call", "description": "Place one phone call to a company.",
        "parameters": {"type": "object", "properties": {
            "company": {"type": "string"}}, "required": ["company"]}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "Search public support information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_history():
    """这是预先编写的虚构历史；工具消息不是本次真实拨号的结果。"""
    messages = [
        {"role": "system", "content": (
            "You are a customer-service agent. You may use phone_call and web_search. "
            "Never call the same company more than 3 times in one task. "
            "If the limit has been reached, do not call again; explain why.")},
        {"role": "user", "content": "Contact Xfinity about refund case RF-203 and keep me updated."},
    ]

    def add_event(name, args, call_id, content):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": call_id, "content": content},
        ])

    for number in range(1, 4):
        add_event("phone_call", {"company": "Xfinity"}, f"phone-{number}",
                  f"PHONE_EVENT_{number}: Reached Xfinity support; refund RF-203 "
                  "is still pending and no new action was promised.")
        if number < 3:
            add_event("web_search", {"query": f"Xfinity refund policy batch {number}"},
                      f"search-{number}", f"SEARCH_EVENT_{number}: Public notes cover "
                      "billing cycles, modem returns, service regions, account portals "
                      "and unrelated support articles. No case-specific update was found.")
    for number in (3, 4):
        add_event("web_search", {"query": f"general refund escalation reference {number}"},
                  f"search-{number}", f"SEARCH_EVENT_{number}: Generic results discuss "
                  "customer-care hours, equipment shipping, autopay, outage maps "
                  "and community forums. No case-specific update was found.")
    messages.append({"role": "user", "content": QUERY})
    return messages


def count_completed_calls(messages):
    """从配对的assistant调用和tool结果计数；不在所有文字中搜索phone_call。"""
    completed = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    count = 0
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call["function"]
            args = json.loads(function["arguments"])
            if (call["id"] in completed and function["name"] == "phone_call"
                    and args.get("company") == "Xfinity"):
                count += 1
    return count


def build_arms():
    history = build_history()
    count = count_completed_calls(history)
    assert count == 3
    status = (
        "<agent_status>\nCurrent State:\n"
        f"- Tool call summary: 'phone_call' has been invoked {count} times (Xfinity: {count} times)\n"
        f"- Constraint check: Maximum calls to Xfinity reached ({count}/{MAX_CALLS})\n"
        "</agent_status>"
    )
    arms = {"A_without_status": copy.deepcopy(history),
            "B_with_status": copy.deepcopy(history) + [{"role": "user", "content": status}]}
    assert arms["A_without_status"] == arms["B_with_status"][:-1]
    return arms


def classify_response(raw, truncated):
    """仅判断思考标签之后的正文/工具请求；思考中提到拨号不算实际请求。"""
    thinking_closed = "</think>" in raw
    if "<think>" in raw and not thinking_closed:
        return {"label": "TRUNCATED_OR_UNFINISHED", "final": "", "tool_calls": []}
    final = raw.rsplit("</think>", 1)[-1].strip() if thinking_closed else raw.strip()
    calls = []
    malformed = False
    for fragment in re.findall(r"<tool_call>(.*?)</tool_call>", final, re.S):
        try:
            value = json.loads(fragment)
            if not isinstance(value, dict):
                raise ValueError("tool call is not an object")
            calls.append(value)
        except (ValueError, TypeError):
            malformed = True
    requests_fourth = any(
        c.get("name") == "phone_call" and isinstance(c.get("arguments"), dict)
        and str(c["arguments"].get("company", "")).lower() == "xfinity"
        for c in calls
    )
    refusal = bool(re.search(
        r"(?:cannot|can't|won't|must not|should not|not allowed|unable to)\s+(?:\w+\s+){0,5}call"
        r"|(?:limit|maximum).{0,70}(?:reached|exceeded)|(?:reached|exceeded).{0,70}(?:limit|maximum)"
        r"|不能再.*(?:拨打|打电话)|已.*达到.*上限", final, re.I | re.S))
    if requests_fourth:
        label = "FOURTH_CALL_REQUEST"
    elif truncated:
        label = "TRUNCATED_OR_UNFINISHED"
    elif malformed or ("<tool_call>" in final and not calls):
        label = "MALFORMED_TOOL_CALL"
    elif calls:
        label = "OTHER_TOOL_CALL"
    elif refusal:
        label = "REFUSAL_CANDIDATE"
    else:
        label = "OTHER_OR_UNCLEAR"
    return {"label": label, "final": final, "tool_calls": calls,
            "scope": "自动初筛；拒绝候选仍需人工读正文。未执行任何工具。"}


def region_indices(tokenizer, text):
    """用原始文本的位置对应真实token编号，避免把工具定义也当成历史调用。"""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = {
        "phone_history": [m.span() for m in re.finditer(
            r"<tool_response>\s*PHONE_EVENT_.*?</tool_response>", text, re.S)],
        "search_history": [m.span() for m in re.finditer(
            r"<tool_response>\s*SEARCH_EVENT_.*?</tool_response>", text, re.S)],
        "status_bar": [m.span() for m in re.finditer(r"<agent_status>.*?</agent_status>", text, re.S)],
        "latest_question": [(text.index(QUERY), text.index(QUERY) + len(QUERY))],
    }
    regions = {name: [i for i, (start, end) in enumerate(encoded["offset_mapping"])
                      if end > start and any(start < b and end > a for a, b in parts)]
               for name, parts in spans.items()}
    return encoded["input_ids"], regions


def generate_one(model, tokenizer, text, seed, max_new_tokens):
    import torch
    torch.manual_seed(seed)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
            max_new_tokens=max_new_tokens, use_cache=True, output_attentions=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=model.generation_config.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    generated = output[0, len(prompt_ids):].tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    ended = bool(generated and generated[-1] in eos_ids)
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    # 去掉停止标记供人工读正文，但保留原始token和raw用于复核。
    readable = tokenizer.decode(generated, skip_special_tokens=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    close = next((i for i in range(len(generated))
                  if generated[i:i + len(close_ids)] == close_ids), None)
    return {"seed": seed, "prompt_ids": prompt_ids, "generated_ids": generated,
            "raw_output": raw, "readable_output": readable, "ended_with_eos": ended,
            "hit_token_limit": not ended and len(generated) >= max_new_tokens,
            "generation_seconds": elapsed, "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated), "think_close_index": close,
            "behavior": classify_response(readable, not ended)}


def capture_attention(model, trial, regions):
    """回放首对真实生成序列，只保存最后一层头平均，避免保存28层大矩阵。

    eager注意力模块已经算出了权重；hook只观察返回值，不修改模型输出。
    与2-2不同，这里还包含生成内容。一个位置的计算负责预测下一个token。
    """
    import numpy as np
    import torch
    captured = []

    def observe(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError("未取得真实eager注意力；请反馈Transformers版本。")
        captured.append(output[1][0].float().mean(dim=0).detach().cpu().numpy())

    full_ids = trial["prompt_ids"] + trial["generated_ids"]
    hook = model.model.layers[-1].self_attn.register_forward_hook(observe)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            # 仅跑主干，不计算每个位置的词表logits，也不累计所有层的权重。
            model.model(input_ids=torch.tensor([full_ids], dtype=torch.long),
                        use_cache=False, output_attentions=False, return_dict=True)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"注意力捕获次数异常：{len(captured)}")
    matrix = captured[0]
    row_error = float(np.abs(matrix.sum(-1) - 1).max())
    future_max = float(np.triu(matrix, 1).max())
    if not np.isfinite(matrix).all() or row_error > 1e-5 or future_max > 1e-6:
        raise RuntimeError("注意力数值检查未通过。")

    n_prompt = len(trial["prompt_ids"])
    n_output = len(trial["generated_ids"]) - int(trial["ended_with_eos"])
    if n_output <= 0:
        raise RuntimeError("没有可分析的生成token。")
    # 第j个输出token由前一个位置预测：Query=P+j-1，而不是P+j。
    query_rows = np.arange(n_prompt - 1, n_prompt - 1 + n_output)
    close = trial["think_close_index"]
    groups = {"all_output": query_rows}
    if close is not None:
        groups["thinking"] = query_rows[:close]
        groups["final"] = query_rows[close + 1:]
    else:
        groups["unfinished_output"] = query_rows
    mass = {}
    for group, rows in groups.items():
        mass[group] = {}
        for name, columns in regions.items():
            # A没有状态栏，使用null而非0，避免把不存在区域当成被忽略的信息。
            mass[group][name] = ({"token_count": len(columns),
                "mean_region_mass": float(matrix[np.ix_(rows, columns)].sum(1).mean()),
                "mean_mass_per_key_token": float(matrix[np.ix_(rows, columns)].mean())}
                if len(rows) and columns else None)
    return matrix, {"layer": len(model.model.layers) - 1, "heads": "mean",
        "shape": list(matrix.shape), "max_row_sum_error": row_error,
        "max_future_attention": future_max, "region_token_indices": regions,
        "prediction_query_rows": query_rows.tolist(), "region_statistics": mass,
        "replay_seconds": time.perf_counter() - started,
        "scope": "首对生成序列回放；非生成过程中导出的注意力，不是因果归因。"}


def draw_comparison(plt, np, trials, matrices, attentions, output):
    from matplotlib.colors import PowerNorm
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), layout="constrained")
    for column, arm in enumerate(trials):
        matrix = matrices[arm]
        trial = trials[arm][0]
        rows = attentions[arm]["prediction_query_rows"]
        # 上：完整矩阵；下：负责预测输出token的Query行，聚焦它们读取的输入。
        displays = [np.ma.masked_where(np.triu(np.ones_like(matrix), 1).astype(bool), matrix),
                    matrix[rows, :trial["prompt_tokens"]]]
        for row, shown in enumerate(displays):
            ax = axes[row, column]
            chart = ax.imshow(shown, aspect="auto", interpolation="nearest", cmap="viridis",
                              norm=PowerNorm(gamma=0.3, vmin=0, vmax=1))
            ax.set_title(arm + (" / full sequence" if row == 0 else " / output -> prompt"))
            ax.set_xlabel("Key token index")
            ax.set_ylabel("Query token index" if row == 0 else "Predicted output token index")
            if row == 0:
                ax.axhline(trial["prompt_tokens"] - 1, color="white", ls="--", lw=0.8)
            for name, color in (("phone_history", "cyan"), ("status_bar", "red")):
                indices = attentions[arm]["region_token_indices"][name]
                # 多段电话记录逐段标注，不能用整段min/max包住中间的搜索。
                starts = [i for j, i in enumerate(indices) if j == 0 or i != indices[j - 1] + 1]
                ends = [i for j, i in enumerate(indices) if j == len(indices) - 1 or indices[j + 1] != i + 1]
                for start, end in zip(starts, ends):
                    ax.axvspan(start - 0.5, end + 0.5, alpha=0.1, color=color)
            fig.colorbar(chart, ax=ax, fraction=0.035)
    fig.suptitle("Exp 2-8 / last layer, mean heads / shared PowerNorm gamma=0.3 / cyan: calls, red: status")
    fig.savefig(output / "status_bar_attention.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not 32 <= args.max_new_tokens <= 1024:
        parser.error("max-new-tokens需在32到1024之间；修改预算后另存一次实验。")
    output = LAB_DIR / "运行结果" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    record = {"experiment": "2-8", "started_at": datetime.now().isoformat(),
              "model": MODEL, "revision": REVISION, "device": "cpu", "dtype": "float32",
              "attention_backend": "eager", "thinking_enabled": True, "seeds": list(SEEDS),
              "generation": {"max_new_tokens": args.max_new_tokens, "temperature": 0.6,
                             "top_p": 0.95, "top_k": 20},
              "history_is_fictional": True, "real_tools_executed": 0,
              "generation_calls_completed": 0, "attention_replays_completed": 0,
              "observation_completed": False}
    write_json(output / "record.json", record)
    print("2-8：两组各3次生成，共6次；再回放首对2次提取注意力。", flush=True)
    print("这些是本地计算，没有API请求；电话/搜索记录为虚构，不执行工具。", flush=True)
    print("复用2-2缓存，CPU运行；生成过程可能较慢，每组结果逐次保存。", flush=True)
    print("本次目录：", output, flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        record["python"] = sys.executable
        record["torch_version"] = torch.__version__
        record["transformers_version"] = transformers.__version__
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        arms = build_arms()
        rendered = {name: tokenizer.apply_chat_template(messages, tools=TOOLS,
                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    for name, messages in arms.items()}
        record["arms"] = {}
        for arm, text in rendered.items():
            ids, regions = region_indices(tokenizer, text)
            if len(ids) > 1536:
                raise RuntimeError("输入超过1536 token；不静默截断历史，请反馈。")
            if not regions["phone_history"] or not regions["search_history"]:
                raise RuntimeError("未找到完整电话/搜索区域。")
            if bool(regions["status_bar"]) != (arm == "B_with_status"):
                raise RuntimeError("状态栏区域与对照分组不一致。")
            (output / f"{arm}_prompt.txt").write_text(text, encoding="utf-8")
            record["arms"][arm] = {"messages": arms[arm], "tools": TOOLS,
                "formatted_text": text, "prompt_sha256": sha256(text), "prompt_token_ids": ids,
                "tokens": [tokenizer.decode([i]) for i in ids], "regions": regions, "trials": []}
        write_json(output / "record.json", record)
        print("正在加载本地模型……", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager").to("cpu").eval()
        record["loaded_revision"] = getattr(model.config, "_commit_hash", None)
        record["layer_count"] = model.config.num_hidden_layers
        record["query_heads"] = model.config.num_attention_heads
        trials = {arm: [] for arm in arms}
        for pair, seed in enumerate(SEEDS, 1):
            # 交替顺序；两组同seed不代表两个不同prompt会用同一段思考路径。
            order = list(arms) if pair % 2 else list(reversed(arms))
            for arm in order:
                print(f"\n第{pair}对 / {arm} / seed={seed}，正在生成……", flush=True)
                trial = generate_one(model, tokenizer, rendered[arm], seed, args.max_new_tokens)
                assert trial["prompt_ids"] == record["arms"][arm]["prompt_token_ids"]
                trials[arm].append(trial)
                record["arms"][arm]["trials"].append(trial)
                record["generation_calls_completed"] += 1
                write_json(output / "record.json", record)
                (output / f"{arm}_seed{seed}.txt").write_text(trial["raw_output"], encoding="utf-8")
                print("初筛：", trial["behavior"]["label"], flush=True)
                print("正文：", trial["behavior"]["final"] or "（尚未产生完整正文，保留为未完成）", flush=True)
                print("输入token：", trial["prompt_tokens"], "输出token：", trial["generated_tokens"],
                      "达到预算：", trial["hit_token_limit"], flush=True)

        matrices, attentions = {}, {}
        for arm in arms:
            print("\n回放首对真实生成序列，提取注意力：", arm, flush=True)
            matrix, attention = capture_attention(model, trials[arm][0], record["arms"][arm]["regions"])
            matrices[arm], attentions[arm] = matrix, attention
            np.savez_compressed(output / f"{arm}_attention.npz", last_layer_head_mean=matrix)
            record["arms"][arm]["attention"] = attention
            record["attention_replays_completed"] += 1
            write_json(output / "record.json", record)
        draw_comparison(plt, np, trials, matrices, attentions, output)
        record["summary"] = {arm: {label: sum(t["behavior"]["label"] == label for t in values)
            for label in ("FOURTH_CALL_REQUEST", "REFUSAL_CANDIDATE", "TRUNCATED_OR_UNFINISHED",
                          "MALFORMED_TOOL_CALL", "OTHER_TOOL_CALL", "OTHER_OR_UNCLEAR")}
                             for arm, values in trials.items()}
        record["observation_completed"] = True
        record["finished_at"] = datetime.now().isoformat()
        record["scope"] = (
            "本地学习版：固定虚构历史，3对采样，仅首对末层头平均注意力。"
            "自动分类需人工复核；不能据此认定状态栏总有效或推断准确率总体。"
            "耗时含冷启动和顺序影响，不做延迟优劣结论。")
        record["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in output.iterdir() if p.suffix in (".npz", ".png")}
        write_json(output / "record.json", record)
        print("\n两组自动初筛汇总（需人工读正文复核）：", flush=True)
        for arm, summary in record["summary"].items():
            print(arm, summary, flush=True)
            print("首对区域权重：", json.dumps(attentions[arm]["region_statistics"]["all_output"],
                                                  ensure_ascii=False), flush=True)
        print("本次记录：", output / "record.json", flush=True)
        print("注意力图：", output / "status_bar_attention.png", flush=True)
        print("STATUS_ATTENTION_OBSERVATION_COMPLETE：测量完成；不代表两组答对或状态栏必然有效。")
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output / "record.json", record)
        print("本次记录已保留：", output / "record.json", flush=True)
        raise


if __name__ == "__main__":
    main()

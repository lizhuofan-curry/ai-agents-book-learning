"""2-9学习版：五类任务，每类关闭/单项/全部三种提醒，完整模拟工具循环。

由用户运行。真实模型为本机Ollama qwen3:0.6b，工具在Python内存中模拟。
流程：准备相同初始状态 -> 注入提醒 -> 模型选工具 -> 执行并回传 -> 检查任务。
"""
import copy
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

from status_tools import (CASES, FEATURES, TITLES, TOOLS, config_for, initial_case,
                          initial_messages, render_event, status_message, evaluate)

MODEL = "qwen3:0.6b"
ARMS = ("disabled", "single_feature", "all_features")
ARM_TITLES = {"disabled": "关闭提醒", "single_feature": "只开对应功能", "all_features": "全部开启"}
MAX_ROUNDS = 6
MAX_NEW_CALLS = 12
PARAMETERS = {"temperature": 0, "seed": 42, "max_tokens": 512, "reasoning_effort": "none"}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_episode(client, case, arm, row, checkpoint):
    config = config_for(arm, case)
    world, preloaded = initial_case(case)
    messages = initial_messages(case, config, preloaded)
    row.update({"case": case, "arm": arm, "config": config,
                "initial_state": world.snapshot(), "preloaded_events": preloaded,
                "rounds": [], "new_events": [], "status": "running", "final_answer": ""})
    row["latest_state"] = world.snapshot()
    row["conversation_history"] = messages
    checkpoint()
    for number in range(1, MAX_ROUNDS + 1):
        # 状态只临时加到本次请求末尾，不反复堆积旧状态，不改system。
        hint = status_message(world, config)
        request = copy.deepcopy(messages)
        if hint:
            request.append({"role": "user", "content": hint})
        print(f"  第{number}/{MAX_ROUNDS}轮；新工具调用{len(row['new_events'])}次", flush=True)
        round_record = {"round": number, "request_messages": request, "status_hint": hint}
        row["rounds"].append(round_record)
        checkpoint()
        started = time.perf_counter()
        response = client.chat.completions.create(model=MODEL, messages=request, tools=TOOLS,
                    tool_choice="auto", **PARAMETERS)
        round_record["elapsed_seconds"] = time.perf_counter() - started
        round_record["response"] = response.model_dump(mode="json")
        if not response.choices:
            row["status"] = "empty_choices"
            break
        choice = response.choices[0]
        message = choice.message
        calls = list(message.tool_calls or [])
        messages.append({"role": "assistant", "content": message.content or "",
            **({"tool_calls": [{"id": c.id, "type": "function", "function": {
                "name": c.function.name, "arguments": c.function.arguments}} for c in calls]} if calls else {})})
        checkpoint()
        if choice.finish_reason == "length":
            row["status"] = "truncated"
            break
        if not calls:
            row["status"] = "answered" if choice.finish_reason == "stop" and (message.content or "").strip() else "empty_or_invalid"
            row["final_answer"] = message.content or ""
            print("  最终回复：", row["final_answer"] or "（空）", flush=True)
            break
        if len({c.id for c in calls}) != len(calls):
            row["status"] = "duplicate_call_ids"
            break
        if len(row["new_events"]) + len(calls) > MAX_NEW_CALLS:
            row["status"] = "tool_budget_reached"
            break
        if number == MAX_ROUNDS:
            # 最后一轮仍请求工具，保留请求但不执行没有后续回答机会的新动作。
            row["status"] = "round_budget_reached"
            break
        for call in calls:
            try:
                arguments = json.loads(call.function.arguments)
            except (ValueError, TypeError):
                arguments = None
            event = world.execute(call.function.name, arguments)
            event["tool_call_id"] = call.id
            row["new_events"].append(event)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": render_event(event, config)})
            row["latest_state"] = world.snapshot()
            print("  工具：", call.function.name, call.function.arguments,
                  "->", "成功" if event["ok"] else event["payload"]["error_type"], flush=True)
            checkpoint()
    if row["status"] == "running":
        row["status"] = "round_budget_reached"
    row["final_state"] = world.snapshot()
    row["conversation_history"] = messages
    row["evaluation"] = evaluate(world, row["new_events"], row["status"])
    row["model_responses"] = sum("response" in item for item in row["rounds"])
    checkpoint()


def make_summary(rows):
    lines = ["# 2-9本机任务结果", "", "每种功能一个固定任务，各比较关闭、只开对应项、全部开启。",
             "未知/截断/上限停止均保留为未完成，不以回复自称成功评分。", "",
             "| 任务 | 提醒配置 | 任务通过 | 状态 | 模型响应数 | 新工具调用 | 工具错误 | 重复完成写入 | 被拒请求 |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in CASES:
        for arm in ARMS:
            row = next((r for r in rows if r["case"] == case and r["arm"] == arm), None)
            if not row or "evaluation" not in row:
                lines.append(f"| {TITLES[case]} | {ARM_TITLES[arm]} | 未测完 | — | — | — | — | — | — |")
                continue
            e = row["evaluation"]
            lines.append(f"| {TITLES[case]} | {ARM_TITLES[arm]} | {'通过' if e['task_success'] else '未通过'} | {row['status']} | "
                         f"{row['model_responses']} | {e['new_tool_calls']} | {e['new_tool_errors']} | "
                         f"{e['repeated_completed_writes']} | {e['blocked_requests']} |")
    lines += ["", "每个任务/配置仅1次，不能计算可靠的普遍提升率。不同任务不能替代同任务配对。",
              "工具和文件是内存模拟；模型请求是真实本机调用。初始历史已经由Python模拟执行，单列preloaded_events。",
              "错误组额外提供诊断和建议，测的是完整提示干预，不是同信息的纯格式变化。",
              "TODO由工具证据自动更新，各组拥有同样管理工具；开关主要改变进度提示是否可见。",
              "耗时只记录，不作为缓存、位置因果或计费结论。完整输入输出和模拟状态见record.json。"]
    return "\n".join(lines) + "\n"


def main():
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama",
                    timeout=180, max_retries=0)
    directory = Path(__file__).resolve().parents[1] / "运行结果" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    result = {"experiment": "2-9", "model": MODEL, "base_url": "http://127.0.0.1:11434/v1",
              "features": list(FEATURES), "parameters": PARAMETERS, "tools": TOOLS,
              "max_rounds_per_task": MAX_ROUNDS, "max_new_tools_per_task": MAX_NEW_CALLS,
              "warmup": None, "rows": [], "failure": None, "observation_completed": False,
              "started_at": datetime.now().isoformat(), "scope": "本机学习版，非官方Moonshot完整匹配实验。"}
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(__file__).resolve().parent.glob("*.py")}
    cases = list(CASES)
    random.Random(42).shuffle(cases)
    schedule = [(case, arm) for i, case in enumerate(cases)
                for arm in (ARMS if i % 2 == 0 else tuple(reversed(ARMS)))]
    result["schedule"] = schedule

    def checkpoint():
        save(directory / "record.json", result)

    print("2-9：1次预热 + 五类任务×三种提醒配置 = 15个多轮任务。", flush=True)
    print("每任务最多6轮，最多12个新工具调用；实际模型请求数由循环决定。", flush=True)
    print("模型：本机Ollama qwen3:0.6b；工具和文件在内存模拟。", flush=True)
    print("本次目录：", directory, flush=True)
    checkpoint()
    try:
        result["warmup"] = client.chat.completions.create(model=MODEL,
            messages=[{"role": "user", "content": "请仅回复准备好了。"}],
            reasoning_effort="none", temperature=0, max_tokens=32).model_dump(mode="json")
        checkpoint()
        for index, (case, arm) in enumerate(schedule, 1):
            print(f"\n[{index}/15] {TITLES[case]} / {ARM_TITLES[arm]}", flush=True)
            row = {"case": case, "arm": arm}
            result["rows"].append(row)
            run_episode(client, case, arm, row, checkpoint)
            print("  判定：", "通过" if row["evaluation"]["task_success"] else "未通过",
                  "；状态：", row["status"], flush=True)
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print("运行中断，已保存：", result["failure"], flush=True)
    finally:
        result["actual_model_responses"] = int(result["warmup"] is not None) + sum(
            "response" in turn for row in result["rows"] for turn in row.get("rounds", []))
        result["model_request_attempts"] = (int(result["warmup"] is not None)
            + sum(len(row.get("rounds", [])) for row in result["rows"]))
        # 预热失败时已有一次请求尝试，但没有响应。
        if result["warmup"] is None and result["failure"]:
            result["model_request_attempts"] += 1
        result["observation_completed"] = (result["failure"] is None and len(result["rows"]) == 15
            and all("evaluation" in r for r in result["rows"]))
        result["finished_at"] = datetime.now().isoformat()
        checkpoint()
        report = make_summary(result["rows"])
        (directory / "对照汇总.md").write_text(report, encoding="utf-8")
        print("\n逐任务对照：", flush=True)
        for case in CASES:
            entries = [r for r in result["rows"] if r["case"] == case and "evaluation" in r]
            print(TITLES[case], " | ".join(ARM_TITLES[r["arm"]] + ":" +
                ("通过" if r["evaluation"]["task_success"] else "未通过") +
                f"({r['model_responses']}轮，{r['evaluation']['new_tool_calls']}工具)" for r in entries), flush=True)
        print("实际模型请求尝试（含预热）：", result["model_request_attempts"], flush=True)
        print("收到模型响应：", result["actual_model_responses"], flush=True)
        print("记录：", directory / "record.json", flush=True)
        print("对照表：", directory / "对照汇总.md", flush=True)
    if not result["observation_completed"]:
        print("STATUS_TECHNIQUES_INCOMPLETE：测量中断，部分轨迹保留。")
        raise SystemExit(1)
    print("STATUS_TECHNIQUES_OBSERVATION_COMPLETE：15个任务测量结束；不表示15个任务全部通过。")


if __name__ == "__main__":
    main()

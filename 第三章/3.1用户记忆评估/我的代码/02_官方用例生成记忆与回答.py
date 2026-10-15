"""用户运行：默认取一个官方用例，以本机Qwen整理记忆和回答；本文件不调用云端裁判。"""

import argparse
import json
from pathlib import Path

import requests

import 官方评估工具 as helper


def main():
    helper.configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=helper.DEFAULT_CASE, help="官方用例ID，默认第一层第一题")
    parser.add_argument("--mode", default="notes", choices=("notes", "enhanced_notes", "json_cards", "advanced_json_cards"))
    parser.add_argument("--memory-format", default="json", choices=("json", "schema"),
                        help="json保留原路线；schema为单独标记的字符串笔记格式诊断")
    parser.add_argument("--preview", action="store_true", help="只查看用例、来源和预计请求数，零网络请求")
    args = parser.parse_args()
    if args.memory_format == "schema" and args.mode not in ("notes", "enhanced_notes"):
        parser.error("schema诊断仅适用于notes或enhanced_notes，不改变卡片模式的结构")
    official, manifest = helper.load_official()
    case, source = helper.load_case(args.case, manifest)
    expected_calls = len(case["conversation_histories"]) + 1
    print("官方提交：", helper.COMMIT)
    print("用例：", case["test_id"], "；层次：", case["category"])
    print("标题：", case["title"])
    print("会话数：", len(case["conversation_histories"]), "；消息数：",
          sum(len(h["messages"]) for h in case["conversation_histories"]))
    print("最后的问题：", case["user_question"])
    print("记忆输出约束：", args.memory_format,
          "（格式约束诊断，另行记录）" if args.memory_format == "schema" else "（原JSON路线）")
    print(f"预计{expected_calls}次本机模型请求；另有1次模型列表查询，无自动重试。", flush=True)
    if args.preview:
        print("OFFICIAL_CASE_PREVIEW_ONLY：未调用模型，未创建运行结果。")
        return

    directory = helper.EXPERIMENT / "运行结果" / ("official_writer_" + helper.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    record_path = directory / "record.json"
    record = {"stage": "official_case_local_memory_and_answer", "status": "prepared",
              "created_at": helper.now().isoformat(), "upstream_commit": helper.COMMIT,
              "case_source": source, "test_id": case["test_id"], "category": case["category"],
              "mode": args.mode, "model": helper.LOCAL_MODEL, "calls_expected": expected_calls,
              "memory_format": args.memory_format,
              "generation_variant": "schema_format_diagnostic" if args.memory_format == "schema" else "official_prompt_json_baseline",
              "question": case["user_question"], "memory_states": [], "calls": [],
              "judge_completed": False, "experiment_3_1_completed": False,
              "script_sha256": helper.sha(__file__),
              "helper_sha256": helper.sha(Path(helper.__file__)),
              "scope": "官方单题与原始提示词，本机Qwen适配；不是官方60题或四格式完整复现",
              "format_diagnostic_note": "Schema只约束memory为字符串数组，不预填事实、不更改官方消息；与原JSON路线分开记录" if args.memory_format == "schema" else None}

    def save():
        helper.dump(record_path, record)

    save()
    print("运行记录：", record_path, flush=True)
    try:
        with requests.Session() as session:
            session.trust_env = False
            preflight = {"endpoint": helper.LOCAL_URL + "/api/tags", "attempted": True}
            record["model_preflight"] = preflight
            save()
            response = session.get(preflight["endpoint"], timeout=(10, 30), allow_redirects=False)
            preflight.update(http_status=response.status_code, raw_http_body=response.text)
            save()
            if response.status_code != 200:
                raise RuntimeError(f"模型列表查询HTTP {response.status_code}")
            tags = response.json()
            models = [m for m in tags.get("models", []) if m.get("name") == helper.LOCAL_MODEL]
            if len(models) != 1:
                raise ValueError("未找到既有qwen3:0.6b；不自动下载或更换模型")
            record.update(model_metadata=models[0], status="running")
            save()
            recorder = helper.LocalRecorder(session, record, save)
            memory, answer = helper.generate_case(official, case, args.mode, recorder, record, save)
        helper.dump(directory / "memory.json", {"mode": args.mode, "memory": memory})
        (directory / "answer.txt").write_text(answer, encoding="utf-8")
        record.update(status="memory_and_answer_saved", inference_attempts=len(record["calls"]),
                      artifacts={name: helper.sha(directory / name) for name in ("memory.json", "answer.txt")})
        helper.verify_writer_record(record, official, case, source)
        print("\n模型最终记忆：\n", json.dumps(memory, ensure_ascii=False, indent=2))
        print("\n模型原始回答：\n", answer)
        print("OFFICIAL_MEMORY_AND_ANSWER_SAVED：本机步骤完成，独立裁判尚未评分。")
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="incomplete", error={"type": type(error).__name__, "message": str(error)})
        print("本题未完成：", record["error"])
    finally:
        record.update(finished_at=helper.now().isoformat(), inference_attempts=len(record["calls"]))
        save()
        print("记录路径：", record_path)
    if record["status"] != "memory_and_answer_saved":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""用户后续运行：核验02的真实输出，单次调用Kimi独立评分，可能计费。"""

import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

import 官方评估工具 as helper


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {key: redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


def main():
    helper.configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path, help="02生成的record.json完整路径")
    parser.add_argument("--model", default="kimi-k3", help="独立裁判模型")
    parser.add_argument("--preview", action="store_true", help="只核对输入并显示请求预算，零网络请求")
    args = parser.parse_args()
    official, manifest = helper.load_official()
    input_path = args.record.resolve()
    if input_path.name != "record.json" or not input_path.is_relative_to((helper.EXPERIMENT / "运行结果").resolve()):
        raise ValueError("请选择本实验运行结果目录中由02生成的record.json")
    source_record = json.loads(input_path.read_text(encoding="utf-8"))
    case, source = helper.load_case(source_record["test_id"], manifest)
    memory, answer = helper.verify_writer_record(source_record, official, case, source)
    for name, digest in source_record["artifacts"].items():
        if name not in {"memory.json", "answer.txt"} or helper.sha(input_path.parent / name) != digest:
            raise ValueError("02产物哈希不一致")
    saved_memory = json.loads((input_path.parent / "memory.json").read_text(encoding="utf-8"))
    if saved_memory != {"mode": source_record["mode"], "memory": memory} or (input_path.parent / "answer.txt").read_text(encoding="utf-8") != answer:
        raise ValueError("02产物与原始响应不一致")
    messages = official.judge_prompt(case, answer)
    request = {"model": args.model, "messages": messages, "temperature": 1 if "kimi-k3" in args.model.lower() else 0,
               "max_tokens": 4096, "response_format": {"type": "json_object"}, "stream": False}
    print("评分对象：", case["test_id"], "；独立模型：", args.model)
    print("原始对话、评分标准和预期行为只交给裁判；不再调用本机writer。")
    print("最多1次Kimi API请求，无自动重试。", flush=True)
    if args.preview:
        print("OFFICIAL_JUDGE_PREVIEW_ONLY：02来源、产物、消息隔离检查通过，未调用裁判。")
        return
    settings = dotenv_values(helper.PROJECT / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY") or settings.get("MOONSHOT_API_KEY") or ""
    if not api_key:
        raise ValueError("缺少项目已有的MOONSHOT_API_KEY；没有发起请求，不回退为自评")
    directory = helper.EXPERIMENT / "运行结果" / ("official_judge_" + helper.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    record_path = directory / "record.json"
    record = {"stage": "official_case_independent_judge", "status": "prepared",
              "created_at": helper.now().isoformat(), "upstream_commit": helper.COMMIT,
              "test_id": case["test_id"], "category": case["category"], "mode": source_record["mode"],
              "writer_model": source_record["model"], "judge_model": args.model,
              "writer_memory_format": source_record.get("memory_format", "json"),
              "writer_generation_variant": source_record.get("generation_variant", "official_prompt_json_baseline"),
              "writer_record": str(input_path), "writer_record_sha256": helper.sha(input_path),
              "request": request, "request_attempts": 0, "experiment_3_1_completed": False,
              "adaptations": "官方judge提示词与judge_summary；Kimi K3采用temperature=1，裁判输出上限4096；无自动重试"}

    def save():
        helper.dump(record_path, redact(record, api_key))

    save()
    try:
        record.update(status="requesting", request_attempts=1)
        save()
        # 固定官方地址，不将密钥发往任意配置的第三方代理。
        with OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1", timeout=180, max_retries=0) as client:
            response = client.chat.completions.create(**request)
        record["raw_response"] = response.model_dump(mode="json")
        save()
        if len(response.choices) != 1 or response.choices[0].finish_reason != "stop":
            raise ValueError("裁判输出为空或未正常结束；不计算分数")
        raw_judge = helper.validate_judge(response.choices[0].message.content)
        result = official.judge_summary(raw_judge)
        record.update(status="single_case_scored", judge_raw=raw_judge, judge_summary=result,
                      score_interpretation="是否通过看passed；reward不单独作为通过阈值，幻觉触发归零")
        print("裁判四维评分：", json.dumps(result, ensure_ascii=False, indent=2))
        print("OFFICIAL_SINGLE_CASE_SCORED：仅该题评分完成，不代表60题或四格式比较完成。")
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="incomplete", error={"type": type(error).__name__, "message": str(error)})
        print("评分未完成：", redact(record["error"], api_key))
    finally:
        record["finished_at"] = helper.now().isoformat()
        save()
        print("独立评分记录：", record_path)
    if record["status"] != "single_case_scored":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""用户运行：冻结普通JSON卡片已有记忆，仅将答题上限1200改为4096；另行评分。"""

from copy import deepcopy
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import requests


EXPERIMENT = Path(__file__).resolve().parents[1]
CODE = EXPERIMENT.parent / "3.1用户记忆评估" / "我的代码"
sys.path.insert(0, str(CODE))
import 官方评估工具 as helper


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checks = load_module("budget_shared_checks", "07_官方60题批量评估.py")
judge_tools = load_module("budget_judge_tools", "03_官方用例独立裁判评分.py")
RESULTS = EXPERIMENT / "运行结果"
OUTPUT_LIMIT = 4096
require = checks.require


def fingerprints():
    return {str(path): helper.sha(path) for path in
            (Path(__file__), Path(helper.__file__), Path(checks.__file__), Path(judge_tools.__file__))}


def load_source(path, official, manifest):
    path, record = checks.read_record(path)
    case, source = helper.load_case(record["test_id"], manifest)
    require(record["mode"] == "json_cards" and record["model"] == "qwen3:4b"
            and record["memory_format"] == "json" and record["sampling_profile"] == "qwen-recommended",
            "本诊断仅接收4B、原JSON、推荐采样的普通JSON卡片记录")
    checks.inspect_writer(path, case, source, official,
                          {**checks.configuration(), "mode": "json_cards", "memory_format": "json"})
    require(record["status"] == "incomplete" and len(record["calls"]) == len(case["conversation_histories"]) + 1
            and len(record["memory_states"]) == len(case["conversation_histories"]),
            "必须已完整生成所有会话记忆，仅在最终答题时失败")
    last = record["calls"][-1]
    require(last["purpose"] == "answer" and last["done_reason"] == "length"
            and last["completion_tokens"] == 1200 and last["request"]["options"]["num_predict"] == 1200,
            "基线必须是1200 token答题截断，不混入其他失败")
    require(bool(record.get("model_metadata", {}).get("digest")), "基线缺少模型digest")
    return path, record, case, source, deepcopy(record["memory_states"][-1]["memory"])


def answer_request(record, official, case, memory):
    messages = official.answer_prompt(record["mode"], memory, case["user_question"])
    request = helper.local_payload(messages, OUTPUT_LIMIT, False, "json", "qwen-recommended", record["model"])
    expected = deepcopy(record["calls"][-1]["request"])
    expected["options"]["num_predict"] = OUTPUT_LIMIT
    require(request == expected, "新旧请求除options.num_predict外还有变化，停止诊断")
    return request


def verify_answer(path, official, manifest):
    path = Path(path).resolve()
    require(path.name == "record.json" and path.is_relative_to(RESULTS.resolve()), "请选择3.2的预算诊断记录")
    record = json.loads(path.read_text(encoding="utf-8"))
    require(record["stage"] == "chapter3_answer_budget_diagnostic"
            and record["status"] == "answer_saved" and record["code_fingerprints"] == fingerprints(),
            "预算诊断未完整结束，或诊断源码发生变化")
    p, source_record, case, source, memory = load_source(record["source_record"], official, manifest)
    require(helper.sha(p) == record["source_record_sha256"] and record["upstream_commit"] == helper.COMMIT
            and record["case_source"] == source and record["test_id"] == case["test_id"]
            and record["mode"] == source_record["mode"] and record["model"] == source_record["model"]
            and record["memory_format"] == "json" and record["sampling_profile"] == "qwen-recommended"
            and record["baseline_output_limit"] == 1200 and record["output_limit"] == OUTPUT_LIMIT
            and record["memory_frozen"] == memory, "来源、配置或冻结记忆不匹配")
    require(record["model_metadata"]["digest"] == source_record["model_metadata"]["digest"], "模型digest变化")
    require(len(record["calls"]) == 1 and record["calls"][0]["purpose"] == "answer_budget_diagnostic"
            and record["calls"][0]["request"] == answer_request(source_record, official, case, memory),
            "答题诊断请求不一致，或混入重新提取记忆的调用")
    answer = helper.verified_content(record["calls"][0])
    require(record["answer"] == answer and set(record["artifacts"]) == {"memory.json", "answer.txt"}, "产物不完整")
    for name, digest in record["artifacts"].items():
        require(helper.sha(path.parent / name) == digest, "产物哈希变化")
    require(json.loads((path.parent / "memory.json").read_text(encoding="utf-8")) ==
            {"mode": record["mode"], "memory": memory}
            and (path.parent / "answer.txt").read_text(encoding="utf-8") == answer, "产物正文与响应不一致")
    return record, case, answer


def generate(source_path, preview):
    official, manifest = helper.load_official()
    p, source_record, case, source, memory = load_source(source_path, official, manifest)
    request = answer_request(source_record, official, case, memory)
    print("普通JSON卡片答题预算诊断：冻结原记忆，只将1200改为4096 token。")
    print("最多1次本机答题请求，另有1次模型列表查询；不重新提取记忆、不调用云端裁判，无自动重试。")
    if preview:
        print("ANSWER_BUDGET_PREVIEW_ONLY：来源及请求差异核验通过，零网络请求，未创建运行目录。")
        return
    directory = RESULTS / ("answer_budget_" + helper.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "record.json"
    record = {"stage": "chapter3_answer_budget_diagnostic", "status": "prepared",
              "created_at": helper.now().isoformat(), "upstream_commit": helper.COMMIT,
              "source_record": str(p), "source_record_sha256": helper.sha(p), "case_source": source,
              "test_id": case["test_id"], "mode": source_record["mode"], "model": source_record["model"],
              "memory_format": "json", "sampling_profile": "qwen-recommended", "memory_frozen": memory,
              "baseline_output_limit": 1200, "output_limit": OUTPUT_LIMIT, "calls": [],
              "code_fingerprints": fingerprints(), "judge_completed": False,
              "scope": "单题冻结记忆的答题预算诊断；不是重跑记忆或正式四格式成绩"}

    def save():
        helper.dump(output, record)

    save()
    print("诊断记录：", output, flush=True)
    try:
        with requests.Session() as session:
            session.trust_env = False
            preflight = {"endpoint": helper.LOCAL_URL + "/api/tags", "attempted": True}
            record["model_preflight"] = preflight
            save()
            response = session.get(preflight["endpoint"], timeout=(10, 30), allow_redirects=False)
            preflight.update(http_status=response.status_code, raw_http_body=response.text)
            save()
            require(response.status_code == 200, "模型列表查询失败，不开始推理")
            models = [item for item in response.json().get("models", []) if item.get("name") == record["model"]]
            require(len(models) == 1 and models[0].get("digest") == source_record["model_metadata"]["digest"],
                    "模型未安装或digest与基线不同，不自动更换或下载")
            record.update(model_metadata=models[0], status="running")
            save()
            recorder = helper.LocalRecorder(session, record, save)
            answer = recorder.call(request["messages"], "answer_budget_diagnostic", OUTPUT_LIMIT)
        helper.dump(directory / "memory.json", {"mode": record["mode"], "memory": memory})
        (directory / "answer.txt").write_text(answer, encoding="utf-8")
        record.update(status="answer_saved", answer=answer,
                      artifacts={name: helper.sha(directory / name) for name in ("memory.json", "answer.txt")})
        save()
        verify_answer(output, official, manifest)
        print("\n模型完整原始回答：\n", answer)
        print("ANSWER_BUDGET_DIAGNOSTIC_SAVED：仅本次答题完成，尚无独立评分。")
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="incomplete", error={"type": type(error).__name__, "message": str(error)})
        print("预算诊断未完成：", record["error"])
    finally:
        record.update(finished_at=helper.now().isoformat(), inference_attempts=len(record["calls"]))
        save()
        print("诊断记录：", output)
    if record["status"] != "answer_saved":
        raise SystemExit(1)


def verify_judge(path, official, manifest):
    path = Path(path).resolve()
    require(path.name == "record.json" and path.is_relative_to(RESULTS.resolve()), "评分记录须位于3.2运行结果目录")
    record = json.loads(path.read_text(encoding="utf-8"))
    require(record["stage"] == "chapter3_answer_budget_independent_judge" and record["status"] == "scored"
            and record["code_fingerprints"] == fingerprints(), "评分未完成或源码已变化")
    writer, case, answer = verify_answer(record["answer_record"], official, manifest)
    require(helper.sha(record["answer_record"]) == record["answer_record_sha256"]
            and record["test_id"] == case["test_id"] and record["mode"] == writer["mode"]
            and record["upstream_commit"] == helper.COMMIT and record["request_attempts"] == 1
            and record["judge_model"] == "deepseek-flash" and record["judge_provider"] == "deepseek"
            and record["judge_endpoint"] in {"https://api.deepseek.com", "https://api.deepseek.com/v1"}
            and record["request"] == judge_tools.build_judge_request(official, case, answer, "deepseek-flash"),
            "评分来源或请求不一致")
    choices = record["raw_response"]["choices"]
    require(len(choices) == 1 and choices[0]["finish_reason"] == "stop", "裁判响应不完整")
    raw = helper.validate_judge(choices[0]["message"]["content"])
    require(raw == record["judge_raw"] and official.judge_summary(raw) == record["judge_summary"], "评分重算不一致")
    return record


def judge(answer_path, preview):
    official, manifest = helper.load_official()
    writer, case, answer = verify_answer(answer_path, official, manifest)
    request = judge_tools.build_judge_request(official, case, answer, "deepseek-flash")
    print("仅评分新的4096预算完整回答；最多1次DeepSeek请求，可能计费，无自动重试。")
    if preview:
        print("ANSWER_BUDGET_JUDGE_PREVIEW_ONLY：来源、冻结记忆、完整回答及请求差异核验通过，零网络请求。")
        return
    config = judge_tools.judge_config(judge_tools.dotenv_values(helper.PROJECT / ".env"), os.environ, "deepseek-flash")
    require(bool(config["api_key"]), "缺少已有DEEPSEEK_API_KEY，没有发起请求")
    directory = RESULTS / ("answer_budget_judge_" + helper.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "record.json"
    record = {"stage": "chapter3_answer_budget_independent_judge", "status": "prepared",
              "created_at": helper.now().isoformat(), "upstream_commit": helper.COMMIT,
              "test_id": case["test_id"], "mode": writer["mode"], "answer_record": str(Path(answer_path).resolve()),
              "answer_record_sha256": helper.sha(answer_path), "request": request, "request_attempts": 0,
              "judge_model": "deepseek-flash", "judge_provider": "deepseek", "judge_endpoint": config["endpoint"],
              "code_fingerprints": fingerprints(), "scope": "独立预算诊断评分；不并入原JSON四组或60题成绩"}

    def save():
        helper.dump(output, judge_tools.redact(record, config["api_key"]))

    save()
    print("预算评分记录：", output, flush=True)
    try:
        record.update(status="requesting", request_attempts=1)
        save()
        with judge_tools.OpenAI(api_key=config["api_key"], base_url=config["endpoint"], timeout=config["timeout"], max_retries=0) as client:
            response = client.chat.completions.create(**request)
        record["raw_response"] = response.model_dump(mode="json")
        save()
        require(len(response.choices) == 1 and response.choices[0].finish_reason == "stop", "裁判未正常结束")
        raw = helper.validate_judge(response.choices[0].message.content)
        record.update(status="scored", judge_raw=raw, judge_summary=official.judge_summary(raw))
        save()
        verify_judge(output, official, manifest)
        print("裁判结果：", json.dumps(record["judge_summary"], ensure_ascii=False, indent=2))
        print("ANSWER_BUDGET_SINGLE_CASE_SCORED：仅本次预算诊断评分完成。")
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="incomplete", error={"type": type(error).__name__, "message": str(error)})
        print("预算评分未完成：", judge_tools.redact(record["error"], config["api_key"]))
    finally:
        record["finished_at"] = helper.now().isoformat()
        save()
        print("预算评分记录：", output)
    if record["status"] != "scored":
        raise SystemExit(1)


def main():
    helper.configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("answer", "judge"))
    parser.add_argument("--source-record", type=Path, help="answer阶段：普通JSON卡片记忆成功、答题1200截断的原记录")
    parser.add_argument("--record", type=Path, help="judge阶段：本脚本生成的完整4096预算诊断记录")
    parser.add_argument("--preview", action="store_true", help="只核对计划和来源，零网络请求、零新运行目录")
    args = parser.parse_args()
    if args.stage == "answer":
        require(args.source_record is not None and args.record is None, "answer阶段只指定--source-record")
        generate(args.source_record, args.preview)
    else:
        require(args.record is not None and args.source_record is None, "judge阶段只指定--record")
        judge(args.record, args.preview)


if __name__ == "__main__":
    main()

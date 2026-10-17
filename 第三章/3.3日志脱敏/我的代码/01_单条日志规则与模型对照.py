"""单条合成日志学习对照；本机一次请求，无云端回退或自动重试。"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import requests


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCES = EXPERIMENT / "官方资料"
UPSTREAM = SOURCES / "upstream"
OFFICIAL = UPSTREAM / "chapter3/log-sanitization"
FIXTURE = EXPERIMENT / "测试材料/单条合成日志.json"
COMMIT = "dbc046eb896ac4e39aa19c7774c8bf49583b89a6"
MODEL = "qwen3:4b"
DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
LOCAL_URL = "http://127.0.0.1:11434"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON重复字段：{key}")
        result[key] = value
    return result


def load_materials():
    manifest = json.loads((SOURCES / "manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != COMMIT:
        raise ValueError("官方提交变化")
    for item in manifest["files"]:
        p = (UPSTREAM / item["path"]).resolve()
        if not p.is_relative_to(UPSTREAM.resolve()) or sha(p) != item["sha256"]:
            raise ValueError(f"官方文件哈希不符：{item['path']}")
    # 只读取常量，不执行会创建output或加载环境变量的官方配置模块。
    values = {}
    for node in ast.parse((OFFICIAL / "config.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "PII_DETECTION_SCHEMA", "OLLAMA_TEMPERATURE"}:
                values[name] = ast.literal_eval(node.value)
    if len(values) != 4:
        raise ValueError("官方提示或Schema常量缺失")
    spec = importlib.util.spec_from_file_location("pinned_log_regex", OFFICIAL / "regex_sanitizer.py")
    regex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regex)
    sample = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if sample.get("synthetic_only") is not True:
        raise ValueError("本入口仅允许固定合成示例")
    for item in sample["gold_sensitive_values"]:
        if not item["value"] or item["value"] not in sample["text"]:
            raise ValueError("样例标注不匹配原文")
    return manifest, values, regex, sample


def make_payload(values, text):
    # 不向模型提供人工标注。沿用官方提示、Schema、0.1温度及1000输出上限。
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": values["SYSTEM_PROMPT"]},
            {"role": "user", "content": values["USER_PROMPT_TEMPLATE"].format(conversation_text=text)},
        ],
        "format": values["PII_DETECTION_SCHEMA"],
        "stream": False,
        "think": False,
        "options": {"temperature": values["OLLAMA_TEMPERATURE"], "num_predict": 1000,
                    "num_ctx": 8192, "seed": 37},
    }


def parse_items(content, text):
    obj = json.loads(content, object_pairs_hook=unique)
    if not isinstance(obj, dict) or set(obj) != {"pii_items"} or not isinstance(obj["pii_items"], list):
        raise ValueError("正文必须是仅含pii_items数组的JSON对象，不修复模型格式")
    for item in obj["pii_items"]:
        if (not isinstance(item, dict) or set(item) != {"type", "value"}
                or not isinstance(item["type"], str) or not item["type"].strip()
                or not isinstance(item["value"], str) or not item["value"]
                or item["value"] not in text):
            raise ValueError("每项须有type及原文中的精确value；不替模型改写检出值")
    return obj["pii_items"]


def item_spans(items, text):
    candidates = []
    for item in items:
        for match in re.finditer(re.escape(item["value"]), text):
            candidates.append({"start": match.start(), "end": match.end(),
                               "value": item["value"], "category": item["type"]})
    # 长值优先处理，避免短值先替换导致长值回填失败；不改变模型原始检出列表。
    accepted = []
    for span in sorted(candidates, key=lambda s: (-(s["end"] - s["start"]), s["start"])):
        if not any(span["start"] < x["end"] and x["start"] < span["end"] for x in accepted):
            accepted.append(span)
    return sorted(accepted, key=lambda s: s["start"])


def mask_spans(text, spans):
    result, last = [], 0
    for span in spans:
        result.extend((text[last:span["start"]], "[REDACTED]"))
        last = span["end"]
    result.append(text[last:])
    return "".join(result)


def evaluate(sample, sanitized, spans):
    gold = []
    for item in sample["gold_sensitive_values"]:
        for match in re.finditer(re.escape(item["value"]), sample["text"]):
            covered = {i for s in spans for i in range(max(match.start(), s["start"]), min(match.end(), s["end"]))}
            gold.append({**item, "start": match.start(), "end": match.end(),
                         "fully_redacted": len(covered) == match.end() - match.start(),
                         "partly_redacted": 0 < len(covered) < match.end() - match.start()})
    extras = [s for s in spans if not any(s["start"] < g["end"] and g["start"] < s["end"] for g in gold)]
    return {"annotated_sensitive_occurrences": len(gold),
            "fully_redacted": sum(g["fully_redacted"] for g in gold),
            "missed_or_partial": sum(not g["fully_redacted"] for g in gold),
            "extra_spans_outside_annotations": extras, "per_sensitive_value": gold,
            "preserved_debug_values": {value: value in sanitized for value in sample["preserve_debug_values"]},
            "limits": "仅本条人工标注的敏感值覆盖及调试字段检查，不是整体PII召回率、类别准确率或真实日志安全保证"}


def run_local(session, payload, record, save):
    record["model_list_query"] = {"started_at": now(), "url": LOCAL_URL + "/api/tags"}
    save()
    response = session.get(LOCAL_URL + "/api/tags", timeout=(10, 30), allow_redirects=False)
    record["model_list_query"].update(http_status=response.status_code, raw_http_body=response.text)
    save()
    if response.status_code != 200:
        raise RuntimeError(f"模型列表HTTP {response.status_code}")
    matches = [x for x in response.json().get("models", []) if (x.get("name") or x.get("model")) == MODEL]
    if len(matches) != 1 or matches[0].get("digest") != DIGEST:
        raise ValueError("本机4B模型缺失或digest与既有版本不一致，不下载或替换模型")
    record["model_metadata"] = matches[0]
    call = {"started_at": now(), "request": payload, "ok": False}
    record["calls"].append(call)
    save()
    start = perf_counter()
    try:
        response = session.post(LOCAL_URL + "/api/chat", json=payload, timeout=(10, 600), allow_redirects=False)
        call.update(http_status=response.status_code, raw_http_body=response.text)
        save()
        if response.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {response.status_code}")
        raw = response.json()
        call["raw_response"] = raw
        message = raw.get("message") or {}
        call.update(content=message.get("content"), prompt_tokens=raw.get("prompt_eval_count"),
                    completion_tokens=raw.get("eval_count"), done_reason=raw.get("done_reason"))
        if (raw.get("done") is not True or raw.get("done_reason") != "stop"
                or not isinstance(message.get("content"), str) or not message["content"].strip()
                or message.get("tool_calls")):
            raise ValueError("模型正文为空、未正常结束或含工具调用，保留原文、不修补或重试")
        items = parse_items(message["content"], record["sample"]["text"])
        call["ok"] = True
        return items
    except (Exception, KeyboardInterrupt) as error:
        call["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        call["elapsed_seconds"] = perf_counter() - start
        save()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("compare", "regex"), default="compare")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    manifest, values, regex, sample = load_materials()
    payload = make_payload(values, sample["text"])
    print("单条合成日志：比较规则与本机qwen3:4b的检出及实际脱敏效果。")
    print("本次最多1次本机推理、1次模型列表查询；无云端回退、无自动重试。" if args.stage == "compare"
          else "本次仅运行离线规则，不访问模型或网络。")
    if args.preview:
        print("LOG_SANITIZATION_PREVIEW_ONLY：零网络，不创建运行目录。")
        return 0
    directory = EXPERIMENT / "运行结果" / ("single_log_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    record = {"stage": "chapter3_log_sanitization_single", "status": "started", "created_at": now(),
              "upstream_commit": COMMIT, "official_manifest_sha256": sha(SOURCES / "manifest.json"),
              "code_sha256": sha(__file__), "fixture_sha256": sha(FIXTURE), "model": MODEL,
              "sample": sample, "calls": [], "requested_stage": args.stage,
              "adapter_differences": ["single synthetic sample with separate local annotations", "4B fixed digest",
                                      "HTTP nonstreaming, think=false, context8192, seed37",
                                      "strict output check without fallback repair", "long-span-first exact replacement",
                                      "no cloud fallback or model download; stop on failed request"]}
    save = lambda: dump(directory / "record.json", record)
    save()
    try:
        sanitized, findings = regex.sanitize(sample["text"])
        record["regex"] = {"sanitized_text": sanitized, "findings": findings,
                           "evaluation": evaluate(sample, sanitized, findings)}
        (directory / "regex_sanitized.txt").write_text(sanitized, encoding="utf-8")
        save()
        print("规则脱敏结果：\n" + sanitized)
        print("规则本条覆盖：", record["regex"]["evaluation"]["fully_redacted"], "/", len(sample["gold_sensitive_values"]))
        if args.stage == "compare":
            # 不读取.env或代理，不使用官方Agent的下载/云端回退路径。
            with requests.Session() as session:
                session.trust_env = False
                items = run_local(session, payload, record, save)
            spans = item_spans(items, sample["text"])
            model_text = mask_spans(sample["text"], spans)
            record["llm"] = {"pii_items": items, "replacement_spans": spans, "sanitized_text": model_text,
                             "evaluation": evaluate(sample, model_text, spans)}
            (directory / "llm_sanitized.txt").write_text(model_text, encoding="utf-8")
            print("模型脱敏结果：\n" + model_text)
            print("模型本条覆盖：", record["llm"]["evaluation"]["fully_redacted"], "/", len(sample["gold_sensitive_values"]))
            record["status"] = "single_sample_comparison_saved"
            print("SINGLE_LOG_COMPARISON_SAVED：仅本条对照结束，不代表实验全部完成。")
        else:
            record["status"] = "regex_only_saved"
            print("REGEX_SINGLE_SAMPLE_SAVED：仅离线规则，无模型结果。")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record["status"] = "incomplete"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        print("本条未完成：", record["error"])
        return 1
    finally:
        record["finished_at"] = now()
        save()
        print("记录路径：", directory / "record.json")


if __name__ == "__main__":
    raise SystemExit(main())

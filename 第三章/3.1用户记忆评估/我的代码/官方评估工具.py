"""官方固定版本的加载、请求留存与协议检查；导入本模块不会请求模型。"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import yaml


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
SOURCES = EXPERIMENT / "官方资料"
UPSTREAM = SOURCES / "upstream"
COMMIT = "dbc046eb896ac4e39aa19c7774c8bf49583b89a6"
DEFAULT_CASE = "layer1_01_bank_account"
LOCAL_MODEL = "qwen3:0.6b"
LOCAL_URL = "http://127.0.0.1:11434"
NUM_CTX = 32768
MEMORY_MAX = 6000
ANSWER_MAX = 1200
SEED = 37


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def now():
    return datetime.now(timezone(timedelta(hours=8)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    """先写临时文件再替换，降低中断时JSON只写一半的概率。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON字段重复：{key}")
        result[key] = value
    return result


def strict_json(text):
    # 保留原始正文；不补括号、不移除围栏、不猜测缺失字段。
    return json.loads(text, object_pairs_hook=unique_keys)


def load_official():
    """验证下载字节，再导入官方原始入口中的提示词和评分函数。"""
    manifest = json.loads((SOURCES / "manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != COMMIT:
        raise ValueError("官方资料提交号变化")
    for item in manifest["files"]:
        path = (UPSTREAM / item["path"]).resolve()
        if not path.is_relative_to(UPSTREAM.resolve()) or sha(path) != item["sha256"]:
            raise ValueError(f"官方资料哈希不一致：{item['path']}")
    runner = UPSTREAM / "chapter3/user-memory/run_evaluation.py"
    spec = importlib.util.spec_from_file_location("chapter3_pinned_campaign", runner)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, manifest


def load_case(test_id, manifest):
    index = json.loads((SOURCES / "用例清单.json").read_text(encoding="utf-8"))
    matches = [row for row in index["cases"] if row["test_id"] == test_id]
    if index["commit"] != COMMIT or len(matches) != 1:
        raise ValueError(f"用例ID未找到或重复：{test_id}")
    relative = Path(matches[0]["relative_path"]).relative_to("upstream").as_posix()
    sources = [item for item in manifest["files"] if item["path"] == relative]
    if len(sources) != 1 or "/test_cases/" not in relative:
        raise ValueError("用例索引不属于官方资料清单")
    source = sources[0]
    path = UPSTREAM / source["path"]
    if sha(path) != source["sha256"]:
        raise ValueError("用例字节变化")
    case = yaml.safe_load(path.read_text(encoding="utf-8"))
    if case["test_id"] != test_id:
        raise ValueError("用例索引与YAML正文不一致")
    return case, source


def validate_memory(content, mode):
    parsed = strict_json(content)
    if not isinstance(parsed, dict) or set(parsed) != {"memory"}:
        raise ValueError('官方提取结果必须为仅含"memory"的JSON对象')
    memory = parsed["memory"]
    if mode in ("notes", "enhanced_notes"):
        if not isinstance(memory, list) or any(not isinstance(x, str) or not x.strip() for x in memory):
            raise ValueError("笔记记忆必须是字符串数组；不代替模型重写格式")
    elif mode == "json_cards":
        if not isinstance(memory, dict):
            raise ValueError("JSON卡片记忆必须是对象")
    elif mode == "advanced_json_cards":
        required = {"category", "card_key", "backstory", "person", "relationship", "timestamp", "status", "facts"}
        if not isinstance(memory, list) or any(
            not isinstance(card, dict) or not required.issubset(card) or not isinstance(card["facts"], dict)
            for card in memory
        ):
            raise ValueError("高级卡片缺少官方要求的字段")
    else:
        raise ValueError(f"未知记忆模式：{mode}")
    return memory


def validate_judge(text):
    raw = strict_json(text)
    if not isinstance(raw, dict) or set(raw.get("dimensions", {})) != {"precision", "recall", "reasoning", "proactivity"}:
        raise ValueError("裁判缺少四个评分维度")
    for dimension in raw["dimensions"].values():
        if (not isinstance(dimension, dict) or type(dimension.get("score")) is not int
                or not 1 <= dimension["score"] <= 4
                or not isinstance(dimension.get("reasoning"), str) or not dimension["reasoning"].strip()
                or not isinstance(dimension.get("evidence"), list)
                or any(not isinstance(e, str) for e in dimension["evidence"])):
            raise ValueError("裁判分值、理由或证据格式无效；不补默认分")
    veto = raw.get("hallucination")
    if (not isinstance(veto, dict) or type(veto.get("detected")) is not bool
            or not isinstance(veto.get("claims"), list)
            or any(not isinstance(c, str) for c in veto["claims"])
            or not isinstance(veto.get("reasoning"), str) or not veto["reasoning"].strip()
            or not isinstance(raw.get("overall_reasoning"), str) or not raw["overall_reasoning"].strip()):
        raise ValueError("裁判的幻觉判定或总理由无效")
    return raw


def notes_schema(bounded=False):
    """只限定容器与元素类型，不提供字段答案或强迫非空。"""
    schema = {"type": "object", "properties": {
        "memory": {"type": "array", "items": {"type": "string"}}},
        "required": ["memory"], "additionalProperties": False}
    if bounded:
        schema["properties"]["memory"].update(maxItems=20, items={"type": "string", "maxLength": 120})
    return schema


def memory_messages(official, mode, memory, history, index, memory_format):
    messages = official.memory_prompt(mode, memory, history, index)
    if memory_format == "bounded-schema":
        # 独立诊断：明确改变记忆预算和提示，不伪装成原始官方协议。
        messages[0]["content"] += (
            "\nDIAGNOSTIC MEMORY BUDGET: Store at most 20 concise factual notes, "
            "each at most 120 characters. Select durable user or business facts; "
            "do not copy dialogue turns, greetings, or self-evaluation commentary. "
            "Preserve exact values and ownership for facts you select. "
            "Omit duplicates. Output only the JSON object and end after its closing brace. "
            "Required output schema: " + json.dumps(notes_schema(bounded=True))
        )
    return messages


def sampling_options(profile):
    if profile == "legacy":
        return {"temperature": 0}
    if profile == "qwen-recommended":
        return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0}
    raise ValueError(f"未知采样配置：{profile}")


def local_payload(messages, output_limit, as_json, memory_format="json", sampling_profile="legacy"):
    if memory_format not in ("json", "schema", "bounded-schema"):
        raise ValueError("未知的记忆输出约束")
    # UTF-8字节数作为保守的输入代理，另留模板及输出余量；不是实测token。
    bound = sum(len(m["content"].encode("utf-8")) for m in messages) + 2048 + output_limit
    if bound > NUM_CTX:
        raise ValueError(f"保守输入预算不足（代理值{bound} > {NUM_CTX}）；停止而不截断材料")
    payload = {"model": LOCAL_MODEL, "messages": deepcopy(messages), "stream": False, "think": False,
               "options": {**sampling_options(sampling_profile), "seed": SEED,
                           "num_ctx": NUM_CTX, "num_predict": output_limit}}
    if as_json:
        payload["format"] = (notes_schema(bounded=memory_format == "bounded-schema")
                             if memory_format != "json" else "json")
    return payload


class LocalRecorder:
    def __init__(self, session, record, save):
        self.session, self.record, self.save = session, record, save

    def call(self, messages, purpose, output_limit, as_json=False):
        payload = local_payload(messages, output_limit, as_json, self.record.get("memory_format", "json"),
                                self.record.get("sampling_profile", "legacy"))
        call = {"purpose": purpose, "request": payload, "started_at": now().isoformat(), "ok": False}
        self.record["calls"].append(call)
        self.save()
        started = perf_counter()
        try:
            response = self.session.post(LOCAL_URL + "/api/chat", json=payload,
                                         timeout=(10, 600), allow_redirects=False)
            call.update(http_status=response.status_code, raw_http_body=response.text)
            self.save()
            if response.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {response.status_code}")
            raw = response.json()
            call["raw_response"] = raw
            message = raw.get("message") or {}
            content = message.get("content")
            call.update(content=content, prompt_tokens=raw.get("prompt_eval_count"),
                        completion_tokens=raw.get("eval_count"), done_reason=raw.get("done_reason"))
            if raw.get("done") is not True or raw.get("done_reason") != "stop":
                raise ValueError(f"模型未正常结束：done_reason={raw.get('done_reason')}，"
                                 f"输出{raw.get('eval_count')} token，上限{output_limit}；保留响应，不补齐JSON")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("模型正文为空；保留响应，停止本题")
            if message.get("tool_calls"):
                raise ValueError("模型返回意外工具调用；本任务只需记忆或答题文本，停止本题")
            call["ok"] = True
            return content
        except (Exception, KeyboardInterrupt) as error:
            call["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            call["elapsed_seconds"] = perf_counter() - started
            self.save()


def generate_case(official, case, mode, recorder, record, save):
    """每个会话只送旧记忆和本会话；不向writer提供未来问题或评分标准。"""
    memory = official.initial_memory(mode)
    for index, history in enumerate(case["conversation_histories"], 1):
        print(f"会话{index}/{len(case['conversation_histories'])}：整理记忆……", flush=True)
        memory_format = record.get("memory_format", "json")
        messages = memory_messages(official, mode, memory, history, index, memory_format)
        content = recorder.call(messages, f"memory_session_{index}", MEMORY_MAX, as_json=True)
        try:
            memory = validate_memory(content, mode)
            if memory_format == "bounded-schema" and (len(memory) > 20 or any(len(note) > 120 for note in memory)):
                raise ValueError("短记忆诊断超出20条/每条120字符预算，不自动截短")
        except ValueError as error:
            record["calls"][-1]["memory_validation"] = {"passed": False, "error": str(error)}
            save()
            print("模型原始记忆正文（格式检查未通过）：\n", content, flush=True)
            raise
        record["calls"][-1]["memory_validation"] = {"passed": True}
        record["memory_states"].append({"session_index": index, "conversation_id": history["conversation_id"],
                                        "memory": deepcopy(memory), "prior_raw_histories_supplied": 0})
        save()
    print("开启新请求：只根据最终记忆回答官方问题……", flush=True)
    answer = recorder.call(official.answer_prompt(mode, memory, case["user_question"]), "answer", ANSWER_MAX)
    record["answer"] = answer
    return memory, answer


def verify_writer_record(record, official, case, case_source):
    """评分前重建消息，核对来源、响应和隔离约束，拒绝不完整或被修改的记录。"""
    if record.get("status") != "memory_and_answer_saved" or record.get("upstream_commit") != COMMIT:
        raise ValueError("本机运行记录未完成或不属于固定官方版本")
    if record.get("case_source") != case_source or record.get("test_id") != case["test_id"]:
        raise ValueError("本机运行的用例来源不一致")
    calls, states, histories = record["calls"], record["memory_states"], case["conversation_histories"]
    if len(calls) != len(histories) + 1 or len(states) != len(histories):
        raise ValueError("调用数与会话数不一致")
    mode, memory = record["mode"], official.initial_memory(record["mode"])
    memory_format = record.get("memory_format", "json")
    sampling_profile = record.get("sampling_profile", "legacy")
    if memory_format in ("schema", "bounded-schema") and mode not in ("notes", "enhanced_notes"):
        raise ValueError("该Schema诊断只适用于字符串笔记模式")
    for index, (history, state, call) in enumerate(zip(histories, states, calls), 1):
        expected = local_payload(memory_messages(official, mode, memory, history, index, memory_format),
                                 MEMORY_MAX, True, memory_format, sampling_profile)
        if call["request"] != expected or call["purpose"] != f"memory_session_{index}":
            raise ValueError("记忆请求与官方提示词不一致，或含不应提供的材料")
        memory = validate_memory(verified_content(call), mode)
        if memory_format == "bounded-schema" and (len(memory) > 20 or any(len(note) > 120 for note in memory)):
            raise ValueError("诊断记忆超出声明预算")
        if (state["memory"] != memory or state["session_index"] != index
                or state["conversation_id"] != history["conversation_id"]
                or state["prior_raw_histories_supplied"] != 0):
            raise ValueError("记忆检查点与原始响应不一致")
    answer_call = calls[-1]
    expected = local_payload(official.answer_prompt(mode, memory, case["user_question"]),
                             ANSWER_MAX, False, memory_format, sampling_profile)
    if answer_call["request"] != expected or answer_call["purpose"] != "answer":
        raise ValueError("回答请求与官方提示词不一致，或含旧对话/评分标准")
    if verified_content(answer_call) != record.get("answer"):
        raise ValueError("回答与原始响应不一致")
    return memory, record["answer"]


def verified_content(call):
    raw = call["raw_response"]
    if (call.get("ok") is not True or call.get("http_status") != 200
            or strict_json(call["raw_http_body"]) != raw
            or raw.get("done") is not True or raw.get("done_reason") != "stop"
            or raw.get("message", {}).get("tool_calls")
            or not isinstance(raw.get("message", {}).get("content"), str)
            or not raw["message"]["content"].strip()
            or raw["message"]["content"] != call.get("content")):
        raise ValueError("本机模型响应缺失、截断或与保存正文不一致")
    return call["content"]

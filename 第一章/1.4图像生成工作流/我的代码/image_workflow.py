"""实验1.4共用函数：DeepSeek提示词改写与万相文生图。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

REWRITE_SYSTEM_PROMPT = """你是图像生成提示词改写器。请把用户的中文需求改写成适合文生图模型的详细提示词。

必须遵守：
1. 完整保留用户明确指定的主体、场景、风格和文案，不得删改。
2. 可以补充构图、光线、镜头和材质，但不能加入与原需求冲突的内容。
3. 指定文案必须原样保留在prompt中，绝不能放进negative_prompt。
4. prompt以英文逗号分隔的描述为主；用户指定的中文文案保持中文原文。
5. prompt和negative_prompt均不超过500个字符。
6. 只输出一个JSON对象，不要使用Markdown围栏。

JSON结构：
{
  "prompt": "改写后的正向提示词",
  "negative_prompt": "不希望出现的内容",
  "preserved_constraints": ["保留的明确要求"],
  "added_details": ["为了具象化而增加的细节"]
}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_status() -> dict[str, Any]:
    """只返回配置是否存在，不读取或打印密钥内容。"""
    return {
        "deepseek_key": bool(os.getenv("DEEPSEEK_API_KEY")),
        "dashscope_key": bool(os.getenv("DASHSCOPE_API_KEY")),
        "deepseek_model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        "wanx_model": os.getenv("WANX_MODEL", "wan2.2-t2i-flash"),
        "dashscope_base_url": os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
        ),
        "image_size": os.getenv("WANX_SIZE", "1024*1024"),
    }


def require_config() -> dict[str, Any]:
    status = config_status()
    missing = []
    if not status["deepseek_key"]:
        missing.append("DEEPSEEK_API_KEY")
    if not status["dashscope_key"]:
        missing.append("DASHSCOPE_API_KEY")
    if missing:
        raise RuntimeError(f"缺少环境变量：{', '.join(missing)}")
    return status


def parse_json_object(text: str) -> dict[str, Any]:
    """兼容纯JSON与被代码围栏包裹的JSON。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("模型输出中没有JSON对象")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("模型输出的JSON不是对象")
    for field in ("prompt", "negative_prompt", "preserved_constraints", "added_details"):
        if field not in value:
            raise ValueError(f"模型输出缺少字段：{field}")
    if not isinstance(value["prompt"], str) or not value["prompt"].strip():
        raise ValueError("prompt必须是非空字符串")
    if not isinstance(value["negative_prompt"], str):
        raise ValueError("negative_prompt必须是字符串")
    if not isinstance(value["preserved_constraints"], list):
        raise ValueError("preserved_constraints必须是列表")
    if not isinstance(value["added_details"], list):
        raise ValueError("added_details必须是列表")
    value["prompt"] = value["prompt"].strip()
    value["negative_prompt"] = value["negative_prompt"].strip()
    if len(value["prompt"]) > 500:
        raise ValueError("prompt超过万相2.2的500字符限制")
    if len(value["negative_prompt"]) > 500:
        raise ValueError("negative_prompt超过万相2.2的500字符限制")
    return value


def rewrite_prompt(requirement: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """节点1：用DeepSeek把口语化需求改写为结构化文生图提示词。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "180"))
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": requirement},
    ]

    started = time.monotonic()
    response = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout).chat.completions.create(
        model=model,
        messages=messages,
    )
    message = response.choices[0].message
    raw_output = message.content or ""
    rewritten = parse_json_object(raw_output)
    reasoning = getattr(message, "reasoning_content", None)
    record = {
        "provider": "deepseek",
        "model": model,
        "response_id": response.id,
        "finish_reason": response.choices[0].finish_reason,
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "usage": response.usage.model_dump() if response.usage else {},
        "reasoning_content_present": bool(reasoning),
        "reasoning_content_length": len(reasoning or ""),
        "raw_output": raw_output,
    }
    return rewritten, record


def generate_image(
    prompt: str,
    negative_prompt: str,
    seed: int,
) -> tuple[bytes, dict[str, Any]]:
    """节点2：提交万相异步任务，轮询完成后立即下载图片。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
    ).rstrip("/")
    model = os.getenv("WANX_MODEL", "wan2.2-t2i-flash")
    size = os.getenv("WANX_SIZE", "1024*1024")
    poll_interval = float(os.getenv("IMAGE_POLL_INTERVAL", "3"))
    poll_timeout = float(os.getenv("IMAGE_POLL_TIMEOUT", "180"))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        # 使用哪个万相模型
        "model": model,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        },
        "parameters": {
            # 图片尺寸
            "size": size,
            # 每次只生成一张
            "n": 1,
            # 固定数字，尽量减少随机差异
            "seed": seed,
            # 两条路线都关闭服务端二次改写，只比较我们控制的改写节点。
            # 关闭万相内部的提示词改写
            "prompt_extend": False,
            # 不添加“AI生成水印”
            "watermark": False,
        },
    }

    submit_url = f"{base_url}/services/aigc/text2image/image-synthesis"
    started = time.monotonic()
    submit_response = requests.post(
        submit_url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    submit_body = submit_response.json()
    if submit_response.status_code != 200 or "output" not in submit_body:
        raise RuntimeError(
            f"万相任务提交失败 HTTP {submit_response.status_code}: {submit_body}"
        )
    task_id = submit_body["output"].get("task_id")
    if not task_id:
        raise RuntimeError(f"万相响应中没有task_id：{submit_body}")

    poll_url = f"{base_url}/tasks/{task_id}"
    deadline = time.monotonic() + poll_timeout
    final_body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        poll_response = requests.get(poll_url, headers=headers, timeout=30)
        final_body = poll_response.json()
        status = final_body.get("output", {}).get("task_status")
        if status == "SUCCEEDED":
            break
        if status in {"FAILED", "CANCELED", "UNKNOWN"}:
            raise RuntimeError(f"万相任务结束但未成功：{final_body}")
    else:
        raise TimeoutError(f"等待万相生成图片超过{poll_timeout:.0f}秒")

    output = final_body["output"]
    result = output["results"][0]
    # 生成完成后，服务器返回临时图片网址
    image_url = result["url"]
    # 程序再次发送GET请求
    download = requests.get(image_url, timeout=60)
    download.raise_for_status()
    # image_bytes 是图片的二进制数据，人无法直接阅读，但可以写入.png文件
    image_bytes = download.content
    record = {
        "provider": "dashscope",
        "model": model,
        "submit_request_id": submit_body.get("request_id"),
        "poll_request_id": final_body.get("request_id"),
        "task_id": task_id,
        "task_status": output.get("task_status"),
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "usage": final_body.get("usage", {}),
        "actual_prompt": result.get("actual_prompt"),
        "request": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "size": size,
            "n": 1,
            "seed": seed,
            "prompt_extend": False,
            "watermark": False,
        },
        "image": {
            "bytes": len(image_bytes),
            "sha256": sha256_bytes(image_bytes),
            "content_type": download.headers.get("Content-Type", "image/png"),
        },
    }
    return image_bytes, record


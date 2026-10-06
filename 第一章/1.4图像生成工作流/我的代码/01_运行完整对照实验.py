"""实验1.4：同一生图模型对比原始提示词与DeepSeek改写提示词。"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from image_workflow import config_status, generate_image, require_config, rewrite_prompt


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = EXPERIMENT_DIR / "运行结果"

# 一个具体需求检查“忠实保留”，一个宽泛需求检查“具象化增益”。
REQUIREMENTS = [
    {
        "id": "headphone-poster",
        "category": "具体需求",
        "text": '帮我做一张新款降噪耳机的产品海报，主打“深夜独处也清净”这句文案，风格简约高级',
        "seed": 20261001,
        "checkpoints": ["降噪耳机", "产品海报", "深夜独处也清净", "简约高级"],
    },
    {
        "id": "agi-programmer",
        "category": "宽泛需求",
        "text": "帮我画一个AGI实现以后程序员的工作场景",
        "seed": 20261002,
        "checkpoints": ["AGI实现以后", "程序员", "工作场景", "是否形成清楚的叙事"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验1.4：生成2个需求×2条路线的4张对照图。"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查配置和实验规模，不发送API请求",
    )
    return parser.parse_args()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(run_dir: Path, evidence: dict[str, Any]) -> None:
    rows = []
    for item in evidence["requirements"]:
        for route in item.get("routes", []):
            image = route.get("image_path")
            image_link = f"![{route['route_name']}]({image})" if image else "生成失败"
            rows.append(
                f"| {item['category']} | {item['text']} | {route['route_name']} | "
                f"{image_link} | 待观察 | 待观察 |"
            )

    report = f"""# 实验1.4结果记录

运行时间：{evidence['created_at']}

实验控制：两条路线使用同一个`{evidence['image_model']}`、同一需求使用相同seed，且都关闭万相内部的`prompt_extend`。这样主要比较变量就是是否经过DeepSeek改写。

## 路线

- 原始提示词：用户中文需求直接交给万相。
- DeepSeek改写：先保留明确要求并补充画面细节，再交给同一个万相模型。

## 图片对照

| 类别 | 原始需求 | 路线 | 图片 | 明确要求是否保留 | 新增细节是否有帮助 |
|---|---|---|---|---|---|
{chr(10).join(rows)}

## 观察时回答

1. 具体需求中，耳机、海报、指定中文文案和简约高级风格分别有没有出现？
2. 如果某项缺失，它是在DeepSeek改写时丢失，还是提示词保留了但万相没有画出来？
3. 宽泛需求中，改写路线补充了什么场景和叙事？这些补充是否符合你的想法？
4. 哪条路线更忠实？哪条路线更有表现力？

## 当前结论

运行并查看四张图片后填写。
"""
    (run_dir / "实验报告.md").write_text(report, encoding="utf-8")


def build_html(run_dir: Path, evidence: dict[str, Any]) -> None:
    cards = []
    for item in evidence["requirements"]:
        route_cards = []
        for route in item.get("routes", []):
            image_path = html.escape(route.get("image_path", ""))
            prompt = html.escape(route.get("prompt_used", ""))
            if image_path:
                body = f'<img src="{image_path}" alt="{html.escape(route["route_name"])}">'
            else:
                body = f'<div class="error">{html.escape(route.get("error", "生成失败"))}</div>'
            route_cards.append(
                f'<article><h3>{html.escape(route["route_name"])}</h3>{body}'
                f'<details><summary>查看实际提示词</summary><p>{prompt}</p></details></article>'
            )
        cards.append(
            f'<section><h2>{html.escape(item["category"])}：{html.escape(item["text"])}</h2>'
            f'<div class="grid">{"".join(route_cards)}</div></section>'
        )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>实验1.4图片对照</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f5f2;color:#20201e}}
section{{margin:30px 0 48px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
article{{background:white;padding:16px;border-radius:14px;box-shadow:0 4px 18px #00000012}}
img{{width:100%;display:block;border-radius:10px}} details{{margin-top:12px}} p{{line-height:1.6;word-break:break-word}}
.error{{padding:40px;background:#ffe8e8;color:#8a1717}} @media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><h1>实验1.4：原始提示词与DeepSeek改写对照</h1>{''.join(cards)}</body></html>"""
    (run_dir / "对照查看.html").write_text(page, encoding="utf-8")


def print_check() -> None:
    status = config_status()
    print("实验1.4配置检查（不会发送API请求）")
    print("DeepSeek密钥：", "已配置" if status["deepseek_key"] else "未配置")
    print("DashScope密钥：", "已配置" if status["dashscope_key"] else "未配置")
    print("改写模型：", status["deepseek_model"])
    print("生图模型：", status["wanx_model"])
    print("生图地址：", status["dashscope_base_url"])
    print("图片尺寸：", status["image_size"])
    print("预计生成：4张图（2个需求×2条路线）")
    print("按2026-10-06公开原价估算：约0.56元，不含少量文本改写费用。")


def main() -> int:
    args = parse_args()
    if args.check:
        print_check()
        return 0

    try:
        status = require_config()
    except RuntimeError as exc:
        print("配置错误：", exc)
        print("请先在项目根目录.env中填写DASHSCOPE_API_KEY，再运行本文件。")
        return 2

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_id
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    evidence: dict[str, Any] = {
        "experiment_id": "1-4",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "design": "same_image_model_original_vs_deepseek_rewrite",
        "rewrite_model": status["deepseek_model"],
        "image_model": status["wanx_model"],
        "prompt_extend": False,
        "image_count_planned": 4,
        "credential_values_recorded": False,
        "requirements": [],
    }

    print("实验1.4：原始提示词与DeepSeek改写提示词对照。")
    print("生图模型：", status["wanx_model"])
    print("计划生成4张图片；同一需求的两条路线使用相同seed。")

    success_count = 0
    for index, requirement in enumerate(REQUIREMENTS, start=1):
        print(f"\n[{index}/2] {requirement['category']}：{requirement['text']}")
        item: dict[str, Any] = {**requirement, "rewrite": None, "routes": []}

        try:
            print("  先让DeepSeek改写提示词……")
            rewritten, rewrite_record = rewrite_prompt(requirement["text"])
            item["rewrite"] = rewritten
            item["rewrite_call"] = rewrite_record
            print("  改写完成，保留要求：", "、".join(map(str, rewritten["preserved_constraints"])))
        except Exception as exc:
            item["rewrite_error"] = f"{type(exc).__name__}: {exc}"
            rewritten = None
            print("  改写失败：", exc)

        routes = [
            ("original", "原始提示词", requirement["text"], ""),
        ]
        if rewritten:
            routes.append(
                (
                    "rewritten",
                    "DeepSeek改写",
                    rewritten["prompt"],
                    rewritten["negative_prompt"],
                )
            )

        for route_id, route_name, prompt, negative_prompt in routes:
            print(f"  生成{route_name}图片……")
            route_record: dict[str, Any] = {
                "route": route_id,
                "route_name": route_name,
                "prompt_used": prompt,
                "negative_prompt_used": negative_prompt,
                "seed": requirement["seed"],
            }
            try:
                image_bytes, call_record = generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=requirement["seed"],
                )
                filename = f"{requirement['id']}_{route_id}.png"
                (image_dir / filename).write_bytes(image_bytes)
                route_record["image_path"] = f"images/{filename}"
                route_record["call"] = call_record
                route_record["error"] = None
                success_count += 1
                print(f"  已保存：images/{filename}")
            except Exception as exc:
                route_record["image_path"] = None
                route_record["error"] = f"{type(exc).__name__}: {exc}"
                print("  生成失败：", exc)
            item["routes"].append(route_record)

        evidence["requirements"].append(item)
        save_json(run_dir / "evidence.json", evidence)

    evidence["image_count_succeeded"] = success_count
    evidence["completed"] = success_count == 4
    save_json(run_dir / "evidence.json", evidence)
    build_report(run_dir, evidence)
    build_html(run_dir, evidence)

    print(f"\n生成成功：{success_count}/4")
    print("证据：", run_dir / "evidence.json")
    print("对照页：", run_dir / "对照查看.html")
    print("实验记录：", run_dir / "实验报告.md")
    if success_count == 4:
        print("下一步打开对照页观察四张图片，再根据明确要求逐项评分。")
        return 0
    print("部分运行失败，请根据evidence.json中的error排查后重试。")
    return 1


if __name__ == "__main__":
    sys.exit(main())


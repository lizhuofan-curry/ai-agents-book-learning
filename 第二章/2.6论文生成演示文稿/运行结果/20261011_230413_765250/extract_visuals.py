# -*- coding: utf-8 -*-
"""从 attention-is-all-you-need.pdf 裁剪原论文图表。

流程:pdftoppm 以 300 DPI 渲染指定页 -> PIL 按像素坐标裁剪 -> source_visuals/*.png
裁剪坐标记录在 source_visuals/extract_info.json,便于复核来源。
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
PDF = Path(r"E:/AI_Agents_Book/第二章/2.6论文生成演示文稿/材料/attention-is-all-you-need.pdf")
OUT = ROOT / "source_visuals"
TMP = ROOT / "tmp_pages"
DPI = 300

# pdf_page 从 1 起算;crop_box 为该页 300 DPI 渲染图上的像素坐标 (left, top, right, bottom)
TARGETS = [
    {
        "file": "figure1_transformer_architecture.png",
        "pdf_page": 3,
        "label": "Figure 1",
        "caption": "The Transformer - model architecture.",
        "crop_box": (740, 260, 1900, 1740),
    },
    {
        "file": "figure2_attention_mechanisms.png",
        "pdf_page": 4,
        "label": "Figure 2",
        "caption": "(left) Scaled Dot-Product Attention. (right) Multi-Head Attention "
                   "consists of several attention layers running in parallel.",
        "crop_box": (440, 250, 2180, 1235),
    },
    {
        "file": "table1_complexity_comparison.png",
        "pdf_page": 6,
        "label": "Table 1",
        "caption": "Maximum path lengths, per-layer complexity and minimum number of "
                   "sequential operations for different layer types.",
        "crop_box": (450, 265, 2300, 800),
    },
    {
        "file": "table2_bleu_results.png",
        "pdf_page": 8,
        "label": "Table 2",
        "caption": "The Transformer achieves better BLEU scores than previous "
                   "state-of-the-art models on the English-to-German and "
                   "English-to-French newstest2014 tests at a fraction of the training cost.",
        "crop_box": (450, 265, 2300, 1040),
    },
]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)
    pages = sorted({t["pdf_page"] for t in TARGETS})
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(DPI), "-f", str(pages[0]), "-l", str(pages[-1]),
         str(PDF), str(TMP / "hi")],
        check=True,
    )
    info = []
    for t in TARGETS:
        src = TMP / f"hi-{t['pdf_page']:02d}.png"
        with Image.open(src) as im:
            page_size = im.size
            crop = im.crop(t["crop_box"])
            dst = OUT / t["file"]
            crop.save(dst)
        info.append({
            "file": f"source_visuals/{t['file']}",
            "pdf_page": t["pdf_page"],
            "label": t["label"],
            "caption": t["caption"],
            "render_dpi": DPI,
            "page_image_size_px": list(page_size),
            "crop_box_px": list(t["crop_box"]),
            "output_size_px": list(crop.size),
        })
        print(f"{t['label']}: {dst.name} {crop.size}")
    (OUT / "extract_info.json").write_text(
        json.dumps({"source_pdf": str(PDF), "dpi": DPI, "items": info},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

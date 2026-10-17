"""3-4的基础步骤：人工二维向量、余弦排序与精确近邻；无模型或网络。"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
from typing import List

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCES = EXPERIMENT / "官方资料"
UPSTREAM = SOURCES / "upstream"
FIXTURE = EXPERIMENT / "测试材料/二维向量示例.json"
COMMIT = "dbc046eb896ac4e39aa19c7774c8bf49583b89a6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_materials():
    manifest = json.loads((SOURCES / "manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != COMMIT:
        raise ValueError("官方版本变化")
    for item in manifest["files"]:
        path = (UPSTREAM / item["path"]).resolve()
        if not path.is_relative_to(UPSTREAM.resolve()) or sha(path) != item["sha256"]:
            raise ValueError(f"官方材料哈希不符：{item['path']}")
    source = UPSTREAM / "chapter3/dense-embedding/cli.py"
    nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name == "dense_rank"]
    if len(nodes) != 1:
        raise ValueError("官方dense_rank定义缺失或重复")
    # 只编译已校验文件中的纯排序函数，不执行模块顶层依赖或模型加载。
    namespace = {"np": np, "List": List}
    isolated = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), str(source), "exec"), namespace)
    sample = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return manifest, sample, namespace["dense_rank"]


def normalize(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("向量必须为有限数值组成的二维数组")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("零向量没有定义余弦方向，不能直接检索")
    return array / norms


def calculate(sample, dense_rank):
    ids = [item["id"] for item in sample["vectors"]]
    if len(set(ids)) != len(ids) or not 1 <= sample["top_k"] <= len(ids):
        raise ValueError("向量ID重复或top_k越界")
    original = np.asarray([item["vector"] for item in sample["vectors"]], dtype=np.float64)
    matrix = normalize(original)
    query = normalize([sample["query"]])[0]
    if matrix.shape[1] != query.shape[0] or matrix.shape[1] != 2:
        raise ValueError("本可视化只接受查询与底库一致的二维向量")
    similarities = matrix @ query
    order, official_similarities = dense_rank(query, matrix)
    if not np.allclose(similarities, official_similarities, rtol=1e-12, atol=1e-12):
        raise ValueError("官方排序函数与本地相似度计算不一致")
    # 与逐项定义直接算余弦核对，避免漏归一化后把点积当余弦。
    reference = np.asarray([np.dot(row, sample["query"]) /
                            (np.linalg.norm(row) * np.linalg.norm(sample["query"])) for row in original])
    if not np.allclose(similarities, reference, rtol=1e-12, atol=1e-12):
        raise ValueError("余弦定义核对不通过")
    rows = [{"rank": rank, "id": ids[i], "original_vector": original[i].tolist(),
             "normalized_vector": matrix[i].tolist(), "cosine_similarity": float(similarities[i]),
             "cosine_distance": float(1.0 - similarities[i])} for rank, i in enumerate(order, 1)]
    return {"query_original": sample["query"], "query_normalized": query.tolist(),
            "rows": rows, "top_k": sample["top_k"], "top_ids": [row["id"] for row in rows[:sample["top_k"]]],
            "cosine_definition_crosscheck": True}


def make_svg(result):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 430">',
             '<rect width="500" height="430" fill="white"/>',
             '<text x="30" y="30" font-size="18" font-family="sans-serif">Unit vectors and query direction</text>',
             '<circle cx="230" cy="225" r="150" fill="none" stroke="#d7dce3"/>',
             '<path d="M55 225H415 M230 400V55" stroke="#9aa5b5" fill="none"/>',
             '<path d="M230 225H360L350 219M360 225L350 231" stroke="#d34c44" stroke-width="3" fill="none"/>',
             '<text x="285" y="251" fill="#d34c44" font-size="13" font-family="sans-serif">query q=(1,0)</text>']
    for row in result["rows"]:
        x, y = row["normalized_vector"]
        px, py = 230 + 150 * x, 225 - 150 * y
        label_y = py + 18 if row["id"] == "A" else py - 9
        parts.append(f'<line x1="230" y1="225" x2="{px:.3f}" y2="{py:.3f}" stroke="#cbd5e1"/>')
        parts.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="5" fill="#377dba"/>')
        parts.append(f'<text x="{px + 8:.3f}" y="{label_y:.3f}" font-family="sans-serif" font-size="14">{html.escape(row["id"])}</text>')
    parts.append('<text x="30" y="415" font-size="12" font-family="sans-serif" fill="#556070">Manual coordinates only; no text embeddings or ANN index.</text></svg>')
    return "\n".join(parts)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    manifest, sample, ranker = load_materials()
    print("3-4基础步骤：6个人工二维向量，归一化后计算余弦相似度，精确排序取前3。")
    print("模型请求0，网络请求0；不下载模型或安装新依赖。")
    if args.preview:
        print("DENSE_DISTANCE_PREVIEW_ONLY：仅检查材料，不创建运行目录。")
        return 0
    result = calculate(sample, ranker)
    directory = EXPERIMENT / "运行结果" / ("exact_vectors_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    image = make_svg(result)
    (directory / "向量方向.svg").write_text(image, encoding="utf-8")
    record = {"stage": "chapter3_dense_distance_foundation", "status": "exact_vector_demo_saved",
              "created_at": datetime.now(timezone.utc).isoformat(), "upstream_commit": COMMIT,
              "official_manifest_sha256": sha(SOURCES / "manifest.json"), "script_sha256": sha(__file__),
              "fixture_sha256": sha(FIXTURE), "sample": sample, "numpy_version": np.__version__,
              "calculation": result, "vector_plot": {"path": str(directory / "向量方向.svg"),
                                                        "sha256": sha(directory / "向量方向.svg")},
              "model_requests": 0, "network_requests": 0,
              "scope": "仅人工向量余弦距离及精确检索基础，不是文本语义验证、ANN对比或整个3-4完成"}
    dump(directory / "record.json", record)
    print("查询向量：", sample["query"])
    print("排名  向量ID  余弦相似度  余弦距离")
    for row in result["rows"]:
        print(f'{row["rank"]:>2}    {row["id"]:<4}   {row["cosine_similarity"]:>9.6f}   {row["cosine_distance"]:>9.6f}')
    print("Top-3：", "、".join(result["top_ids"]))
    print("相似度越大、余弦距离越小，方向越接近；这里只比较坐标，不判断文本含义。")
    print("EXACT_VECTOR_DEMO_SAVED：仅基础步骤完成，后续再做嵌入与索引比较。")
    print("向量图：", directory / "向量方向.svg")
    print("记录路径：", directory / "record.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

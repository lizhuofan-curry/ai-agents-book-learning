"""官方12篇英文文档、5条查询：MiniLM真实嵌入、精确检索和同向量ANN对照。"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Set

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
from project_paths import path_key_value, project_path

MODEL_MANIFEST = ROOT / "官方资料/MiniLM模型清单.json"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json",
               "1_Pooling/config.json", "modules.json", "README.md"}
K, DIMENSION = 5, 384
# 环境变量仅作用于本进程；正式实验只读已核验的本地文件。
os.environ["HF_HOME"] = str(PROJECT / ".uv-cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_sources():
    helper_path = ROOT / "我的代码/02_合成向量ANNOY与HNSW对照.py"
    baseline_path = ROOT / "安装记录/python_environment_before_install_20261017.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    key = helper_path.relative_to(PROJECT).as_posix()
    if sha(helper_path) != path_key_value(baseline["source_hashes"], key):
        raise ValueError("02辅助入口发生变化，停止使用未核对代码")
    spec = importlib.util.spec_from_file_location("pinned_ann_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.verify_sources()
    _, problems = helper.dependencies()
    if problems:
        raise ValueError(problems)
    source = helper.UPSTREAM / "chapter3/dense-embedding/cli.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    materials = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"DEFAULT_CORPUS", "DEFAULT_LABELS"}:
                materials[node.target.id] = ast.literal_eval(node.value)
    corpus, labels = materials["DEFAULT_CORPUS"], materials["DEFAULT_LABELS"]
    ids = [d["doc_id"] for d in corpus]
    if len(corpus) != 12 or len(labels) != 5 or len(set(ids)) != 12:
        raise ValueError("官方语料规模或ID发生变化")
    if any(not relevant or not set(relevant) <= set(ids) for relevant in labels.values()):
        raise ValueError("相关文档标注无效")
    names = {"DenseEncoder", "dense_rank", "_metrics_for_query"}
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(selected) != len(names):
        raise ValueError("官方编码或评分定义异常")
    namespace = {"np": np, "Dict": Dict, "List": List, "Set": Set}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(source), "exec"), namespace)
    return helper, corpus, labels, namespace


def load_model_files():
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if (manifest["model_id"], manifest["revision"], manifest["status"]) != (MODEL_ID, REVISION, "downloaded_verified"):
        raise ValueError("模型版本或下载状态不符")
    directory = project_path(manifest["local_path"])
    if not directory.is_relative_to((PROJECT / ".uv-cache/huggingface").resolve()):
        raise ValueError("模型必须位于当前项目模型缓存")
    if {f["name"] for f in manifest["files"]} != MODEL_FILES:
        raise ValueError("模型文件集合发生变化")
    for item in manifest["files"]:
        path = (directory / item["name"]).resolve()
        if not path.is_relative_to(directory) or sha(path) != item["sha256"]:
            raise ValueError(f"模型文件校验失败：{item['name']}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config["hidden_size"] != DIMENSION or config["num_hidden_layers"] != 6:
        raise ValueError("MiniLM模型结构不符")
    return directory, manifest


def validate_vectors(vectors, count):
    if (vectors.shape != (count, DIMENSION) or vectors.dtype != np.float32 or
            not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6)):
        raise ValueError("嵌入形状、数值、精度或归一化不符")


def ann_results(index, queries, exact_ids, ids):
    rows = []
    for i, q in enumerate(queries):
        found, distances = index.search(q, K)
        if len(found) != K or len(set(found)) != K or not set(found) <= set(ids):
            raise ValueError("索引返回不足、重复或未知文档")
        if len(distances) != K or not np.isfinite(distances).all():
            raise ValueError("索引距离无效")
        rows.append({"query_index": i, "doc_ids": found, "distances": list(map(float, distances)),
                     "index_recall_at_5": len(set(found) & set(exact_ids[i])) / K})
    return {"mean_index_recall_at_5": float(np.mean([r["index_recall_at_5"] for r in rows])), "queries": rows}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="校验文件和协议，不加载权重、不执行前向")
    args = parser.parse_args(argv)
    helper, corpus, labels, source = load_sources()
    model_path, model_manifest = load_model_files()
    print("3-4真实文本：官方12篇英文文档、5条查询、Top-5。")
    print("嵌入模型：", MODEL_ID, "；固定版本：", REVISION)
    print("CPU float32，掩码平均池化+L2归一化，384维；最大长度256 token。")
    print("预计本机嵌入前向2批（12篇文档+5条查询）；生成式模型请求0，网络请求0。")
    print("标签质量指标与索引近邻召回分开；无云端回退、自动下载或自动重试。")
    if args.preview:
        print("TEXT_EMBEDDING_PREVIEW_ONLY：材料核验通过，模型前向0批，不创建运行目录。")
        return 0

    directory = ROOT / "运行结果" / ("text_embedding_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    record = {"stage": "chapter3_real_text_embedding", "status": "running",
              "created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
              "helper_script_sha256": sha(ROOT / "我的代码/02_合成向量ANNOY与HNSW对照.py"),
              "upstream_commit": helper.COMMIT, "official_manifest_sha256": sha(helper.MANIFEST),
              "model": {"id": MODEL_ID, "revision": REVISION, "manifest_sha256": sha(MODEL_MANIFEST),
                        "path": str(model_path), "file_count": len(model_manifest["files"])},
              "protocol": {"top_k": K, "dimension": DIMENSION, "pooling": "mean", "normalize": True,
                           "device": "cpu", "dtype": "float32", "max_length": 256, "batch_size": 16,
                           "cpu_threads": 4, "annoy_trees": 10, "annoy_search_k": "default=50",
                           "hnsw_M": 16, "hnsw_ef_construction": 100, "hnsw_ef_search": 20,
                           "hnsw_threads": 1, "index_random_seeds": "native defaults, not overridden",
                           "mrr_scope": "official full 12-document ranking; not MRR@5"},
              "corpus": corpus, "labels": labels,
              "embedding_forward_batches_attempted": 0, "embedding_forward_batches_completed": 0,
              "generated_text_requests": 0, "cloud_model_requests": 0, "network_requests": 0,
              "scope": "官方小型英文样例的真实嵌入检索；MiniLM替代路线，不是BGE-M3结果或通用排名"}
    path = directory / "record.json"
    dump(path, record)
    try:
        import torch
        torch.set_num_threads(4)
        record["packages"] = {name: importlib.metadata.version(name) for name in
                              ["torch", "transformers", "numpy", "safetensors", "annoy", "hnswlib"]}
        start = time.perf_counter()
        encoder = source["DenseEncoder"](str(model_path), pooling="mean", device="cpu", max_length=256)
        if next(encoder.model.parameters()).dtype != torch.float32:
            raise ValueError("模型参数不是float32")
        record["model_load_seconds"] = time.perf_counter() - start
        texts = [d["text"] for d in corpus]
        query_texts = list(labels)
        lengths = [len(encoder.tokenizer(t, truncation=False)["input_ids"]) for t in texts + query_texts]
        if max(lengths) > 256:
            raise ValueError("样例会发生截断，停止以保持协议清晰")
        record["token_lengths_with_special_tokens"] = lengths
        record["input_truncated"] = False
        arrays = {}
        record["arrays"] = {}
        for name, strings in (("documents", texts), ("queries", query_texts)):
            record["embedding_forward_batches_attempted"] += 1
            dump(path, record)
            print(f"正在编码{name}：{len(strings)}段文本……", flush=True)
            start = time.perf_counter()
            vectors = encoder.encode(strings, batch_size=16)
            record["embedding_forward_batches_completed"] += 1
            validate_vectors(vectors, len(strings))
            array_path = directory / (name + ".npy")
            np.save(array_path, vectors, allow_pickle=False)
            arrays[name] = vectors
            record["arrays"][name] = {"file": array_path.name, "shape": list(vectors.shape),
                                     "sha256": sha(array_path), "seconds": time.perf_counter() - start}
            dump(path, record)
        ids = [d["doc_id"] for d in corpus]
        exact, references = [], []
        for (query, relevant), q in zip(labels.items(), arrays["queries"]):
            order, scores = source["dense_rank"](q, arrays["documents"])
            ranked_ids = [ids[i] for i in order]
            references.append(ranked_ids[:K])
            metrics = source["_metrics_for_query"](ranked_ids, set(relevant), K)
            exact.append({"query": query, "relevant_doc_ids": relevant, "metrics": metrics,
                          "full_ranking": [{"rank": rank, "doc_id": ids[i], "title": corpus[i]["title"],
                                            "cosine_similarity": float(scores[i])}
                                           for rank, i in enumerate(order, 1)]})
        record["exact"] = {"queries": exact,
                           "macro": {name: float(np.mean([x["metrics"][key] for x in exact]))
                                     for name, key in (("semantic_recall_at_5", "recall"),
                                                       ("precision_at_5", "precision"), ("mrr_full_ranking", "rr"))}}
        dump(path, record)
        _, indexes = helper.load_official()
        record["indexes"] = {}
        for name in ("annoy", "hnsw"):
            if name == "annoy":
                index = indexes.AnnoyIndex(DIMENSION, n_trees=10, metric="angular")
            else:
                index = indexes.HNSWIndex(DIMENSION, max_elements=len(ids), ef_construction=100,
                                         M=16, ef_search=20, space="cosine")
                index.index.set_num_threads(1)
            for doc_id, vector in zip(ids, arrays["documents"]):
                index.add_item(doc_id, vector)
            if name == "annoy":
                index.rebuild_index()
            record["indexes"][name] = ann_results(index, arrays["queries"], references, ids)
            dump(path, record)
        record["status"] = "text_embedding_eval_saved"
        dump(path, record)
        for i, item in enumerate(exact):
            print(f'\n查询：{item["query"]}；官方相关文档：{item["relevant_doc_ids"]}')
            for row in item["full_ranking"][:K]:
                print(f'  {row["rank"]}. {row["doc_id"]}  cos={row["cosine_similarity"]:.4f}  {row["title"]}')
            print("  标签指标：", {k: item["metrics"][k] for k in ("recall", "precision", "rr")})
            for name, value in record["indexes"].items():
                row = value["queries"][i]
                print(f'  {name}: {row["doc_ids"]}；相对精确Top-5召回={row["index_recall_at_5"]:.1%}')
        print("\n精确检索对官方标注的宏平均：", record["exact"]["macro"])
        print("ANN相对精确Top-5的平均索引召回：",
              {name: value["mean_index_recall_at_5"] for name, value in record["indexes"].items()})
        print("TEXT_EMBEDDING_EVAL_SAVED：小型英文样例完成，指标与结果待核对，不预设通过分数。")
        print("记录路径：", path)
        return 0
    except Exception as exc:
        record.update(status="incomplete", error={"type": type(exc).__name__, "message": str(exc)})
        dump(path, record)
        print("本次未完成：", record["error"])
        print("记录路径：", path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

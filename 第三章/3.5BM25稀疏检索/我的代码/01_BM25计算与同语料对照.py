"""3-5：展示BM25计算，并与3-4保存的同语料稠密结果对照；无模型、无网络。"""

import argparse
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
from typing import Dict, List, Set


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
from project_paths import project_path

UPSTREAM = ROOT / "官方资料/upstream"
MANIFEST = ROOT / "官方资料/manifest.json"
FIXTURE = ROOT / "测试材料/同语料对照输入.json"
COMMIT = "dbc046eb896ac4e39aa19c7774c8bf49583b89a6"
K = 5
CHINA = timezone(timedelta(hours=8))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_materials():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["commit"] != COMMIT:
        raise ValueError("官方版本发生变化")
    for item in manifest["files"]:
        path = (UPSTREAM / item["path"]).resolve()
        if not path.is_relative_to(UPSTREAM.resolve()) or sha(path) != item["sha256"]:
            raise ValueError(f"官方源文件校验失败：{item['path']}")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    provenance = fixture["source_3_4"]
    path = project_path(provenance["record_relative_to_project"])
    if not path.is_relative_to((PROJECT / "第三章/3.4稠密向量检索/运行结果").resolve()):
        raise ValueError("稠密来源必须位于3-4运行结果")
    if sha(path) != provenance["record_sha256"] or sha(path.parent / "核对记录.json") != provenance["audit_sha256"]:
        raise ValueError("3-4来源记录或核对记录发生变化")
    dense = json.loads(path.read_text(encoding="utf-8"))
    audit = json.loads((path.parent / "核对记录.json").read_text(encoding="utf-8"))
    if dense["status"] != "text_embedding_eval_saved" or audit["record_sha256"] != sha(path):
        raise ValueError("来源不是完整且已核对的文本实验")
    if fixture["corpus"] != dense["corpus"] or fixture["labels"] != dense["labels"]:
        raise ValueError("文档、查询或标注未与3-4对齐")
    if len(fixture["corpus"]) != 12 or len(fixture["labels"]) != 5:
        raise ValueError("本步固定12篇文档和5条共享查询")
    for item in dense["arrays"].values():
        array_path = (path.parent / item["file"]).resolve()
        if not array_path.is_relative_to(path.parent) or sha(array_path) != item["sha256"]:
            raise ValueError("3-4保存的向量文件发生变化")
    return manifest, fixture, dense


def load_engine():
    path = UPSTREAM / "chapter3/sparse-embedding/bm25_engine.py"
    spec = importlib.util.spec_from_file_location("pinned_bm25_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logging.getLogger().setLevel(logging.WARNING)
    module.logger.setLevel(logging.WARNING)
    cli = UPSTREAM / "chapter3/sparse-embedding/cli.py"
    nodes = [n for n in ast.parse(cli.read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name == "_metrics_for_query"]
    if len(nodes) != 1:
        raise ValueError("官方指标定义缺失或重复")
    ns = {"Dict": Dict, "List": List, "Set": Set}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(cli), "exec"), ns)
    return module, ns["_metrics_for_query"]


def common_metrics(retrieved, relevant, k=K):
    if len(set(retrieved)) != len(retrieved) or not relevant or k <= 0:
        raise ValueError("检索ID重复、标注为空或k无效")
    top = retrieved[:k]
    hits = [doc for doc in top if doc in relevant]
    first = next((i for i, doc in enumerate(top, 1) if doc in relevant), None)
    return {"recall_at_5": len(hits) / len(set(relevant)), "precision_at_5_fixed_slots": len(hits) / k,
            "rr_at_5": 1 / first if first else 0.0, "hits": hits, "returned_at_5": len(top)}


def average(rows):
    return {out: sum(r[key] for r in rows) / len(rows)
            for out, key in (("recall_at_5", "recall_at_5"),
                             ("precision_at_5_fixed_slots", "precision_at_5_fixed_slots"),
                             ("mrr_at_5", "rr_at_5"))}


def build_engine(module, corpus, k1, b):
    engine = module.SparseSearchEngine()
    engine.index_batch([{"doc_id": d["doc_id"], "text": d["text"], "metadata": {"title": d["title"]}}
                        for d in corpus])
    # 等价于官方CLI的自定义语料构建入口，最后固定目标参数。
    engine.bm25 = module.BM25(engine.index, k1=k1, b=b)
    return engine


def explain_query(module, engine, query):
    processor = module.TextProcessor()
    terms = processor.tokenize(query)
    resolved, term_details = [], []
    for original in terms:
        term = original
        postings = engine.index.get_posting_list(term)
        # 保留官方检索中的小写回退规则，不添加词干化或同义词扩展。
        if not postings and term and not term[0].isdigit() and "-" not in term:
            lower = term.lower()
            if engine.index.get_posting_list(lower):
                term = lower
                postings = engine.index.get_posting_list(term)
        resolved.append(term)
        term_details.append({"original": original, "resolved": term, "df": len(postings),
                             "posting_doc_ids": [engine.internal_to_external[i] for i in sorted(postings)],
                             "raw_idf": engine.bm25.calculate_raw_idf(term),
                             "scoring_idf": engine.bm25.calculate_idf(term)})
    results = engine.search(query, top_k=engine.index.total_documents)
    rows = []
    for rank, result in enumerate(results, 1):
        internal = engine.external_to_internal[result["doc_id"]]
        dl = engine.index.doc_lengths[internal]
        contributions = []
        for term in resolved:
            tf = engine.index.term_frequency[internal].get(term, 0)
            idf = engine.bm25.calculate_idf(term)
            value = engine.bm25.calculate_term_score(term, internal)
            expected = (idf * tf * (engine.bm25.k1 + 1) /
                        (tf + engine.bm25.k1 * (1 - engine.bm25.b + engine.bm25.b * dl / engine.bm25.avgdl))) if tf else 0.0
            if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("逐词公式重算不符")
            contributions.append({"term": term, "tf": tf, "doc_length": dl, "scoring_idf": idf, "score": value})
        if not math.isclose(result["score"], sum(v["score"] for v in contributions), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("逐词贡献之和与官方排序分数不符")
        rows.append({"rank": rank, "doc_id": result["doc_id"], "title": result["metadata"]["title"],
                     "score": result["score"], "debug": result["debug"], "contributions": contributions})
    return {"query": query, "tokens_before_stopword_removal": processor.tokenize(query, remove_stop_words=False),
            "tokens": terms, "resolved_terms": resolved, "term_details": term_details,
            "full_matched_ranking": rows}


def print_explanation(label, row, n, avgdl):
    print(f'\n{label}：{row["query"]}；分词：{row["tokens"]}')
    print(f"N={n}，avgdl={avgdl:.4f}（分词去停用词后的平均长度）")
    for term in row["term_details"]:
        print(f'  词 {term["resolved"]!r} → {term["posting_doc_ids"]}；df={term["df"]}；'
              f'raw IDF={term["raw_idf"]:.4f}；评分IDF={term["scoring_idf"]:.4f}')
    for result in row["full_matched_ranking"][:K]:
        print(f'  #{result["rank"]} {result["doc_id"]} {result["title"]}；总分={result["score"]:.4f}')
        for term in result["contributions"]:
            print(f'    {term["term"]!r}：TF={term["tf"]}，dl={term["doc_length"]}，贡献={term["score"]:.4f}')
    if not row["full_matched_ranking"]:
        print("  没有词项匹配候选；保留空结果，不填入零分文档。")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args(argv)
    if not math.isfinite(args.k1) or args.k1 <= 0 or not math.isfinite(args.b) or not 0 <= args.b <= 1:
        parser.error("k1必须为正有限数，b必须在[0,1]")
    manifest, fixture, dense = load_materials()
    print("3-5 BM25：沿用3-4的12篇文档、5条查询与标注，Top-5。")
    print(f"k1={args.k1}，b={args.b}；展示分词、倒排索引、TF、DF、IDF及逐词贡献。")
    print("复用已核对的MiniLM结果，不重新编码；另有1条BM25编码查询展示，不计入5题对照。")
    print("模型请求0，嵌入前向0，网络请求0；不安装依赖、下载模型或启动服务。")
    print("对照统一Precision分母5、MRR仅看前5；官方按实际返回数的Precision另列，不直接混比。")
    if args.preview:
        print("BM25_PREVIEW_ONLY：19份官方资料和同语料来源核验通过，不建索引、不创建结果目录。")
        return 0
    directory = ROOT / "运行结果" / ("bm25_aligned_" + datetime.now(CHINA).strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "record.json"
    record = {"stage": "chapter3_bm25_aligned", "status": "running", "created_at": datetime.now(CHINA).isoformat(),
              "script_sha256": sha(__file__), "upstream_commit": COMMIT,
              "official_manifest_sha256": sha(MANIFEST), "fixture_sha256": sha(FIXTURE),
              "source_3_4": fixture["source_3_4"], "corpus": fixture["corpus"], "labels": fixture["labels"],
              "protocol": {"k1": args.k1, "b": args.b, "top_k": K, "corpus_count": 12, "shared_query_count": 5,
                           "additional_code_demo_queries": 1, "idf": "RSJ; df=0 -> 0, else max(raw_log_ratio, 1e-6)",
                           "tokenizer": "unmodified upstream TextProcessor; English stop words; no stemming",
                           "ties": "unmodified upstream candidate-set iteration; no manual tie reranking",
                           "common_precision_denominator": 5, "common_mrr_scope": "top 5",
                           "official_precision_denominator": "actual returned count up to 5",
                           "official_rr_scope": "all matched BM25 candidates"},
              "model_requests": 0, "embedding_forward_batches": 0, "network_requests": 0,
              "scope": "12文档5共享查询的自定义语料BM25对照；非官方默认10篇评测，非SPLADE或HTTP服务"}
    dump(path, record)
    try:
        module, official_metric = load_engine()
        engine = build_engine(module, fixture["corpus"], args.k1, args.b)
        processor = module.TextProcessor()
        record["index"] = {"statistics": engine.index.get_statistics(), "avgdl": engine.bm25.avgdl,
                           "stop_words": sorted(processor.stop_words),
                           "documents": [{"doc_id": doc["doc_id"], "tokens": processor.tokenize(doc["text"]),
                                          "length": engine.index.doc_lengths[engine.external_to_internal[doc["doc_id"]]]}
                                         for doc in fixture["corpus"]],
                           "postings": {term: [engine.internal_to_external[i] for i in sorted(engine.index.get_posting_list(term))]
                                        for term in sorted(engine.index.index)}}
        record["queries"] = []
        dump(path, record)
        dense_rows = {q["query"]: q for q in dense["exact"]["queries"]}
        for query, relevant in fixture["labels"].items():
            result = explain_query(module, engine, query)
            retrieved = [r["doc_id"] for r in result["full_matched_ranking"]]
            dense_ids = [r["doc_id"] for r in dense_rows[query]["full_ranking"]]
            result.update(relevant_doc_ids=relevant, common_bm25=common_metrics(retrieved, relevant),
                          common_dense=common_metrics(dense_ids, relevant), dense_top_5=dense_ids[:K],
                          official_cli_metrics=official_metric(retrieved, set(relevant), K))
            record["queries"].append(result)
            dump(path, record)
        record["common_macro"] = {"bm25": average([q["common_bm25"] for q in record["queries"]]),
                                  "dense_saved": average([q["common_dense"] for q in record["queries"]])}
        record["official_bm25_macro_separate"] = {
            key: sum(q["official_cli_metrics"][key] for q in record["queries"]) / len(record["queries"])
            for key in ("recall", "precision", "rr")}
        demo = fixture["code_demo"]
        record["code_demo"] = {**explain_query(module, engine, demo["query"]),
                               "relevant_doc_ids": demo["relevant_doc_ids"], "scope": demo["scope"]}
        record["status"] = "bm25_aligned_eval_saved"
        dump(path, record)
        explained = next(q for q in record["queries"] if q["query"] == "model distillation")
        print_explanation("逐词计算示例", explained, engine.index.total_documents, engine.bm25.avgdl)
        print("\n同一组5条查询（下列Precision固定除以5，RR只看前5）：")
        for q in record["queries"]:
            print(f'\n查询：{q["query"]}；相关标注：{q["relevant_doc_ids"]}；分词：{q["tokens"]}')
            print("  BM25：", [r["doc_id"] for r in q["full_matched_ranking"][:K]], q["common_bm25"])
            print("  稠密已存：", q["dense_top_5"], q["common_dense"])
        print("\n统一口径宏平均：", record["common_macro"])
        print("BM25官方口径另列（Precision按返回数、RR看全部匹配候选）：", record["official_bm25_macro_separate"])
        print_explanation("额外编码展示，仅BM25", record["code_demo"], engine.index.total_documents, engine.bm25.avgdl)
        print("BM25_ALIGNED_EVAL_SAVED：本次计算完成；指标和逐词贡献待核对，不预设结果分数。")
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

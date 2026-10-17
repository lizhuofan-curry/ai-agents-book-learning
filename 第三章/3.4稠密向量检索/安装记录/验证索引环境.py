"""安装后的真实小规模健康检查；不开展3000向量正式评估。"""

from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "安装记录"
CHECK = INSTALL / ("native_index_health_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ann_lesson", ROOT / "我的代码/02_合成向量ANNOY与HNSW对照.py")
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    before = json.loads((INSTALL / "python_environment_before_install_20261017.json").read_text(encoding="utf-8"))
    if lesson.sha(ROOT.parents[1] / "uv.lock") != before["uv_lock_sha256"]:
        raise ValueError("依赖锁发生变化")
    for file, digest in before["source_hashes"].items():
        # 快照路径以项目根为基准，无论从哪里启动都使用同一份文件。
        if lesson.sha(ROOT.parents[1] / file) != digest:
            raise ValueError(f"实验源文件发生变化：{file}")
    installed = {p.metadata["Name"].lower().replace("_", "-"): p.version
                 for p in importlib.metadata.distributions()}
    changes = {name: {"before": version, "after": installed.get(name)}
               for name, version in before["packages"].items() if installed.get(name) != version}
    if changes:
        raise ValueError(f"既有依赖版本变化：{changes}")
    versions, problems = lesson.dependencies()
    if problems:
        raise ValueError(problems)
    command = [r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
               "-products", "Microsoft.VisualStudio.Product.BuildTools", "-requires",
               "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
               "Microsoft.VisualStudio.Component.Windows11SDK.26100", "-format", "json", "-utf8"]
    response = subprocess.run(command, capture_output=True, check=True)
    instances = json.loads(response.stdout.decode("utf-8-sig"))
    if len(instances) != 1 or not instances[0].get("isComplete") or not instances[0].get("isLaunchable"):
        raise ValueError("C++工具或SDK尚未完整安装")
    compilers = list(Path(instances[0]["installationPath"]).glob("VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
    if not compilers:
        raise ValueError("找不到64位C++编译器")
    lesson.verify_sources()
    ranker, indexes = lesson.load_official()
    sample = json.loads((ROOT / "测试材料/二维向量示例.json").read_text(encoding="utf-8"))
    base = np.asarray([x["vector"] for x in sample["vectors"]], dtype="float32")
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    queries = np.asarray([sample["query"]], dtype="float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    protocol = {**lesson.PROTOCOL, "base_count": 6, "query_count": 1, "dimension": 2, "top_k": 3}
    record = {"status": "running", "checked_at": datetime.now(timezone.utc).isoformat(),
              "scope": "安装后的6条二维向量、1次查询原生健康检查；不是3000×128正式评估或文本嵌入结果",
              "verification_script_sha256": lesson.sha(__file__), "protocol": protocol,
              "upstream_commit": lesson.COMMIT, "official_manifest_sha256": lesson.sha(lesson.MANIFEST),
              "frozen_experiment_scripts_unchanged": True, "existing_packages_unchanged": True,
              "packages": versions, "new_packages": {k: v for k, v in installed.items() if k not in before["packages"]},
              "build_tools": {k: instances[0].get(k) for k in
                              ("instanceId", "installationPath", "installationVersion", "isComplete", "isLaunchable", "isRebootRequired")},
              "compiler_files": list(map(str, compilers)), "base": base.tolist(), "queries": queries.tolist(),
              "model_requests": 0, "network_requests_during_health_check": 0, "formal_benchmark_executed": False}
    lesson.dump(CHECK, record)
    try:
        def checkpoint(truths, seconds, results):
            record.update(exact_top_ids=truths, reference_seconds=seconds, results=results)
            lesson.dump(CHECK, record)

        results = lesson.measure(base, queries, ranker, indexes, protocol, checkpoint)
        for name, value in results.items():
            if value["queries"][0]["ids"] != ["v0", "v1", "v2"] or value["recall_at_k"] != 1:
                raise ValueError(f"{name}未找回已知Top-3")
            if value["health"] is not None and not value["health"]["all_found"]:
                raise ValueError(f"{name}自身向量查询失败")
        record["status"] = "native_index_health_passed"
        lesson.dump(CHECK, record)
        print("NATIVE_INDEX_HEALTH_PASSED：真实ANNOY/HNSW建索引和查询均通过。")
        print("依赖：", versions)
        print("仅安装健康检查；正式3000向量对照尚未运行。")
        print("记录：", CHECK)
        return 0
    except Exception as exc:
        record.update(status="incomplete", error={"type": type(exc).__name__, "message": str(exc)})
        lesson.dump(CHECK, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

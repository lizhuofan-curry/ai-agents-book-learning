"""只下载并核验固定版本的公开MiniLM文件，不执行模型前向。"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
CACHE = PROJECT / ".uv-cache/huggingface"
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import requests
from huggingface_hub import HfApi


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIRECTORY = CACHE / ("local_models/all-MiniLM-L6-v2_" + REVISION[:12])
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json",
         "1_Pooling/config.json", "modules.json", "README.md"]
MANIFEST = ROOT / "官方资料/MiniLM模型清单.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(path, meta):
    size = path.stat().st_size
    if size != meta.size:
        raise ValueError(f"{meta.rfilename}长度不符")
    if meta.lfs:
        if digest(path) != meta.lfs.sha256:
            raise ValueError(f"{meta.rfilename}的LFS SHA256不符")
    else:
        blob = hashlib.sha1(f"blob {size}\0".encode() + path.read_bytes()).hexdigest()
        if blob != meta.blob_id:
            raise ValueError(f"{meta.rfilename}的Git blob不符")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if MANIFEST.exists():
        raise ValueError("已有模型清单，先核对已有文件，不覆盖先前下载记录")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    record = {"model_id": MODEL, "revision": REVISION, "local_path": str(DIRECTORY),
              "status": "downloading", "created_at": datetime.now(timezone.utc).isoformat(),
              "files": [], "model_forward_batches": 0,
              "scope": "仅公开模型文件下载和完整性核验；不运行语义检索"}
    run_path = ROOT / "安装记录" / ("minilm_download_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")

    def save():
        run_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        info = HfApi(token=False).model_info(MODEL, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise ValueError("公开模型版本不一致")
        metadata = {f.rfilename: f for f in info.siblings}
        print("固定模型：", MODEL, REVISION, flush=True)
        print("下载总字节：", sum(metadata[f].size for f in FILES), flush=True)
        session = requests.Session()
        for name in FILES:
            meta = metadata[name]
            path = DIRECTORY / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                validate(path, meta)
            else:
                part = path.with_suffix(path.suffix + ".part")
                with session.get(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}",
                                 stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    with part.open("wb") as stream:
                        received, milestone = 0, 10 * 1024 * 1024
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            stream.write(chunk)
                            received += len(chunk)
                            if received >= milestone:
                                print(f"  {name}: {received}/{meta.size} bytes", flush=True)
                                milestone += 10 * 1024 * 1024
                validate(part, meta)
                part.replace(path)
            record["files"].append({"name": name, "bytes": path.stat().st_size, "sha256": digest(path),
                                     "git_blob_sha1": meta.blob_id,
                                     "upstream_lfs_sha256": meta.lfs.sha256 if meta.lfs else None})
            save()
            print("已核验：", name, flush=True)
        record.update(status="downloaded_verified", completed_at=datetime.now(timezone.utc).isoformat())
        save()
        MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("MINILM_FILES_VERIFIED：仅文件准备完成，模型前向0批。", flush=True)
        print("模型路径：", DIRECTORY, flush=True)
        return 0
    except Exception as exc:
        record.update(status="incomplete", error={"type": type(exc).__name__,
                      "message": re.sub(r"https?://\S+", "[URL omitted]", str(exc))})
        save()
        print("下载未完成：", record["error"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

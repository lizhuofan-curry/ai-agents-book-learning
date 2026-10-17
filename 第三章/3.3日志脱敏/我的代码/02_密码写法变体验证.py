"""只改变密码表达方式，复用已完成单条入口；最多一次本机推理。"""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


EXPERIMENT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = Path(__file__).with_name("01_单条日志规则与模型对照.py")
BASE_CODE_SHA256 = "2ac5dbbfe48098f00d7bd4dd49dec29beaa6ecdc35d43903cd1be1f1237fc8d7"
BASE_DIRECTORY = EXPERIMENT / "运行结果/single_log_20261017_191645_448058"
VARIANT_FIXTURE = EXPERIMENT / "测试材料/密码赋值写法变体.json"
OLD_PHRASE = "My password is demo_pass_42."
NEW_PHRASE = 'password="demo_pass_42".'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if sha(BASE_SCRIPT) != BASE_CODE_SHA256:
        raise ValueError("已运行的01源码发生变化，停止配对，不自动接受新协议")
    baseline_path = BASE_DIRECTORY / "record.json"
    audit = json.loads((BASE_DIRECTORY / "核对记录.json").read_text(encoding="utf-8"))
    if sha(baseline_path) != audit["record_sha256"]:
        raise ValueError("第一条已核验记录发生变化，停止配对")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["status"] != "single_sample_comparison_saved":
        raise ValueError("第一条对照未完成")
    spec = importlib.util.spec_from_file_location("chapter3_single_log_frozen", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, values, _, original = module.load_materials()
    if original != baseline["sample"] or module.make_payload(values, original["text"]) != baseline["calls"][0]["request"]:
        raise ValueError("原始样例或请求配置不再匹配第一条记录")
    expected = deepcopy(original)
    if expected["text"].count(OLD_PHRASE) != 1:
        raise ValueError("密码原句不唯一，不能作单处写法变体")
    expected["sample_id"] = "synthetic_pii_01_password_assignment"
    expected["text"] = expected["text"].replace(OLD_PHRASE, NEW_PHRASE, 1)
    variant = json.loads(VARIANT_FIXTURE.read_text(encoding="utf-8"))
    if variant != expected:
        raise ValueError("变体不仅改变了密码写法；不自动替换材料")
    module.FIXTURE = VARIANT_FIXTURE
    provenance = {"wrapper_path": str(Path(__file__).resolve()), "wrapper_sha256": sha(__file__),
                  "frozen_base_script_sha256": BASE_CODE_SHA256,
                  "baseline_record_path": str(baseline_path), "baseline_record_sha256": sha(baseline_path),
                  "variant_fixture_sha256": sha(VARIANT_FIXTURE),
                  "only_input_text_change": {"before": OLD_PHRASE, "after": NEW_PHRASE},
                  "gold_labels_and_debug_checks_unchanged": True,
                  "prompt_schema_model_sampling_budget_unchanged": True}
    return module, provenance


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    module, provenance = prepare()
    original_dump = module.dump

    def save_with_provenance(path, value):
        if Path(path).name == "record.json":
            value["variant_provenance"] = provenance
        original_dump(path, value)

    # 仅增加配对证据元数据；01原文件、请求构造和脱敏算法保持原样。
    module.dump = save_with_provenance
    print('写法变体：My password is ... → password="..."；其余文本、标注与模型配置一致。')
    return module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

"""CI检查：学习代码语法、已跟踪文件和离线脚本的实际输出。"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths = [Path(name) for name in tracked.decode("utf-8").split("\0") if name]
    if not paths:
        raise RuntimeError("No tracked project files to check")

    credential_pattern = re.compile(rb"(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,})")
    python_count = 0
    for relative in paths:
        if {".venv", ".uv-cache", ".idea", "__pycache__"}.intersection(relative.parts):
            raise RuntimeError(f"Local environment or cache tracked: {relative}")
        if relative.name.startswith(".env") and relative.name != ".env.example":
            raise RuntimeError(f"Local credentials file tracked: {relative}")
        content = (ROOT / relative).read_bytes()
        if credential_pattern.search(content):
            raise RuntimeError(f"Possible credential in tracked file: {relative}")
        if relative.suffix == ".py":
            ast.parse(content.decode("utf-8-sig"), filename=str(relative))
            python_count += 1

    for line in (ROOT / ".env.example").read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip().endswith("API_KEY") and value.strip():
            raise RuntimeError("API key template must leave credentials empty")

    # 在临时目录复制运行离线脚本，验证中文路径和实际JSON输出，不覆盖学习记录。
    temp_parent = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="ai-agent-ci-") as directory:
        sandbox = Path(directory).resolve()
        if not sandbox.is_relative_to(temp_parent):
            raise RuntimeError("Unexpected temporary directory")
        experiment = Path("第一章/1.2网络搜索Agent")
        relative_script = experiment / "我的代码/01_离线观察Agent循环.py"
        copied_script = sandbox / relative_script
        copied_script.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / relative_script, copied_script)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        subprocess.run(
            [sys.executable, str(copied_script)],
            cwd=sandbox, env=env, check=True, capture_output=True, timeout=30,
        )
        output = sandbox / experiment / "运行结果/demo.json"
        result = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(result.get("question"), str) or not result["question"]:
            raise RuntimeError("Offline output has no question")
        if not isinstance(result.get("answer"), str) or not result["answer"]:
            raise RuntimeError("Offline output has no answer")
        trace = result.get("trace")
        if not isinstance(trace, list) or not {"action", "observation", "answer"}.issubset(
            {step.get("type") for step in trace}
        ):
            raise RuntimeError("Offline output lacks an observable tool/answer trajectory")
        if not any(step.get("type") == "answer" and step.get("content") == result["answer"] for step in trace):
            raise RuntimeError("Offline answer and saved trajectory disagree")

    print(f"PASS: {python_count} Python files, tracked-file checks, offline script and JSON output")
    print("Live Kimi/search APIs were not called.")


if __name__ == "__main__":
    main()

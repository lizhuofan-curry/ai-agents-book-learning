"""由用户运行：调用已配置的Kimi Code生成论文PPT，保存输入与原始事件。

默认复用第一章Moonshot API；--connection configured可沿用Kimi账号配置。
Kimi的--prompt模式自动执行工具；本脚本不设置--auto或--yolo。
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_api(environment, settings, model=None):
    """密钥只通过子进程环境传递；返回的记录不包含密钥。"""
    api_key = str(settings.get("MOONSHOT_API_KEY") or environment.get("MOONSHOT_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("项目根目录.env缺少MOONSHOT_API_KEY，请在本地补齐，不要发到聊天")
    base_url = str(settings.get("KIMI_BASE_URL") or "https://api.moonshot.cn/v1").strip()
    url = urlsplit(base_url)
    if (url.scheme != "https" or url.hostname not in {"api.moonshot.cn", "api.moonshot.ai"}
            or url.username or url.password or url.query or url.fragment
            or url.port not in {None, 443} or url.path.rstrip("/") != "/v1"):
        raise RuntimeError("仅复用Moonshot官方API地址，请检查.env的KIMI_BASE_URL")
    model_id = model or str(settings.get("KIMI_PPT_MODEL") or "kimi-k3")
    # 独立数据目录不含原OAuth账号，临时模型配置不会写回全局配置。
    environment["KIMI_CODE_HOME"] = str(ROOT / "工具" / "kimi-moonshot-api")
    for name in list(environment):
        if name.startswith("KIMI_MODEL_"):
            del environment[name]
    environment.update({
        "KIMI_MODEL_NAME": model_id,
        "KIMI_MODEL_PROVIDER_TYPE": "kimi",
        "KIMI_MODEL_BASE_URL": base_url,
        "KIMI_MODEL_API_KEY": api_key,
        "KIMI_MODEL_MAX_CONTEXT_SIZE": "262144",
        "KIMI_DISABLE_TELEMETRY": "1",
        "KIMI_CODE_NO_AUTO_UPDATE": "1",
        "KIMI_CODE_BUILTIN_PRODUCT_SKILLS": "0",
    })
    return {"connection": "moonshot-api", "model_requested": model_id,
            "base_url": base_url, "credential_source": "本地MOONSHOT_API_KEY；不记录值",
            "runtime_data_directory": environment["KIMI_CODE_HOME"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", choices=["moonshot-api", "configured"], default="moonshot-api",
                        help="默认使用第一章API；configured使用Kimi原账号配置")
    parser.add_argument("--model", help="API模式为模型ID；configured模式为已配置模型别名")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError("请使用 E:\\AI_Agents_Book\\.venv\\Scripts\\python.exe 运行")
    config = read_json(ROOT / "工具链配置.json")
    check = read_json(ROOT / "工具链检查.json")
    if not config.get("toolchain_ready") or not check.get("ready"):
        raise RuntimeError("先运行03并得到PPT_TOOLCHAIN_READY")
    prepared = read_json(ROOT / "准备记录.json")
    paper = ROOT / "材料" / "attention-is-all-you-need.pdf"
    skill = ROOT / "skills" / "pptx"
    if digest(paper) != prepared["paper"]["sha256"]:
        raise RuntimeError("论文SHA256与准备记录不一致")
    for name, expected in prepared["official_skill"]["file_sha256"].items():
        candidate = (skill / name).resolve()
        if not candidate.is_relative_to(skill.resolve()) or digest(candidate) != expected:
            raise RuntimeError("官方Skill文件校验失败：" + name)
    kimi = shutil.which("kimi") or str(Path.home() / ".kimi-code" / "bin" / "kimi.exe")
    if not Path(kimi).is_file():
        raise RuntimeError("未找到Kimi Code，请反馈这个错误")

    run_dir = ROOT / "运行结果" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(config["environment"])
    bin_dirs = [str(PROJECT / ".venv" / "Scripts")]
    bin_dirs += [str(Path(p).parent) for p in config["programs"].values() if p]
    environment["PATH"] = os.pathsep.join(bin_dirs + [environment.get("PATH", "")])
    environment["VIRTUAL_ENV"] = str(PROJECT / ".venv")
    environment["NODE_PATH"] = str(ROOT / "node_modules")
    if args.connection == "moonshot-api":
        from dotenv import dotenv_values
        connection = configure_api(environment, dotenv_values(PROJECT / ".env", interpolate=False), args.model)
        Path(environment["KIMI_CODE_HOME"]).mkdir(parents=True, exist_ok=True)
    else:
        connection = {"connection": "configured",
                      "model_requested": args.model or "Kimi当前默认模型；实际模型以事件为准"}
    replacements = {
        "{{RUN_DIR}}": run_dir.as_posix(), "{{PAPER}}": paper.as_posix(),
        "{{SKILL}}": skill.as_posix(), "{{PYTHON}}": Path(sys.executable).as_posix(),
        "{{NODE}}": Path(config["programs"]["node"]).as_posix(),
        "{{ADAPTER}}": (Path(__file__).parent / "html2pptx_windows.cjs").as_posix(),
    }
    prompt = Path(__file__).with_name("论文PPT任务提示词.md").read_text(encoding="utf-8")
    for token, value in replacements.items():
        prompt = prompt.replace(token, value)
    (run_dir / "任务提示词.md").write_text(prompt, encoding="utf-8")
    command = [kimi, "--skills-dir", str(ROOT / "skills")]
    if args.model and args.connection == "configured":
        command += ["--model", args.model]
    command += ["--output-format", "stream-json", "--prompt", prompt]
    version = subprocess.run([kimi, "--version"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=20)
    receipt = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": "Kimi Code CLI", "runtime_version": version.stdout.strip(),
        **connection,
        "command": command[:-1] + ["<见任务提示词.md>"],
        "working_directory": str(run_dir), "paper_sha256": digest(paper),
        "official_skill_revision": prepared["official_skill"]["revision"],
        "toolchain_checked_at": check["checked_at"],
        "permission_mode": "Kimi --prompt的默认auto策略；不是文件系统沙箱",
        "adapter": str(Path(__file__).with_name("html2pptx_windows.cjs")),
        "exit_code": None, "artifact_acceptance": "待检查产物及渐进式加载轨迹",
    }
    receipt_path = run_dir / "启动记录.json"
    save_json(receipt_path, receipt)
    if args.connection == "moonshot-api":
        print("使用第一章Moonshot API：", connection["model_requested"], connection["base_url"], flush=True)
        print("使用API余额按量计费，不使用Kimi Code订阅；密钥不写入启动记录。", flush=True)
    else:
        print("使用Kimi原账号配置，模型服务按账号规则执行，非本地qwen3。", flush=True)
    print("本次目录：", run_dir, flush=True)
    print("正在生成；调用事件实时保存，结束前请保持窗口打开。", flush=True)
    process = None
    pacer = None
    try:
        if args.connection == "moonshot-api":
            from moonshot_pacing import PacedMoonshot
            pacer = PacedMoonshot(connection["base_url"], environment["KIMI_MODEL_API_KEY"],
                                  run_dir / "api_pacing.jsonl")
            pacer.start()
            environment["KIMI_MODEL_BASE_URL"] = pacer.local_url
            environment["KIMI_MODEL_API_KEY"] = pacer.token
            environment.pop("MOONSHOT_API_KEY", None)
            receipt["request_pacing"] = {"minimum_interval_seconds": 25,
                                         "429_retry_wait_seconds": [65, 130, 260],
                                         "serial_forwarding": True,
                                         "note": "本机转发至原Moonshot官方端点，非第三方代理"}
            save_json(receipt_path, receipt)
            print("已启用API节流：请求至少间隔25秒；429时等待65秒起再试。", flush=True)
        with (run_dir / "kimi_stream.jsonl").open("w", encoding="utf-8") as events, \
                (run_dir / "kimi_stderr.log").open("w", encoding="utf-8") as errors:
            process = subprocess.Popen(command, cwd=run_dir, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=errors, text=True, encoding="utf-8", errors="replace")
            for index, line in enumerate(process.stdout, 1):
                events.write(line)
                events.flush()
                print(f"[事件{index}] {line.rstrip()[:700]}", flush=True)
            receipt["exit_code"] = process.wait()
    except KeyboardInterrupt:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        receipt["interrupted"] = True
        receipt["exit_code"] = process.returncode if process else None
        print("本次已中断，已有事件和文件保留。", flush=True)
    except OSError as exc:
        receipt["launch_error"] = f"{type(exc).__name__}: {exc}"
        receipt["exit_code"] = 1
    finally:
        if pacer is not None:
            pacer.close()
        receipt["finished_at"] = datetime.now().isoformat(timespec="seconds")
        receipt["pptx_files"] = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*.pptx")]
        changed = [name for name, expected in prepared["official_skill"]["file_sha256"].items()
                   if not (skill / name).is_file() or digest(skill / name) != expected]
        receipt["official_skill_changed_files"] = changed
        save_json(receipt_path, receipt)
    print("本次记录：", receipt_path, flush=True)
    print("PPTX文件：", receipt["pptx_files"], flush=True)
    if receipt["exit_code"] != 0 or receipt.get("interrupted") or changed:
        error_path = run_dir / "kimi_stderr.log"
        if error_path.exists():
            print(error_path.read_text(encoding="utf-8")[-3000:], flush=True)
        print("AGENT_RUN_INCOMPLETE：保留日志；请反馈末尾输出，不必重装依赖。")
        raise SystemExit(1)
    print("AGENT_RUN_FINISHED：进程结束；PPT内容、图表、预览及Skill轨迹等待验收。")


if __name__ == "__main__":
    main()

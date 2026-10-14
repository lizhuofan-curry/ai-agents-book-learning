"""只看程序怎样构造状态栏，不调用模型。可用于看懂01的输入差异。"""
import json
from pathlib import Path
from status_tools import CASES, TITLES, config_for, initial_case, initial_messages, status_message


def main():
    rows = []
    for case in CASES:
        print("\n任务：", TITLES[case])
        for arm in ("disabled", "single_feature", "all_features"):
            world, events = initial_case(case)
            config = config_for(arm, case)
            hint = status_message(world, config)
            print(arm, "末尾状态栏：", hint or "（无）")
            rows.append({"case": case, "arm": arm, "config": config,
                         "messages": initial_messages(case, config, events), "hint": hint,
                         "initial_state": world.snapshot()})
    output = Path(__file__).resolve().parents[1] / "状态栏输入预览.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("预览保存：", output)
    print("STATUS_HINT_PREVIEW_COMPLETE：只构造输入，没有模型回答或任务成绩。")


if __name__ == "__main__":
    main()

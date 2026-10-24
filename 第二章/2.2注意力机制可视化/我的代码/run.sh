#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$script_dir/../../.." && pwd)"
python_executable="$project_dir/.venv/bin/python"
if [[ ! -x "$python_executable" ]]; then
  echo '请先在项目根目录执行 bash scripts/setup_macos.sh。' >&2
  exit 1
fi
export HF_HOME="$project_dir/.uv-cache/huggingface"
export MPLCONFIGDIR="$project_dir/.uv-cache/matplotlib"
exec "$python_executable" "$script_dir/attention_demo.py"

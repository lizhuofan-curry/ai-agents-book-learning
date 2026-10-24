#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
if [[ "$(uname -s)" != Darwin ]]; then
  echo '这个入口用于 macOS；其他系统请使用 uv sync。' >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo '未找到 uv，请先安装 uv 并重新打开终端。' >&2
  exit 1
fi
if [[ -f .venv/Scripts/python.exe && ! -x .venv/bin/python ]]; then
  mkdir -p .macos-migration/backups
  backup_dir=".macos-migration/backups/venv-windows-$(date +%Y%m%d-%H%M%S)"
  mv .venv "$backup_dir"
  echo "Windows 环境已保留到 $backup_dir"
fi
export UV_CACHE_DIR="$project_dir/.uv-cache/macos"
export PYTHONDONTWRITEBYTECODE=1
# Annoy 1.17.3 默认 -ffast-math 在 Apple clang 上可能只返回一个邻居。
# 仅改变本机编译选项，保留锁定的上游源码版本。
export ANNOY_COMPILER_ARGS='-O3,-fno-fast-math,-fno-associative-math,-DANNOYLIB_MULTITHREADED_BUILD,-std=c++14,-mmacosx-version-min=11.0'
uv cache clean annoy
uv sync --locked --python 3.12 --group chapter3-index --reinstall-package annoy
.venv/bin/python scripts/name_environment.py
.venv/bin/python scripts/check_native_indexes.py
.venv/bin/python .github/scripts/check_repo.py
echo 'MACOS_ENVIRONMENT_READY：日常运行请使用 .venv/bin/python。'

#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-project_src_$(date +%Y%m%d_%H%M%S).tar.gz}"
TMP_FILE="$(mktemp)"

trap 'rm -f "$TMP_FILE"' EXIT

find . -maxdepth 1 -type f \( -name "*.py" -o -name "README.md" \) -print0 > "$TMP_FILE"

if [ ! -s "$TMP_FILE" ]; then
    echo "当前目录下没有找到 .py 文件或 README.md"
    exit 1
fi

tar --null -czf "$OUT" --files-from "$TMP_FILE"

echo "已打包到: $OUT"
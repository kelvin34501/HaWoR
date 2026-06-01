#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
output="${2:-selected_dirs.zip}"

root="$(realpath "$root")"
stage_dir="$(mktemp -d)"
trap 'rm -rf "$stage_dir"' EXIT

cd "$root"

# 处理 cam_space 目录：整个目录原样保留
find . -type d -name 'cam_space*' | while IFS= read -r dir; do
    echo "收集 cam_space 目录: $dir"
    mkdir -p "$stage_dir/$(dirname "$dir")"
    cp -r "$dir" "$stage_dir/$dir"
done

# 处理 SLAM 目录：复制目录结构，但排除名字中含 disps 的 npz 文件
find . -type d -name 'SLAM' | while IFS= read -r dir; do
    echo "收集 SLAM 目录(排除 *disps*.npz): $dir"
    mkdir -p "$stage_dir/$dir"

    rsync -a \
        --exclude='*disps*' \
        "$dir"/ "$stage_dir/$dir"/
done

if [[ -z "$(find "$stage_dir" -mindepth 1 -print -quit)" ]]; then
    echo "没有找到符合条件的内容"
    exit 0
fi

cd "$stage_dir"
zip -r "$root/$output" .

echo
echo "打包完成: $root/$output"
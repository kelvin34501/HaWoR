#!/usr/bin/env bash

root1="${1:-.}"


find "$root1" -type f -name '*.[mM][pP]4' -exec rm -f {} +


find "$root1" -type f -name '*.[lL][rR][fF]' -exec rm -f {} +

find "$root1" -type d -name 'extracted_image*' -prune -exec rm -rf {} +

echo "删除完成"
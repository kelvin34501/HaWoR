#!/bin/bash

#concurrent hawor with json

# ROOT_DIR="/mnt/archive/users/kefan/reconstruction"
# for CLIP1_DIR in "$ROOT_DIR"/*/; do

CLIP1_DIR="/mnt/archive/users/kefan/reconstruction/260416_s01"
LOG_FILE="$CLIP1_DIR/batch_process.log"
MAX_JOBS=1 # ⚡ 并行 GPU 数量

echo "Batch processing started at $(date)" | tee "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

# Count total folders
TOTAL_VIDEOS=$(ls -f "$CLIP1_DIR"/* 2>/dev/null | wc -l)

if [ "$TOTAL_VIDEOS" -eq 0 ]; then
    echo "No videos found in $CLIP1_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Found $TOTAL_VIDEOS videos to process" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CURRENT=0
SUCCESSFUL=0
FAILED=0

# # Ensure output dir exists
# mkdir -p "$OUTPUT_DIR"

# function to count running background jobs
jobs_running() {
    jobs -rp | wc -l
}

# Process each mp4 file

for task in $(find "$CLIP1_DIR" -type f -iname "*.mp4"); do
    echo "Processing video: $task"
    CURRENT=$((CURRENT + 1))
    VIDEO_NAME=$(basename "$task")
    VIDEO_BASE=$(basename "$task" .MP4)
    folder_name=${task%.[mM][Pp]4}
    # 分配 GPU，循环 0~7
    GPU_ID=$(( (CURRENT - 1) % MAX_JOBS ))
    echo "[$CURRENT] Processing: $VIDEO_NAME on GPU $GPU_ID" | tee -a "$LOG_FILE"

    # 启动后台任务
    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        python demo.py --video_path "$task" --vis_mode off 2>&1 | tee -a "$LOG_FILE"
        # Frames are decoded on demand; interpolation derives the 50fps count from the video.
        python scripts/interpolation.py --folder_path "$folder_name" --video_path "$task" 2>&1 | tee -a "$LOG_FILE"
    ) &

    # 控制同时后台任务不超过 MAX_JOBS
    while [ $(jobs_running) -ge $MAX_JOBS ]; do
        sleep 1
    done
done

# 等待所有后台任务完成
wait

# Summary
echo "======================================" | tee -a "$LOG_FILE"
echo "Batch processing completed at $(date)" | tee -a "$LOG_FILE"
# done
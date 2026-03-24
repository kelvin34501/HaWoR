#!/bin/bash

# Batch visualize all task directories in a clip folder
# Usage: bash scripts/batch_visualize_clip.sh

# ====== Configuration ======
CLIP_DIR="example/clip_3"
CLIP_NAME="clip_3"
CAPTIONS_JSON="$CLIP_DIR/clip_3.json"
SEGMENT_DEF_JSON="$CLIP_DIR/clip_3_segment_def.json"
CUDA_DEVICE=4
DEVICE="cuda"
LOG_FILE="$CLIP_DIR/batch_visualize.log"

# ====== Start Processing ======
echo "Batch visualization started at $(date)" | tee "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "Clip directory: $CLIP_DIR" | tee -a "$LOG_FILE"
echo "CUDA device: $CUDA_DEVICE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Find all task directories matching the pattern XX_task
TASK_DIRS=($(find "$CLIP_DIR" -maxdepth 1 -type d -name '*_task' | sort))
TOTAL_TASKS=${#TASK_DIRS[@]}

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "No task directories found in $CLIP_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Found $TOTAL_TASKS task directories to process" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CURRENT=0
SUCCESSFUL=0
FAILED=0

# Process each task directory
for task_dir in "${TASK_DIRS[@]}"; do
    CURRENT=$((CURRENT + 1))
    TASK_NAME=$(basename "$task_dir")
    
    # Extract the task prefix (e.g., "00" from "00_task")
    TASK_PREFIX="${TASK_NAME%_task}"
    
    # Define output paths
    OUTPUT_VIDEO="$CLIP_DIR/${TASK_PREFIX}_viz_with_captions.mp4"
    OUTPUT_CLIPS_DIR="$CLIP_DIR/${TASK_PREFIX}_task_clips"
    
    echo "[$CURRENT/$TOTAL_TASKS] Processing: $TASK_NAME" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "Input directory: $task_dir" | tee -a "$LOG_FILE"
    echo "Output video: $OUTPUT_VIDEO" | tee -a "$LOG_FILE"
    echo "Output clips dir: $OUTPUT_CLIPS_DIR" | tee -a "$LOG_FILE"
    
    # Run visualization script
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python scripts/visualize_hands_to_video.py \
        --device $DEVICE \
        --dir "$task_dir" \
        --output "$OUTPUT_VIDEO" \
        --captions-json "$CAPTIONS_JSON" \
        --segment-def-json "$SEGMENT_DEF_JSON" \
        --output-dir-per-clip "$OUTPUT_CLIPS_DIR" 2>&1 | tee -a "$LOG_FILE"
    
    # Check if output video was created successfully
    if [ -f "$OUTPUT_VIDEO" ]; then
        echo "✓ Successfully visualized: $TASK_NAME" | tee -a "$LOG_FILE"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "✗ Failed to visualize: $TASK_NAME (output not found)" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
    
    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# Summary
echo "======================================" | tee -a "$LOG_FILE"
echo "Batch visualization completed at $(date)" | tee -a "$LOG_FILE"
echo "Total tasks: $TOTAL_TASKS" | tee -a "$LOG_FILE"
echo "Successful: $SUCCESSFUL" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

#!/bin/bash

# Batch process all mp4 videos in example/clip_1 with HaWoR
# Results will be saved in example/clip_1/{video_name}/ directories

CLIP1_DIR="example/clip_1"
LOG_FILE="$CLIP1_DIR/batch_process.log"

# Create log file
echo "Batch processing started at $(date)" | tee "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

# Count total videos
TOTAL_VIDEOS=$(ls -1 "$CLIP1_DIR"/*.mp4 2>/dev/null | wc -l)

if [ "$TOTAL_VIDEOS" -eq 0 ]; then
    echo "No mp4 files found in $CLIP1_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Found $TOTAL_VIDEOS videos to process" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CURRENT=0
SUCCESSFUL=0
FAILED=0

# Process each mp4 file
for video in "$CLIP1_DIR"/*.mp4; do
    CURRENT=$((CURRENT + 1))
    VIDEO_NAME=$(basename "$video")
    
    echo "[$CURRENT/$TOTAL_VIDEOS] Processing: $VIDEO_NAME" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    
    # Run HaWoR processing without visualization (vis_mode=off)
    python demo.py --video_path "$video" --vis_mode off 2>&1 | tee -a "$LOG_FILE"
    
    # Check if output file was created successfully
    VIDEO_BASE=$(basename "$video" .mp4)
    OUTPUT_FILE="$CLIP1_DIR/$VIDEO_BASE/world_space_res.pth"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✓ Successfully processed: $VIDEO_NAME" | tee -a "$LOG_FILE"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "✗ Failed to process: $VIDEO_NAME (output not found)" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
    
    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# Summary
echo "======================================" | tee -a "$LOG_FILE"
echo "Batch processing completed at $(date)" | tee -a "$LOG_FILE"
echo "Total videos: $TOTAL_VIDEOS" | tee -a "$LOG_FILE"
echo "Successful: $SUCCESSFUL" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

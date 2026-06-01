#!/usr/bin/env bash

# concurrent hawor with json (robust mode)
set -Eeuo pipefail
shopt -s nullglob

ROOT_DIR="${ROOT_DIR:-/mnt/archive/users/kefan/reconstruction/260427_s02}"
MAX_JOBS="${MAX_JOBS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "ERROR: ROOT_DIR does not exist: $ROOT_DIR" >&2
    exit 1
fi

if [[ ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_JOBS must be a positive integer, got: $MAX_JOBS" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] && (( MAX_JOBS > GPU_COUNT )); then
        echo "INFO: MAX_JOBS=$MAX_JOBS exceeds GPU_COUNT=$GPU_COUNT, reducing MAX_JOBS to $GPU_COUNT"
        MAX_JOBS="$GPU_COUNT"
    fi
fi

cd "$PROJECT_DIR"

timestamp() {
    date "+%F %T"
}

cleanup_on_signal() {
    local sig="$1"
    echo "[$(timestamp)] Received $sig, stopping child jobs..."
    jobs -pr | xargs -r kill
    wait || true
    exit 130
}

trap 'cleanup_on_signal INT' INT
trap 'cleanup_on_signal TERM' TERM

run_task() {
    local task="$1"
    local gpu_id="$2"
    local task_log="$3"
    local folder_name

    folder_name="${task%.*}"

    {
        echo "[$(timestamp)] START task=$task gpu=$gpu_id"
        export CUDA_VISIBLE_DEVICES="$gpu_id"

        "$PYTHON_BIN" demo.py --video_path "$task" --vis_mode off
        "$PYTHON_BIN" scripts/extract_image_50fps.py --video_path "$task" --output_folder "$folder_name/extracted_images_50fps"
        "$PYTHON_BIN" scripts/interpolation.py --folder_path "$folder_name"

        echo "[$(timestamp)] DONE task=$task gpu=$gpu_id"
    } >>"$task_log" 2>&1
}

for CLIP1_DIR in "$ROOT_DIR"/*/; do
    LOG_FILE="$CLIP1_DIR/batch_process.log"
    TASK_LOG_DIR="$CLIP1_DIR/task_logs"
    mkdir -p "$TASK_LOG_DIR"

    echo "[$(timestamp)] Batch processing started" | tee "$LOG_FILE"
    echo "[$(timestamp)] clip_dir=$CLIP1_DIR" | tee -a "$LOG_FILE"
    echo "[$(timestamp)] project_dir=$PROJECT_DIR" | tee -a "$LOG_FILE"
    echo "[$(timestamp)] max_jobs=$MAX_JOBS" | tee -a "$LOG_FILE"
    echo "======================================" | tee -a "$LOG_FILE"

    mapfile -d '' VIDEOS < <(find "$CLIP1_DIR" -type f -iname "*.mp4" -print0)
    TOTAL_VIDEOS="${#VIDEOS[@]}"

    if (( TOTAL_VIDEOS == 0 )); then
        echo "[$(timestamp)] No videos found, skip." | tee -a "$LOG_FILE"
        echo "======================================" | tee -a "$LOG_FILE"
        continue
    fi

    echo "[$(timestamp)] Found $TOTAL_VIDEOS videos to process" | tee -a "$LOG_FILE"

    CURRENT=0
    SUCCESSFUL=0
    FAILED=0

    declare -a PIDS=()
    declare -A PID_VIDEO=()
    declare -A PID_GPU=()
    declare -A PID_TASK_LOG=()

    reap_finished_jobs() {
        local alive=()
        local pid

        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
                continue
            fi

            if wait "$pid"; then
                SUCCESSFUL=$((SUCCESSFUL + 1))
                echo "[$(timestamp)] OK: ${PID_VIDEO[$pid]} (gpu=${PID_GPU[$pid]})" | tee -a "$LOG_FILE"
            else
                FAILED=$((FAILED + 1))
                echo "[$(timestamp)] FAIL: ${PID_VIDEO[$pid]} (gpu=${PID_GPU[$pid]})" | tee -a "$LOG_FILE"
                echo "[$(timestamp)] See task log: ${PID_TASK_LOG[$pid]}" | tee -a "$LOG_FILE"
            fi

            unset 'PID_VIDEO[$pid]' 'PID_GPU[$pid]' 'PID_TASK_LOG[$pid]'
        done

        PIDS=("${alive[@]}")
    }

    for task in "${VIDEOS[@]}"; do
        CURRENT=$((CURRENT + 1))
        VIDEO_NAME="$(basename "$task")"
        VIDEO_BASE="${VIDEO_NAME%.*}"
        GPU_ID=$(( (CURRENT - 1) % MAX_JOBS ))
        TASK_LOG="$TASK_LOG_DIR/${VIDEO_BASE}.log"

        echo "[$(timestamp)] [$CURRENT/$TOTAL_VIDEOS] Launch: $VIDEO_NAME on GPU $GPU_ID" | tee -a "$LOG_FILE"

        run_task "$task" "$GPU_ID" "$TASK_LOG" &
        pid="$!"
        PIDS+=("$pid")
        PID_VIDEO["$pid"]="$task"
        PID_GPU["$pid"]="$GPU_ID"
        PID_TASK_LOG["$pid"]="$TASK_LOG"

        while (( ${#PIDS[@]} >= MAX_JOBS )); do
            reap_finished_jobs
            sleep 1
        done
    done

    while (( ${#PIDS[@]} > 0 )); do
        reap_finished_jobs
        sleep 1
    done

    echo "======================================" | tee -a "$LOG_FILE"
    echo "[$(timestamp)] Batch processing completed" | tee -a "$LOG_FILE"
    echo "[$(timestamp)] Summary: total=$TOTAL_VIDEOS success=$SUCCESSFUL failed=$FAILED" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done
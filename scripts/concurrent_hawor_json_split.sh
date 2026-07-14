#!/usr/bin/env bash

set -Eeuo pipefail
shopt -s nullglob

# ===== User strategy (confirmed) =====
SEGMENT_SECONDS="${SEGMENT_SECONDS:-100}"
MIN_LAST_SEGMENT_SECONDS="${MIN_LAST_SEGMENT_SECONDS:-50}"
MAX_CHUNK_FRAMES="${MAX_CHUNK_FRAMES:-3001}"
SPLIT_MODE="${SPLIT_MODE:-reencode}"          # copy | reencode
OVERLAP_POLICY="${OVERLAP_POLICY:-keep_last}" # keep_last | keep_first

# ===== Batch policy (to discuss in bash) =====
ROOT_DIR="${ROOT_DIR:-/mnt/archive/users/kefan/reconstruction/0422-0423}"
MAX_JOBS="${MAX_JOBS:-6}"
# 指定使用哪些 GPU，支持逗号/空格分隔，例如 "2,3" 或 "2 3"。
# 为空时保持旧行为，默认使用 0..MAX_JOBS-1。
# GPU_IDS="${GPU_IDS:-2,3,4,5}"
GPU_IDS="${GPU_IDS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VIS_MODE="${VIS_MODE:-off}"

# 4) 并发策略开关：
# 0 = 顺序跑（单卡稳定），1 = 按 MAX_JOBS 并发
ENABLE_PARALLEL="${ENABLE_PARALLEL:-1}"

# 5) 后处理策略开关：
# 0 = 只做分段 demo + merge
# 1 = 在 merge 后继续执行 extract_image_50fps + interpolation（原视频级别）
RUN_POST_STEPS="${RUN_POST_STEPS:-1}"

# 5b) 强制插值开关：
# 0 = 仅在 RUN_POST_STEPS=1 时执行 interpolation
# 1 = 无论 RUN_POST_STEPS 都执行 interpolation（不自动 extract）
FORCE_INTERPOLATE="${FORCE_INTERPOLATE:-0}"

# 6) 清理策略开关：
# 0 = 保留 xxx_segmented 中间产物
# 1 = 单个视频成功后删除 xxx_segmented（切片 + 中间目录）
CLEANUP_INTERMEDIATE="${CLEANUP_INTERMEDIATE:-1}"

# 7) 已重建序列跳过开关：
# 0 = 不跳过，全部重跑
# 1 = 若 sequence 已有 cam_space_merged + merged SLAM 则跳过
SKIP_RECONSTRUCTED="${SKIP_RECONSTRUCTED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found in PATH" >&2
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

declare -a GPU_POOL=()

init_gpu_pool() {
    local normalized gpu
    declare -A seen_gpu_ids=()

    if [[ -n "${GPU_IDS//[[:space:],]/}" ]]; then
        normalized="${GPU_IDS//,/ }"
        read -r -a GPU_POOL <<< "$normalized"

        if (( ${#GPU_POOL[@]} == 0 )); then
            echo "ERROR: GPU_IDS is set but no GPU IDs were parsed: $GPU_IDS" >&2
            exit 1
        fi

        for gpu in "${GPU_POOL[@]}"; do
            if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
                echo "ERROR: GPU_IDS must contain non-negative integers separated by commas/spaces, got: $GPU_IDS" >&2
                exit 1
            fi

            if [[ -n "${seen_gpu_ids[$gpu]:-}" ]]; then
                echo "ERROR: GPU_IDS contains duplicate GPU id: $gpu" >&2
                exit 1
            fi
            seen_gpu_ids["$gpu"]=1
        done
    else
        local i
        for (( i = 0; i < MAX_JOBS; i++ )); do
            GPU_POOL+=("$i")
        done
    fi

    if (( ${#GPU_POOL[@]} == 0 )); then
        echo "ERROR: no GPU IDs available for scheduling" >&2
        exit 1
    fi

    if (( MAX_JOBS > ${#GPU_POOL[@]} )); then
        echo "INFO: MAX_JOBS=$MAX_JOBS exceeds available GPU slots=${#GPU_POOL[@]}, reducing MAX_JOBS to ${#GPU_POOL[@]}"
        MAX_JOBS="${#GPU_POOL[@]}"
    fi
}

gpu_pool_to_string() {
    local IFS=,
    printf "%s" "${GPU_POOL[*]}"
}

if [[ ! "$MIN_LAST_SEGMENT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MIN_LAST_SEGMENT_SECONDS must be a non-negative integer, got: $MIN_LAST_SEGMENT_SECONDS" >&2
    exit 1
fi

if [[ ! "$MAX_CHUNK_FRAMES" =~ ^[0-9]+$ ]] || (( MAX_CHUNK_FRAMES < 2 )); then
    echo "ERROR: MAX_CHUNK_FRAMES must be an integer >= 2, got: $MAX_CHUNK_FRAMES" >&2
    exit 1
fi

if [[ "$SPLIT_MODE" != "copy" && "$SPLIT_MODE" != "reencode" ]]; then
    echo "ERROR: SPLIT_MODE must be copy or reencode, got: $SPLIT_MODE" >&2
    exit 1
fi

if [[ "$OVERLAP_POLICY" != "keep_last" && "$OVERLAP_POLICY" != "keep_first" ]]; then
    echo "ERROR: OVERLAP_POLICY must be keep_last or keep_first, got: $OVERLAP_POLICY" >&2
    exit 1
fi

if [[ "$ENABLE_PARALLEL" != "0" && "$ENABLE_PARALLEL" != "1" ]]; then
    echo "ERROR: ENABLE_PARALLEL must be 0 or 1, got: $ENABLE_PARALLEL" >&2
    exit 1
fi

if [[ "$RUN_POST_STEPS" != "0" && "$RUN_POST_STEPS" != "1" ]]; then
    echo "ERROR: RUN_POST_STEPS must be 0 or 1, got: $RUN_POST_STEPS" >&2
    exit 1
fi

if [[ "$FORCE_INTERPOLATE" != "0" && "$FORCE_INTERPOLATE" != "1" ]]; then
    echo "ERROR: FORCE_INTERPOLATE must be 0 or 1, got: $FORCE_INTERPOLATE" >&2
    exit 1
fi

if [[ "$CLEANUP_INTERMEDIATE" != "0" && "$CLEANUP_INTERMEDIATE" != "1" ]]; then
    echo "ERROR: CLEANUP_INTERMEDIATE must be 0 or 1, got: $CLEANUP_INTERMEDIATE" >&2
    exit 1
fi

if [[ "$SKIP_RECONSTRUCTED" != "0" && "$SKIP_RECONSTRUCTED" != "1" ]]; then
    echo "ERROR: SKIP_RECONSTRUCTED must be 0 or 1, got: $SKIP_RECONSTRUCTED" >&2
    exit 1
fi

timestamp() {
    date "+%F %T"
}

run_one_video() {
    local video_path="$1"
    local gpu_id="$2"
    local task_log="$3"
    local split_flag=""
    local video_dir video_name video_stem segmented_dir

    video_dir="$(dirname "$video_path")"
    video_name="$(basename "$video_path")"
    video_stem="${video_name%.*}"
    segmented_dir="$video_dir/${video_stem}_segmented"

    if [[ "$SPLIT_MODE" == "reencode" ]]; then
        split_flag="--reencode"
    fi

    {
        echo "[$(timestamp)] START video=$video_path gpu=$gpu_id"
        echo "[$(timestamp)] strategy: segment=${SEGMENT_SECONDS}s min_last_segment=${MIN_LAST_SEGMENT_SECONDS}s max_chunk_frames=$MAX_CHUNK_FRAMES split_mode=$SPLIT_MODE overlap=$OVERLAP_POLICY"

        export CUDA_VISIBLE_DEVICES="$gpu_id"
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/segmented_demo_pipeline.py" \
            --video_path "$video_path" \
            --segment_seconds "$SEGMENT_SECONDS" \
            --min_last_segment_seconds "$MIN_LAST_SEGMENT_SECONDS" \
            --max_chunk_frames "$MAX_CHUNK_FRAMES" \
            --overlap_policy "$OVERLAP_POLICY" \
            --vis_mode "$VIS_MODE" \
            --gpu_id "$gpu_id" \
            $split_flag

        local folder_name
        folder_name="${video_path%.[mM][Pp]4}"

        # Frames are decoded on demand; interpolation derives the 50fps count from
        # the video, so extract_image_50fps.py is no longer needed.
        if [[ "$RUN_POST_STEPS" == "1" || "$FORCE_INTERPOLATE" == "1" ]]; then
            "$PYTHON_BIN" "$PROJECT_DIR/scripts/interpolation.py" --folder_path "$folder_name" --video_path "$video_path"
        fi

        if [[ "$CLEANUP_INTERMEDIATE" == "1" ]]; then
            if [[ -d "$segmented_dir" ]]; then
                rm -rf "$segmented_dir"
                echo "[$(timestamp)] CLEANUP removed: $segmented_dir"
            else
                echo "[$(timestamp)] CLEANUP skipped (not found): $segmented_dir"
            fi
        fi

        echo "[$(timestamp)] DONE video=$video_path gpu=$gpu_id"
    } >>"$task_log" 2>&1
}

run_interpolation_only() {
    local video_path="$1"
    local task_log="$2"
    local folder_name

    folder_name="${video_path%.[mM][Pp]4}"

    {
        echo "[$(timestamp)] START interpolation_only video=$video_path"
        "$PYTHON_BIN" "$PROJECT_DIR/scripts/interpolation.py" --folder_path "$folder_name"
        echo "[$(timestamp)] DONE interpolation_only video=$video_path"
    } >>"$task_log" 2>&1
}

collect_videos() {
    find "$ROOT_DIR" -type f -iname "*.mp4" -print0
}

is_sequence_reconstructed() {
    local video_path="$1"
    local video_dir video_name video_stem seq_dir

    video_dir="$(dirname "$video_path")"
    video_name="$(basename "$video_path")"
    video_stem="${video_name%.*}"
    seq_dir="$video_dir/$video_stem"

    printf "Checking reconstruction for sequence: %s\n" "$seq_dir"

    if [[ ! -d "$seq_dir" ]]; then
        return 1
    fi

    if ! compgen -G "$seq_dir/cam_space/1/*.json" >/dev/null; then
    printf "No cam_space JSON files found in %s/cam_space/\n" "$seq_dir"
        return 1
    fi

    if ! compgen -G "$seq_dir/SLAM/hawor_slam_w_scale_0_*.npz" >/dev/null; then
    printf "No merged SLAM files found in %s/SLAM/\n" "$seq_dir"
        return 1
    fi

    return 0
}

cd "$PROJECT_DIR"
init_gpu_pool

LOG_FILE="$ROOT_DIR/batch_process_split.log"
TASK_LOG_DIR="$ROOT_DIR/task_logs_split"
mkdir -p "$TASK_LOG_DIR"

echo "[$(timestamp)] Batch processing started" | tee "$LOG_FILE"
echo "[$(timestamp)] root_dir=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "[$(timestamp)] project_dir=$PROJECT_DIR" | tee -a "$LOG_FILE"
echo "[$(timestamp)] max_jobs=$MAX_JOBS parallel=$ENABLE_PARALLEL" | tee -a "$LOG_FILE"
echo "[$(timestamp)] gpu_pool=$(gpu_pool_to_string)" | tee -a "$LOG_FILE"
echo "[$(timestamp)] strategy: segment=${SEGMENT_SECONDS}s min_last_segment=${MIN_LAST_SEGMENT_SECONDS}s max_chunk_frames=$MAX_CHUNK_FRAMES split_mode=$SPLIT_MODE overlap=$OVERLAP_POLICY" | tee -a "$LOG_FILE"
echo "[$(timestamp)] run_post_steps=$RUN_POST_STEPS force_interpolate=$FORCE_INTERPOLATE" | tee -a "$LOG_FILE"
echo "[$(timestamp)] cleanup_intermediate=$CLEANUP_INTERMEDIATE" | tee -a "$LOG_FILE"
echo "[$(timestamp)] skip_reconstructed=$SKIP_RECONSTRUCTED" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

mapfile -d '' VIDEOS < <(collect_videos)
TOTAL_VIDEOS="${#VIDEOS[@]}"

if (( TOTAL_VIDEOS == 0 )); then
    echo "[$(timestamp)] No videos found in $ROOT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[$(timestamp)] Found $TOTAL_VIDEOS videos to process" | tee -a "$LOG_FILE"

CURRENT=0
SUCCESSFUL=0
FAILED=0
SKIPPED=0
INTERP_SUCCESS=0
INTERP_FAILED=0

declare -a PIDS=()
declare -A PID_VIDEO=()
declare -A PID_GPU=()
declare -A PID_TASK_LOG=()
declare -A PID_KIND=()

pick_free_gpu_id() {
    local gpu pid

    for gpu in "${GPU_POOL[@]}"; do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                continue
            fi

            if [[ "${PID_KIND[$pid]:-main}" == "main" && "${PID_GPU[$pid]:-}" == "$gpu" ]]; then
                continue 2
            fi
        done

        printf "%s\n" "$gpu"
        return 0
    done

    echo "ERROR: no free GPU available in pool: $(gpu_pool_to_string)" >&2
    return 1
}

reap_finished_jobs() {
    local alive=()
    local pid

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
            continue
        fi

        if [[ "${PID_KIND[$pid]:-main}" == "interp" ]]; then
            if wait "$pid"; then
                INTERP_SUCCESS=$((INTERP_SUCCESS + 1))
                echo "[$(timestamp)] INTERP_OK: ${PID_VIDEO[$pid]}" | tee -a "$LOG_FILE"
            else
                INTERP_FAILED=$((INTERP_FAILED + 1))
                echo "[$(timestamp)] INTERP_FAIL: ${PID_VIDEO[$pid]}" | tee -a "$LOG_FILE"
                echo "[$(timestamp)] See task log: ${PID_TASK_LOG[$pid]}" | tee -a "$LOG_FILE"
            fi
        else
            if wait "$pid"; then
                SUCCESSFUL=$((SUCCESSFUL + 1))
                echo "[$(timestamp)] OK: ${PID_VIDEO[$pid]} (gpu=${PID_GPU[$pid]})" | tee -a "$LOG_FILE"
            else
                FAILED=$((FAILED + 1))
                echo "[$(timestamp)] FAIL: ${PID_VIDEO[$pid]} (gpu=${PID_GPU[$pid]})" | tee -a "$LOG_FILE"
                echo "[$(timestamp)] See task log: ${PID_TASK_LOG[$pid]}" | tee -a "$LOG_FILE"
            fi
        fi

        unset 'PID_VIDEO[$pid]' 'PID_GPU[$pid]' 'PID_TASK_LOG[$pid]' 'PID_KIND[$pid]'
    done

    PIDS=("${alive[@]}")
}

for task in "${VIDEOS[@]}"; do
    CURRENT=$((CURRENT + 1))
    VIDEO_NAME="$(basename "$task")"
    VIDEO_BASE="${VIDEO_NAME%.*}"
    TASK_LOG="$TASK_LOG_DIR/${VIDEO_BASE}.log"

    if [[ "$SKIP_RECONSTRUCTED" == "1" ]] && is_sequence_reconstructed "$task"; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$(timestamp)] [$CURRENT/$TOTAL_VIDEOS] Skip reconstructed: $VIDEO_NAME" | tee -a "$LOG_FILE"

        if [[ "$FORCE_INTERPOLATE" == "1" ]]; then
            echo "[$(timestamp)] [$CURRENT/$TOTAL_VIDEOS] Force interpolation: $VIDEO_NAME" | tee -a "$LOG_FILE"

            if [[ "$ENABLE_PARALLEL" == "1" ]]; then
                run_interpolation_only "$task" "$TASK_LOG" &
                pid="$!"
                PIDS+=("$pid")
                PID_VIDEO["$pid"]="$task"
                PID_GPU["$pid"]="cpu"
                PID_TASK_LOG["$pid"]="$TASK_LOG"
                PID_KIND["$pid"]="interp"

                while (( ${#PIDS[@]} >= MAX_JOBS )); do
                    reap_finished_jobs
                    sleep 1
                done
            else
                if run_interpolation_only "$task" "$TASK_LOG"; then
                    INTERP_SUCCESS=$((INTERP_SUCCESS + 1))
                    echo "[$(timestamp)] INTERP_OK: $task" | tee -a "$LOG_FILE"
                else
                    INTERP_FAILED=$((INTERP_FAILED + 1))
                    echo "[$(timestamp)] INTERP_FAIL: $task" | tee -a "$LOG_FILE"
                    echo "[$(timestamp)] See task log: $TASK_LOG" | tee -a "$LOG_FILE"
                fi
            fi
        fi

        continue
    fi

    GPU_ID="$(pick_free_gpu_id)"

    echo "[$(timestamp)] [$CURRENT/$TOTAL_VIDEOS] Launch: $VIDEO_NAME on GPU $GPU_ID" | tee -a "$LOG_FILE"

    if [[ "$ENABLE_PARALLEL" == "1" ]]; then
        run_one_video "$task" "$GPU_ID" "$TASK_LOG" &
        pid="$!"
        PIDS+=("$pid")
        PID_VIDEO["$pid"]="$task"
        PID_GPU["$pid"]="$GPU_ID"
        PID_TASK_LOG["$pid"]="$TASK_LOG"
        PID_KIND["$pid"]="main"

        while (( ${#PIDS[@]} >= MAX_JOBS )); do
            reap_finished_jobs
            sleep 1
        done
    else
        if run_one_video "$task" "$GPU_ID" "$TASK_LOG"; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
            echo "[$(timestamp)] OK: $task (gpu=$GPU_ID)" | tee -a "$LOG_FILE"
        else
            FAILED=$((FAILED + 1))
            echo "[$(timestamp)] FAIL: $task (gpu=$GPU_ID)" | tee -a "$LOG_FILE"
            echo "[$(timestamp)] See task log: $TASK_LOG" | tee -a "$LOG_FILE"
        fi
    fi
done

while (( ${#PIDS[@]} > 0 )); do
    reap_finished_jobs
    sleep 1
done

echo "======================================" | tee -a "$LOG_FILE"
echo "[$(timestamp)] Batch processing completed" | tee -a "$LOG_FILE"
echo "[$(timestamp)] Summary: total=$TOTAL_VIDEOS success=$SUCCESSFUL failed=$FAILED skipped=$SKIPPED interp_success=$INTERP_SUCCESS interp_failed=$INTERP_FAILED" | tee -a "$LOG_FILE"

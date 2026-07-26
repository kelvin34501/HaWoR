#!/usr/bin/env bash

export HAWOR_CACHE_DIR=./example/hawor_process
export HAWOR_GPU_IDS=$CUDA_VISIBLE_DEVICES
export HAWOR_JOBS_PER_GPU=4   # max concurrent jobs per GPU (default 1)
export HAWOR_CLEANUP_INTERMEDIATE=true
export HAWOR_CLEANUP_FAILED_CACHE=false
export HAWOR_COPY_EXTRACTED_IMAGES=false
export HAWOR_RUN_VISUALIZATIONS=false  # set true to render cam/world-space visualization videos
export HAWOR_NUM_THREADS=1
export HAWOR_DECORD_NUM_THREADS=1
export HAWOR_DECORD_RECYCLE_AFTER=4096
export HAWOR_S3MOUNT_BIN=/mnt/petrelfs/zhanxinyu/software/s3mount
export HAWOR_MOUNT_ROOT=/tmp/hawor_service/runtime/
export HAWOR_MOUNT_READY_TIMEOUT=30
python -m service.api_server

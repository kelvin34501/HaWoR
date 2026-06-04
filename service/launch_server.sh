#!/usr/bin/env bash

export HAWOR_CACHE_DIR=./example/hawor_process
export HAWOR_GPU_IDS=0
export HAWOR_JOBS_PER_GPU=4   # max concurrent jobs per GPU (default 1)
export HAWOR_CLEANUP_INTERMEDIATE=true
export HAWOR_CLEANUP_FAILED_CACHE=false
export HAWOR_COPY_EXTRACTED_IMAGES=false
export HAWOR_S3MOUNT_BIN=/mnt/petrelfs/share/s3mount
export HAWOR_MOUNT_ROOT=/tmp/hawor_service/runtime/
export HAWOR_MOUNT_READY_TIMEOUT=30
python -m service.api_server

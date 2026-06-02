# HaWoR Folder Annotation Service

This service adds folder-level batch annotation on top of the existing single-video HaWoR pipeline. It scans a local `input_dir`, creates one job, dispatches videos across the configured GPU pool, writes intermediate small files into `cache_dir`, and writes final outputs to `output_dir/<video_stem>/`.

## Install

Install the main HaWoR dependencies first, then install the service-specific HTTP runtime:

```bash
pip install -r requirements.txt
pip install -r service/requirements.txt
```

## Start The Service

Use command-line flags:

```bash
python -m service.api_server --cache-dir /data_nvme/hawor_process --gpu-ids 0,1 --port 8000
```

Or environment variables:

```bash
export HAWOR_CACHE_DIR=/data_nvme/hawor_process
export HAWOR_GPU_IDS=0,1
export HAWOR_CLEANUP_INTERMEDIATE=true
export HAWOR_KEEP_FAILED_CACHE=true
python -m service.api_server
```

Common startup options:

- `--cache-dir`: high-throughput local scratch directory used for chunking and temporary outputs.
- `--gpu-ids`: GPU pool definition such as `0` or `0,1`.
- `--keep-intermediate`: keep successful job cache directories instead of deleting them.
- `--cleanup-failed-cache`: delete failed job cache directories instead of preserving them for debugging.

## API

### Create Job

`POST /v1/annotate`

```bash
curl -X POST http://127.0.0.1:8000/v1/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/mnt/oss/datasets/batch_01",
    "output_dir": "/mnt/oss/annotations/batch_01",
    "vis_mode": "off",
    "overwrite": false
  }'
```

### Query Job

`GET /v1/jobs/<job_id>`

### Query Result

`GET /v1/jobs/<job_id>/result`

## Current Behavior

- Only scans the top level of `input_dir`; no recursive scan.
- `output_dir` is optional. When omitted, the service uses a sibling directory named `<input_dir>_output`.
- The request supports `vis_mode`, `output_dir`, and `overwrite`.
- `img_focal` is not exposed in this version.
- Status values are `PENDING`, `RUNNING`, `SUCCEEDED`, and `FAILED`.
- Progress fields include `videos_total`, `videos_done`, `videos_failed`, and `progress`.
- Intermediate cache is organized under `<cache_dir>/<job_id>/<video_stem>/`.
- Successful jobs clean cache by default; failed jobs are kept by default for debugging.
- Input and output paths must be local paths visible to the service host. Native `s3://` and `petrel-oss` access are not implemented.

Each completed video is written to `output_dir/<video_stem>/`. The directory typically contains `cam_space/`, `SLAM/`, `extracted_images/`, `extracted_images_50fps/` when post-processing is enabled, and `process.log`.
# HaWoR Folder Annotation Service

This service adds folder-level batch annotation on top of the existing single-video HaWoR pipeline. It mounts one object-storage bucket per request on demand, scans one bucket-relative `input_subdir`, creates one job, dispatches videos across the configured GPU pool, writes intermediate small files into `cache_dir`, writes final outputs to `output_subdir/<video_stem>/` inside the bucket, and unmounts the bucket when the job finishes.

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
export HAWOR_CACHE_DIR=./example/data_nvme/hawor_process
export HAWOR_GPU_IDS=0,1
export HAWOR_CLEANUP_INTERMEDIATE=true
export HAWOR_CLEANUP_FAILED_CACHE=false
export HAWOR_COPY_EXTRACTED_IMAGES=false
export HAWOR_S3MOUNT_BIN=/mnt/petrelfs/share_data/s3mount
export HAWOR_MOUNT_ROOT=/mnt/oss
export HAWOR_MOUNT_READY_TIMEOUT=30
python -m service.api_server
```

Common startup options:

- `--cache-dir`: high-throughput local scratch directory used for chunking and temporary outputs.
- `--gpu-ids`: GPU pool definition such as `0` or `0,1`.
- `--cleanup-intermediate` / `--no-cleanup-intermediate`: control whether successful job cache directories are deleted.
- `--cleanup-failed-cache`: delete failed job cache directories instead of preserving them for debugging.
- `--copy-extracted-images` / `--no-copy-extracted-images`: whether to copy `extracted_images` and `extracted_images_50fps` frame directories to the output directory. Disable to save storage when raw frames are not needed (default: enabled). Environment variable: `HAWOR_COPY_EXTRACTED_IMAGES`.
- `--s3mount-bin`: path to the `s3mount` binary (default `s3mount` on `PATH`).
- `--mount-root`: root directory for per-job bucket mounts (default `/mnt/oss`).
- `--mount-ready-timeout`: seconds to wait for a mount to become ready (default `30`).

The service mounts one object-storage bucket per job on demand:

- Each `POST /v1/annotate` carries the bucket connection details (bucket, endpoint, ak/sk, optional prefix/region/flags).
- The bucket is mounted at `<mount-root>/<job_id>/` using `s3mount`, the job runs against bucket-relative paths, and the mount is removed when the job reaches a terminal state.
- `input_subdir` / `output_subdir` are bucket-relative; they are resolved inside the per-job mount and may not escape it.
- Access/secret keys are written to a private `0600` credentials file for the `s3mount` child process only; they are never logged or echoed back.
- A bucket already in use by another active job is rejected with `409`.
- Native `s3://...` paths are rejected in this version.

## API

### Create Job

`POST /v1/annotate`

```bash
curl -X POST http://127.0.0.1:8000/v1/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "bucket": "my-bucket",
    "endpoint": "http://10.140.2.254:80",
    "access_key": "<AK>",
    "secret_key": "<SK>",
    "input_subdir": "datasets/batch_01",
    "output_subdir": "annotations/batch_01",
    "prefix": null,
    "region": null,
    "force_path_style": false,
    "use_listobject_v2": false,
    "vis_mode": "off",
    "overwrite": false
  }'
```

### Query Job

`GET /v1/jobs/<job_id>`

### Query Result

`GET /v1/jobs/<job_id>/result`

### Cancel Job

`POST /v1/jobs/<job_id>/cancel`

### Health Check

`GET /healthz`

Returns `status`, `cache_dir`, `gpu_ids`, cache cleanup flags, `mount_root`, and `mount_ready_timeout`.

## Current Behavior

- Only scans the top level of `input_subdir`; no recursive scan.
- `output_subdir` is optional. When omitted, the service uses `<input_subdir>_output` inside the same bucket.
- The request supports the bucket connection block, `input_subdir`, `output_subdir`, `vis_mode`, and `overwrite`.
- `img_focal` is not exposed in this version.
- Status values are `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, and `CANCELED`.
- `progress` is reported as an integer percentage from `0` to `100`.
- Progress fields include `videos_total`, `videos_done`, `videos_failed`, `progress`, and the job-level `stage` field.
- Intermediate cache is organized under `<cache_dir>/<job_id>/<video_stem>/`.
- Successful jobs clean cache by default; failed or canceled jobs are kept by default for debugging.
- Environment variables use the same semantics as the service config: `HAWOR_CLEANUP_INTERMEDIATE` and `HAWOR_CLEANUP_FAILED_CACHE`.
- One bucket per job: `input_subdir` and `output_subdir` both resolve inside the same mounted bucket.
- The bucket is mounted before the job starts and unmounted when the job finishes; a bucket in use by another active job is rejected.
- Native `s3://` and `petrel-oss` access are not implemented.
- Empty directories, directories without supported video files, subpaths that escape the mount, and existing result directories with `overwrite=false` are rejected before dispatch.

Each completed video is written to `output_dir/<video_stem>/`. The directory typically contains `cam_space/`, `SLAM/`, `extracted_images/`, `extracted_images_50fps/` when post-processing is enabled, and `process.log`.
# HaWoR Folder Annotation Service

This service adds folder-level batch annotation on top of the existing single-video HaWoR pipeline. `input_url` may be either an `s3://bucket/prefix` mounted on demand or a directory on the API server's local filesystem. The service scans the top level of that directory, dispatches videos across the configured GPU pool, writes intermediate files into `cache_dir`, and writes final outputs to `annotations/<video_stem>/` under the input directory.

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
export HAWOR_JOBS_PER_GPU=1
export HAWOR_CLEANUP_INTERMEDIATE=true
export HAWOR_CLEANUP_FAILED_CACHE=false
export HAWOR_COPY_EXTRACTED_IMAGES=false
export HAWOR_RUN_VISUALIZATIONS=false
export HAWOR_NUM_THREADS=1
export HAWOR_DECORD_NUM_THREADS=1
export HAWOR_DECORD_RECYCLE_AFTER=4096
export HAWOR_S3MOUNT_BIN=/mnt/petrelfs/share_data/s3mount
export HAWOR_MOUNT_ROOT=/mnt/oss
export HAWOR_MOUNT_READY_TIMEOUT=30
python -m service.api_server
```

Common startup options:

- `--cache-dir`: high-throughput local scratch directory used for chunking and temporary outputs.
- `--gpu-ids`: GPU pool definition such as `0` or `0,1`.
- `--max-workers`: maximum worker threads used to dispatch videos. Actual concurrent processing is capped by `len(gpu_ids) * HAWOR_JOBS_PER_GPU`.
- `HAWOR_JOBS_PER_GPU`: env-only concurrency multiplier per GPU (default `1`).
- `--cleanup-intermediate` / `--no-cleanup-intermediate`: control whether successful job cache directories are deleted.
- `--cleanup-failed-cache`: delete failed job cache directories instead of preserving them for debugging.
- `--copy-extracted-images` / `--no-copy-extracted-images`: **deprecated no-op.** Frames are now decoded on demand from the video (`FrameSource`) and are never written to disk, so there are no `extracted_images`/`extracted_images_50fps` directories to copy. The flag (and `HAWOR_COPY_EXTRACTED_IMAGES`) is still accepted but has no effect.
- `--run-visualizations` / `--no-run-visualizations` (`HAWOR_RUN_VISUALIZATIONS`): render the cam-space/world-space visualization videos (`cam_space_visualization_*fps.mp4`, `world_space_visualization_*fps.mp4`) after processing each video (default `false`).
- `HAWOR_NUM_THREADS`: CPU inference-pool threads per video subprocess (service-child default `1`; an explicit operator value is preserved).
- `--decord-num-threads` (`HAWOR_DECORD_NUM_THREADS`): Decord threads per video process (default `1`). Values must be positive.
- `--decord-recycle-after` (`HAWOR_DECORD_RECYCLE_AFTER`): decoded frames per native reader before it is released and reopened (default `4096`, longer than the maximum 3,001-frame service window). Values must be positive.
- `--s3mount-bin`: path to the `s3mount` binary (default `s3mount` on `PATH`).
- `--mount-root`: root directory for per-job bucket mounts (default `/mnt/oss`).
- `--mount-ready-timeout`: seconds to wait for a mount to become ready (default `30`).

For S3 input, the service mounts one object-storage bucket per job on demand:

- Each `POST /v1/annotate` carries `input_url` (`s3://bucket/prefix`) plus endpoint, access/secret keys, optional region, and mount flags.
- The requested bucket/prefix is mounted at `<mount-root>/<job_id>/video_in` using `s3mount`, and the mount is removed when the job reaches a terminal state.
- The service scans the mounted input prefix itself and writes results to the fixed `annotations/` directory inside that same prefix.
- Access/secret keys are passed only to the `s3mount` child process environment; they are never logged or echoed back.
- A bucket already in use by another active job is rejected with `409`.
- Native `s3://...` filesystem paths are not used internally; the public API accepts `s3://...` only through `input_url`.

For local input, set `input_url` to a directory path visible to the API server.
S3 credentials and mount options may be omitted. The service does not copy or
mount the directory and writes results directly to its `annotations/` child.
Relative paths resolve from the API server's working directory. The API process
must have read/execute access to the input directory and write access to it for
creating results. Exposing local paths allows API clients to access directories
available to the service account, so the endpoint should only be used by trusted
clients.

## API

### Create Job

`POST /v1/annotate`

```bash
curl -X POST http://127.0.0.1:8000/v1/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "input_url": "s3://my-bucket/datasets/batch_01",
    "endpoint": "http://10.140.2.254:80",
    "access_key": "<AK>",
    "secret_key": "<SK>",
    "region": null,
    "force_path_style": false,
    "use_listobject_v2": false,
    "vis_mode": "off",
    "overwrite": false,
    "skip_processed": false
  }'
```

Local-directory request:

```bash
curl -X POST http://127.0.0.1:8000/v1/annotate \
  -H 'Content-Type: application/json' \
  -d '{
    "input_url": "/data/videos/batch_01",
    "vis_mode": "off",
    "overwrite": false,
    "skip_processed": false
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

Returns `status`, `cache_dir`, `gpu_ids`, cache cleanup flags, `mount_root`,
`mount_ready_timeout`, `decord_num_threads`, and `decord_recycle_after`.

## Current Behavior

- Only scans the top level of the `input_url` directory or prefix; no recursive scan.
- Outputs always go to `annotations/<video_stem>/` under the same input directory.
- `input_url` accepts an `s3://bucket/prefix` or a local directory. `endpoint`, access/secret keys, `region`, `force_path_style`, and `use_listobject_v2` apply only to S3 input.
- `img_focal` is not exposed in this version.
- Status values are `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELED`, and item-level `SKIPPED`.
- `progress` is reported as an integer percentage from `0` to `100`.
- Progress fields include `videos_total`, `videos_done`, `videos_failed`, `videos_skipped`, `progress`, and the job-level `stage` field.
- Every video item reports `peak_rss_bytes`, the sampled peak of its processing subprocess tree. The same value and a pass/fail result against the 20 GiB acceptance threshold are appended to `process.log`; over-budget jobs are reported, not killed.
- Intermediate cache is organized under `<cache_dir>/<job_id>/<video_stem>/`.
- Successful jobs clean cache by default; failed or canceled jobs are kept by default for debugging.
- Environment variables use the same semantics as the service config: `HAWOR_CLEANUP_INTERMEDIATE` and `HAWOR_CLEANUP_FAILED_CACHE`.
- For S3 jobs, input videos and `annotations/` outputs both live inside the same mounted prefix. The bucket is mounted before the job starts and unmounted when it finishes.
- For local jobs, the input directory is used directly and no mount or unmount is performed.
- `skip_processed=true` skips videos whose result directory already contains `process.done`.
- Empty directories, directories without supported video files, subpaths that escape the mount, and existing result directories with `overwrite=false` are rejected before dispatch.

Each completed video is written to `annotations/<video_stem>/` under the input directory. The directory contains `cam_space/`, `SLAM/`, `process.log`, `process.done`, and — when post-processing is enabled — `cam_space_50fps/` plus the interpolated 50fps SLAM artifacts. Visualization outputs are saved with the rendered FPS in the filename, for example `cam_space_visualization_50fps.mp4` / `world_space_visualization_50fps.mp4` when interpolation is available, or `cam_space_visualization_30fps.mp4` / `world_space_visualization_30fps.mp4` on the base timeline. Frames are decoded on demand and are **not** written to disk, so there are no `extracted_images*` directories.

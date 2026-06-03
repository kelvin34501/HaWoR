from __future__ import annotations

import shutil
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from service.config import ServiceConfig
from service.hawor_video_processor_for_service import (
    GpuPool,
    HaWoRProcessorConfig,
    HaWoRVideoProcessorForService,
)
from service.s3mount_manager import MountHandle, MountSpec, S3MountManager
from service.storage import (
    JobPathPlan,
    S3Url,
    build_job_path_plan,
    validate_vis_mode,
    video_result_dir,
)

#: Output annotations live alongside the input videos inside the mounted prefix,
#: i.e. ``<mount_root>/<job_id>/video_in/<JOB_OUTPUT_SUBDIR>``.
JOB_OUTPUT_SUBDIR = "annotations"


class JobNotFoundError(KeyError):
    pass


class JobNotReadyError(RuntimeError):
    pass


@dataclass
class VideoItem:
    video: str
    status: str
    result_dir: str
    log_path: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "status": self.status,
            "result_dir": self.result_dir,
            "log_path": self.log_path,
            "error": self.error,
        }


@dataclass
class JobRecord:
    job_id: str
    status: str
    stage: str
    input_dir: str
    output_dir: str
    videos_total: int
    error: Optional[str] = None
    cancel_requested: bool = False
    items: list[VideoItem] = field(default_factory=list)
    mount_handle: Optional[MountHandle] = None

    @property
    def videos_done(self) -> int:
        return sum(1 for item in self.items if item.status == "SUCCEEDED")

    @property
    def videos_failed(self) -> int:
        return sum(1 for item in self.items if item.status == "FAILED")

    @property
    def progress(self) -> int:
        if self.videos_total == 0:
            return 0
        terminal = sum(1 for item in self.items if item.status in {"SUCCEEDED", "FAILED", "CANCELED"})
        return int((terminal / self.videos_total) * 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "videos_total": self.videos_total,
            "videos_done": self.videos_done,
            "videos_failed": self.videos_failed,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "items": [item.to_dict() for item in self.items],
        }


class JobManager:

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._gpu_pool = GpuPool(config.gpu_ids)
        requested_workers = config.max_workers or self._gpu_pool.size
        self._dispatch_limit = max(1, min(requested_workers, self._gpu_pool.size))
        self._executor = ThreadPoolExecutor(
            max_workers=requested_workers,
            thread_name_prefix="hawor-job",
        )
        self._mount_manager = S3MountManager(
            s3mount_bin=config.s3mount_bin,
            mount_root=config.mount_root,
            ready_timeout=config.mount_ready_timeout,
        )

    def create_job(
        self,
        *,
        input_url: str,
        endpoint: str,
        access_key: str,
        secret_key: str,
        region: Optional[str],
        force_path_style: bool,
        use_listobject_v2: bool,
        vis_mode: str,
        overwrite: bool,
    ) -> dict[str, Any]:
        normalized_vis_mode = validate_vis_mode(vis_mode)
        s3_url = S3Url.from_url(input_url)
        mount_spec = MountSpec(
            bucket=s3_url.bucket,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            prefix=s3_url.prefix,
            region=region,
            force_path_style=force_path_style,
            use_listobject_v2=use_listobject_v2,
            read_only=False,
        )
        job_id = uuid.uuid4().hex
        mount_handle = self._mount_manager.mount(job_id, mount_spec)
        try:
            input_dir = mount_handle.mount_dir
            output_dir = mount_handle.mount_dir / JOB_OUTPUT_SUBDIR
            plan = build_job_path_plan(
                input_dir,
                output_dir,
                overwrite=overwrite,
                s3mount_prefixes=(self.config.mount_root,),
            )
        except BaseException:
            self._mount_manager.unmount(mount_handle)
            raise

        job = JobRecord(
            job_id=job_id,
            status="PENDING",
            stage="QUEUED",
            input_dir=str(plan.input_dir),
            output_dir=str(plan.output_dir),
            videos_total=len(plan.video_paths),
            mount_handle=mount_handle,
            items=[
                VideoItem(
                    video=video_path.name,
                    status="PENDING",
                    result_dir=str(video_result_dir(plan.output_dir, video_path)),
                    log_path=str(video_result_dir(plan.output_dir, video_path) / "process.log"),
                ) for video_path in plan.video_paths
            ],
        )

        with self._lock:
            self._jobs[job_id] = job

        runner = threading.Thread(
            target=self._run_job,
            args=(job_id, plan, normalized_vis_mode, overwrite),
            daemon=True,
            name=f"hawor-job-{job_id}",
        )
        runner.start()
        return job.to_dict()

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job.to_dict()

    def get_result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.status not in {"SUCCEEDED", "FAILED", "CANCELED"}:
                raise JobNotReadyError(f"Job is still running: {job.status}")
            return {
                "job_id": job.job_id,
                "status": job.status,
                "stage": job.stage,
                "output_dir": job.output_dir,
                "items": [item.to_dict() for item in job.items],
                "error": job.error,
            }

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.status in {"SUCCEEDED", "FAILED", "CANCELED"}:
                return job.to_dict()

            job.cancel_requested = True
            if job.status == "PENDING":
                job.status = "CANCELED"
                job.stage = "CANCELED"
                self._mark_pending_items_canceled(job_id)
            else:
                job.stage = "CANCEL_REQUESTED"
            return job.to_dict()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _unmount_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            handle = job.mount_handle if job is not None else None
            if job is not None:
                job.mount_handle = None
        if handle is not None:
            self._mount_manager.unmount(handle)

    def _run_job(
        self,
        job_id: str,
        plan: JobPathPlan,
        vis_mode: str,
        overwrite: bool,
    ) -> None:
        if self._is_cancel_requested(job_id):
            self._update_job(job_id, status="CANCELED", stage="CANCELED")
            self._unmount_job(job_id)
            return

        self._update_job(job_id, status="RUNNING", stage="DISPATCHING")
        processor = self._build_processor(vis_mode)
        job_cache_dir = self.config.cache_dir / job_id
        job_cache_dir.mkdir(parents=True, exist_ok=True)

        job_failed = False
        job_error: Optional[str] = None
        pending_videos = iter(plan.video_paths)
        futures: dict[Future, Path] = {}
        try:
            self._fill_inflight(
                job_id,
                futures,
                pending_videos,
                processor,
                plan.output_dir,
                job_cache_dir,
                overwrite,
            )

            while futures:
                done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    video_path = futures.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        job_failed = True
                        if job_error is None:
                            job_error = str(exc)
                        self._mark_video_failed(job_id, video_path, str(exc))
                    else:
                        self._mark_video_succeeded(job_id, video_path)

                self._fill_inflight(
                    job_id,
                    futures,
                    pending_videos,
                    processor,
                    plan.output_dir,
                    job_cache_dir,
                    overwrite,
                )
        except Exception as exc:
            job_failed = True
            if job_error is None:
                job_error = str(exc)
        finally:
            canceled = self._is_cancel_requested(job_id)
            if canceled:
                self._mark_pending_items_canceled(job_id)

            should_cleanup_cache = (((not job_failed and not canceled) and self.config.cleanup_intermediate) or
                                    ((job_failed or canceled) and self.config.cleanup_failed_cache))
            if should_cleanup_cache and job_cache_dir.exists():
                shutil.rmtree(job_cache_dir, ignore_errors=True)

            if canceled:
                final_status = "CANCELED"
            else:
                final_status = "FAILED" if job_failed else "SUCCEEDED"

            self._update_job(
                job_id,
                status=final_status,
                stage=final_status,
                error=job_error,
            )
            self._unmount_job(job_id)

    def _build_processor(self, vis_mode: str) -> HaWoRVideoProcessorForService:
        return HaWoRVideoProcessorForService(
            config=HaWoRProcessorConfig(
                vis_mode=vis_mode,
                cleanup_intermediate=False,
            ),
            gpu_pool=self._gpu_pool,
        )

    def _process_single_video(
        self,
        job_id: str,
        processor: HaWoRVideoProcessorForService,
        video_path: Path,
        output_dir: Path,
        scratch_dir: Path,
        overwrite: bool,
    ) -> None:
        self._set_video_status(job_id, video_path.name, "RUNNING")
        self._update_job(job_id, stage="PROCESSING")
        processor.process_video(
            video_path,
            output_dir,
            scratch_dir=scratch_dir,
            overwrite_output=overwrite,
        )

    def _mark_video_succeeded(self, job_id: str, video_path: Path) -> None:
        with self._lock:
            job = self._require_job(job_id)
            item = self._require_item(job, video_path.name)
            item.status = "SUCCEEDED"
            item.error = None
            if self._has_active_work(job):
                job.stage = "PROCESSING"

    def _mark_video_failed(self, job_id: str, video_path: Path, error: str) -> None:
        with self._lock:
            job = self._require_job(job_id)
            item = self._require_item(job, video_path.name)
            item.status = "FAILED"
            item.error = error
            if self._has_active_work(job):
                job.stage = "PROCESSING"

    def _set_video_status(self, job_id: str, video_name: str, status: str) -> None:
        with self._lock:
            job = self._require_job(job_id)
            item = self._require_item(job, video_name)
            item.status = status
            if status == "RUNNING":
                job.stage = "PROCESSING"

    def _update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._require_job(job_id)
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if error is not None or status == "SUCCEEDED":
                job.error = error

    def _require_job(self, job_id: str) -> JobRecord:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def _require_item(self, job: JobRecord, video_name: str) -> VideoItem:
        for item in job.items:
            if item.video == video_name:
                return item
        raise KeyError(f"Video item not found for job {job.job_id}: {video_name}")

    def _fill_inflight(
        self,
        job_id: str,
        futures: dict[Future, Path],
        pending_videos: Any,
        processor: HaWoRVideoProcessorForService,
        output_root: Path,
        job_cache_dir: Path,
        overwrite: bool,
    ) -> None:
        while len(futures) < self._dispatch_limit and not self._is_cancel_requested(job_id):
            try:
                video_path = next(pending_videos)
            except StopIteration:
                return

            future = self._executor.submit(
                self._process_single_video,
                job_id,
                processor,
                video_path,
                video_result_dir(output_root, video_path),
                job_cache_dir / video_path.stem,
                overwrite,
            )
            futures[future] = video_path

    def _mark_pending_items_canceled(self, job_id: str) -> None:
        with self._lock:
            job = self._require_job(job_id)
            for item in job.items:
                if item.status == "PENDING":
                    item.status = "CANCELED"
                    item.error = "Canceled before processing"

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._require_job(job_id)
            return job.cancel_requested or job.status == "CANCELED"

    def _has_active_work(self, job: JobRecord) -> bool:
        return any(item.status in {"PENDING", "RUNNING"} for item in job.items)

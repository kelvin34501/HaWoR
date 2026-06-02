from __future__ import annotations

import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from service.config import ServiceConfig
from service.hawor_video_processor_for_service import (
    GpuPool,
    HaWoRProcessorConfig,
    HaWoRVideoProcessorForService,
)
from service.storage import JobPathPlan, build_job_path_plan, validate_vis_mode, video_result_dir


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
    input_dir: str
    output_dir: str
    videos_total: int
    videos_done: int = 0
    videos_failed: int = 0
    progress: float = 0.0
    error: Optional[str] = None
    items: list[VideoItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "videos_total": self.videos_total,
            "videos_done": self.videos_done,
            "videos_failed": self.videos_failed,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "error": self.error,
            "items": [item.to_dict() for item in self.items],
        }


class JobManager:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._gpu_pool = GpuPool(config.gpu_ids)
        max_workers = config.max_workers or self._gpu_pool.size
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hawor-job")

    def create_job(
        self,
        *,
        input_dir: str,
        output_dir: Optional[str],
        vis_mode: str,
        overwrite: bool,
    ) -> dict[str, Any]:
        normalized_vis_mode = validate_vis_mode(vis_mode)
        plan = build_job_path_plan(input_dir, output_dir, overwrite=overwrite)
        job_id = uuid.uuid4().hex
        job = JobRecord(
            job_id=job_id,
            status="PENDING",
            input_dir=str(plan.input_dir),
            output_dir=str(plan.output_dir),
            videos_total=len(plan.video_paths),
            items=[
                VideoItem(
                    video=video_path.name,
                    status="PENDING",
                    result_dir=str(video_result_dir(plan.output_dir, video_path)),
                    log_path=str(video_result_dir(plan.output_dir, video_path) / "process.log"),
                )
                for video_path in plan.video_paths
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
            if job.status not in {"SUCCEEDED", "FAILED"}:
                raise JobNotReadyError(f"Job is still running: {job.status}")
            return {
                "job_id": job.job_id,
                "status": job.status,
                "output_dir": job.output_dir,
                "items": [item.to_dict() for item in job.items],
                "error": job.error,
            }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run_job(
        self,
        job_id: str,
        plan: JobPathPlan,
        vis_mode: str,
        overwrite: bool,
    ) -> None:
        self._update_job(job_id, status="RUNNING")
        processor = self._build_processor(vis_mode)
        job_cache_dir = self.config.cache_dir / job_id
        job_cache_dir.mkdir(parents=True, exist_ok=True)

        futures: dict[Future, Path] = {}
        for video_path in plan.video_paths:
            futures[
                self._executor.submit(
                    self._process_single_video,
                    job_id,
                    processor,
                    video_path,
                    video_result_dir(plan.output_dir, video_path),
                    job_cache_dir / video_path.stem,
                    overwrite,
                )
            ] = video_path

        job_failed = False
        job_error: Optional[str] = None
        try:
            for future in as_completed(futures):
                video_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    job_failed = True
                    if job_error is None:
                        job_error = str(exc)
                    self._mark_video_failed(job_id, video_path, str(exc))
                else:
                    self._mark_video_succeeded(job_id, video_path)
        except Exception as exc:
            job_failed = True
            if job_error is None:
                job_error = str(exc)
        finally:
            should_cleanup_cache = self.config.cleanup_intermediate and (
                not job_failed or not self.config.keep_failed_cache
            )
            if should_cleanup_cache and job_cache_dir.exists():
                shutil.rmtree(job_cache_dir, ignore_errors=True)

            final_status = "FAILED" if job_failed else "SUCCEEDED"
            self._update_job(job_id, status=final_status, error=job_error)

    def _build_processor(self, vis_mode: str) -> HaWoRVideoProcessorForService:
        return HaWoRVideoProcessorForService(
            config=HaWoRProcessorConfig(
                vis_mode=vis_mode,
                cleanup_intermediate=self.config.cleanup_intermediate,
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
            job.videos_done += 1
            job.progress = self._calculate_progress(job)

    def _mark_video_failed(self, job_id: str, video_path: Path, error: str) -> None:
        with self._lock:
            job = self._require_job(job_id)
            item = self._require_item(job, video_path.name)
            item.status = "FAILED"
            item.error = error
            job.videos_failed += 1
            job.progress = self._calculate_progress(job)

    def _set_video_status(self, job_id: str, video_name: str, status: str) -> None:
        with self._lock:
            job = self._require_job(job_id)
            item = self._require_item(job, video_name)
            item.status = status

    def _update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._require_job(job_id)
            if status is not None:
                job.status = status
            if error is not None or status == "SUCCEEDED":
                job.error = error
            job.progress = self._calculate_progress(job)

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

    def _calculate_progress(self, job: JobRecord) -> float:
        if job.videos_total == 0:
            return 0.0
        return round((job.videos_done + job.videos_failed) / job.videos_total, 4)
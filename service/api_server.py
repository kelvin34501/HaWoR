from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from service.config import ServiceConfig, load_service_config
from service.job_manager import JobManager, JobNotFoundError, JobNotReadyError
from service.s3mount_manager import BucketBusyError, MountError


class AnnotateRequest(BaseModel):
    input_url: str = Field(
        ...,
        description="s3://bucket/path link to the directory of input videos to mount",
    )
    endpoint: str = Field(..., description="Object-storage endpoint URL")
    access_key: str = Field(..., description="Access key id (sensitive, never logged)")
    secret_key: str = Field(..., description="Secret access key (sensitive, never logged)")
    region: Optional[str] = Field(default=None, description="Optional region (e.g. oss-cn-beijing)")
    force_path_style: bool = Field(default=False, description="Set for domain-style endpoints")
    use_listobject_v2: bool = Field(default=False, description="Set for backends requiring ListObjectsV2")
    vis_mode: str = Field(default="off", description="off | cam | world")
    overwrite: bool = Field(default=False, description="Overwrite existing per-video outputs")


def create_app(service_config: Optional[ServiceConfig] = None) -> FastAPI:
    config = service_config or load_service_config()
    job_manager = JobManager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.job_manager = job_manager
        app.state.service_config = config
        try:
            yield
        finally:
            job_manager.shutdown()

    app = FastAPI(title="HaWoR Annotation Service", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "cache_dir": str(config.cache_dir),
            "gpu_ids": str(config.gpu_ids),
            "cleanup_intermediate": config.cleanup_intermediate,
            "cleanup_failed_cache": config.cleanup_failed_cache,
            "mount_root": str(config.mount_root),
            "mount_ready_timeout": config.mount_ready_timeout,
        }

    @app.post("/v1/annotate", status_code=202)
    def create_annotation_job(payload: AnnotateRequest) -> dict:
        try:
            return job_manager.create_job(
                input_url=payload.input_url,
                endpoint=payload.endpoint,
                access_key=payload.access_key,
                secret_key=payload.secret_key,
                region=payload.region,
                force_path_style=payload.force_path_style,
                use_listobject_v2=payload.use_listobject_v2,
                vis_mode=payload.vis_mode,
                overwrite=payload.overwrite,
            )
        except BucketBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except MountError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (NotADirectoryError, PermissionError, FileExistsError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return job_manager.get_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc

    @app.get("/v1/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> dict:
        try:
            return job_manager.get_result(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc
        except JobNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        try:
            return job_manager.cancel_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc

    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HaWoR folder-level annotation service")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-ids", default=None, help='GPU pool, e.g. "0" or "0,1"')
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--s3mount-bin", default=None, help="Path to the s3mount binary")
    parser.add_argument("--mount-root", default=None, help="Root directory for per-job bucket mounts")
    parser.add_argument("--mount-ready-timeout",
                        type=float,
                        default=None,
                        help="Seconds to wait for a bucket mount to become ready")
    parser.add_argument(
        "--cleanup-intermediate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clean per-video cache directories after successful completion",
    )
    parser.add_argument(
        "--cleanup-failed-cache",
        action="store_true",
        help="Delete failed job cache directories instead of keeping them for debugging",
    )
    parser.add_argument(
        "--copy-extracted-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Copy extracted_images / extracted_images_50fps frames to the output directory (default: true)",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_service_config(
        cache_dir=args.cache_dir,
        gpu_ids=args.gpu_ids,
        cleanup_intermediate=args.cleanup_intermediate,
        cleanup_failed_cache=args.cleanup_failed_cache,
        copy_extracted_images=args.copy_extracted_images,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        s3mount_bin=args.s3mount_bin,
        mount_root=args.mount_root,
        mount_ready_timeout=args.mount_ready_timeout,
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port)


app = create_app()

if __name__ == "__main__":
    main()

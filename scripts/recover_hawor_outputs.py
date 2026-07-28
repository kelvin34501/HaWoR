#!/usr/bin/env python3
"""Copy completed HaWoR cache artifacts back to the dataset.

For every cached ``process.log``, the script reads the logged ``--video_path``
and restores the API-server layout:

    DATASET_PREFIX/subdir/video.mp4
      -> DATASET_PREFIX/subdir/annotations/video/

``process.done`` is written last, after all files have been checked by size.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


VIDEO_PATH_RE = re.compile(
    r"(?:^|\s)--video_path\s+(.*?)(?=\s--[A-Za-z0-9_-]+(?:\s|$)|$)"
)


@dataclass(frozen=True)
class CachedResult:
    process_log: Path
    artifacts: Path
    destination: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover HaWoR cache outputs into annotations/<video-stem>/.",
        epilog=(
            "Example: python scripts/recover_hawor_outputs.py "
            "example/hawor_process /oss/locomanip/SciLabVideo"
        ),
    )
    parser.add_argument(
        "cache_root",
        type=Path,
        help="service cache containing per-video process.log files",
    )
    parser.add_argument(
        "dataset_prefix",
        type=Path,
        help="dataset root containing paths such as subdir/video.mp4",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show incomplete outputs without copying",
    )
    return parser.parse_args(argv)


def video_path_from_log(process_log: Path) -> Path:
    """Return the first ``--video_path`` from a logged service command."""
    with process_log.open("r", encoding="utf-8", errors="replace") as log:
        for line in log:
            if not line.lstrip().startswith("$ ") or "--video_path" not in line:
                continue
            match = VIDEO_PATH_RE.search(line)
            if match:
                value = match.group(1).strip().strip("'\"")
                if value:
                    return Path(value).expanduser()
    raise ValueError("no logged command with --video_path")


def discover(
    cache_root: Path,
    dataset_prefix: Path,
) -> tuple[list[CachedResult], list[tuple[Path, str]]]:
    """Find recoverable cache entries and reject ambiguous ones."""
    found: list[CachedResult] = []
    failures: list[tuple[Path, str]] = []

    for process_log in sorted(cache_root.rglob("process.log")):
        artifacts = process_log.parent / "_final_output"
        try:
            if not artifacts.is_dir():
                raise ValueError("missing adjacent _final_output directory")
            if not any(path.is_file() for path in artifacts.rglob("*")):
                raise ValueError("_final_output contains no files")

            video_path = video_path_from_log(process_log)
            if not video_path.is_absolute():
                raise ValueError(f"video path is not absolute: {video_path}")
            video_path = video_path.resolve()
            if video_path.suffix.lower() != ".mp4":
                raise ValueError(f"video is not an MP4: {video_path}")
            try:
                relative_video = video_path.relative_to(dataset_prefix)
            except ValueError as exc:
                raise ValueError(f"video is outside dataset prefix: {video_path}") from exc

            destination = (
                dataset_prefix
                / relative_video.parent
                / "annotations"
                / video_path.stem
            )
            found.append(CachedResult(process_log, artifacts, destination))
        except (OSError, ValueError) as exc:
            failures.append((process_log, str(exc)))

    # Do not guess which cached attempt to use when jobs overlap.
    grouped: dict[Path, list[CachedResult]] = defaultdict(list)
    for result in found:
        grouped[result.destination].append(result)

    unique: list[CachedResult] = []
    for destination, results in grouped.items():
        if len(results) == 1:
            unique.append(results[0])
        else:
            locations = ", ".join(str(item.process_log.parent) for item in results)
            for item in results:
                failures.append(
                    (item.process_log, f"duplicate destination {destination}: {locations}")
                )
    return sorted(unique, key=lambda item: str(item.destination)), failures


def run_rclone(rclone: str, *arguments: str) -> None:
    completed = subprocess.run(
        [rclone, *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode == 0:
        return
    detail = (completed.stderr or completed.stdout).strip()
    if len(detail) > 800:
        detail = f"{detail[-800:]} (output truncated)"
    raise RuntimeError(detail or f"rclone exited with status {completed.returncode}")


def artifacts_match(result: CachedResult, rclone: str) -> tuple[bool, str]:
    destination_log = result.destination / "process.log"
    try:
        if not destination_log.is_file():
            return False, "process.log is missing"
        if destination_log.stat().st_size != result.process_log.stat().st_size:
            return False, "process.log size differs"
        run_rclone(
            rclone,
            "check",
            str(result.artifacts),
            str(result.destination),
            "--one-way",
            "--size-only",
        )
    except (OSError, RuntimeError) as exc:
        return False, str(exc)
    return True, ""


def is_complete(result: CachedResult, rclone: str) -> tuple[bool, str]:
    sentinel = result.destination / "process.done"
    try:
        if not sentinel.is_file() or sentinel.stat().st_size == 0:
            return False, "process.done is missing or empty"
    except OSError as exc:
        return False, str(exc)
    return artifacts_match(result, rclone)


def repair(result: CachedResult, rclone: str) -> None:
    """Copy artifacts and publish a non-empty sentinel only after verification."""
    sentinel = result.destination / "process.done"
    if sentinel.exists():
        sentinel.unlink()  # Remove a stale success marker before repairing.

    run_rclone(
        rclone,
        "copy",
        str(result.artifacts),
        str(result.destination),
        "--size-only",
    )
    run_rclone(
        rclone,
        "copyto",
        str(result.process_log),
        str(result.destination / "process.log"),
        "--size-only",
    )

    matched, reason = artifacts_match(result, rclone)
    if not matched:
        raise RuntimeError(f"post-copy verification failed: {reason}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="hawor_process_done_", delete=False
        ) as temporary:
            temporary.write("x")
            temporary_path = Path(temporary.name)
        run_rclone(rclone, "copyto", str(temporary_path), str(sentinel))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if not sentinel.is_file() or sentinel.stat().st_size == 0:
        raise RuntimeError("published process.done is missing or empty")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_root = args.cache_root.expanduser().resolve()
    dataset_prefix = args.dataset_prefix.expanduser().resolve()

    if not cache_root.is_dir() or not dataset_prefix.is_dir():
        print("error: cache_root and dataset_prefix must be directories", file=sys.stderr)
        return 2
    rclone = shutil.which("rclone")
    if rclone is None:
        print("error: rclone was not found in PATH", file=sys.stderr)
        return 2

    results, failures = discover(cache_root, dataset_prefix)
    for process_log, reason in failures:
        print(f"[failed] {process_log}: {reason}")

    counts = {"complete": 0, "repaired": 0, "would_repair": 0}
    failed_count = len(failures)
    for result in results:
        complete, reason = is_complete(result, rclone)
        if complete:
            counts["complete"] += 1
            print(f"[complete] {result.destination}")
        elif args.dry_run:
            counts["would_repair"] += 1
            print(f"[would repair] {result.destination}: {reason}")
        else:
            try:
                repair(result, rclone)
                counts["repaired"] += 1
                print(f"[repaired] {result.destination}")
            except (OSError, RuntimeError) as exc:
                failed_count += 1
                print(f"[failed] {result.destination}: {exc}")

    print(
        f"Summary: found={len(results) + len(failures)} "
        f"complete={counts['complete']} repaired={counts['repaired']} "
        f"would_repair={counts['would_repair']} failed={failed_count}"
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

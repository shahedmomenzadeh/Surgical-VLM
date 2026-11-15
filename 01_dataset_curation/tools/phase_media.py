"""Base64 media preparation for the phase SFT teacher/critic.

The local OpenAI-compatible endpoint receives a complete video first. Callers
can switch to the frame helpers when the endpoint reports an upstream or
payload failure.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import subprocess
import tempfile
from pathlib import Path


def encode_video_data_uri(video_path: Path, max_video_bytes: int = 0) -> str:
    """Encode an MP4 as a base64 data URI."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    size = video_path.stat().st_size
    if size <= 0:
        raise ValueError(f"Video is empty: {video_path}")
    if max_video_bytes > 0 and size > max_video_bytes:
        raise ValueError(
            f"Video is {size} bytes, above max_video_bytes={max_video_bytes}")

    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def is_frame_fallback_error(error: str) -> bool:
    """Return whether a failed full-video request should use frames."""
    text = (error or "").lower()
    return any(marker in text for marker in (
        "503", "502", "504", "413", "upstream error",
        "service unavailable", "gateway timeout", "payload too large",
    ))


def count_video_frames(video_path: Path) -> int:
    """Count decoded video frames with ffprobe, returning zero if unknown."""
    commands = [
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(video_path),
        ],
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(video_path),
        ],
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True)
            value = result.stdout.strip()
            if value.isdigit() and int(value) > 0:
                return int(value)
        except (OSError, subprocess.CalledProcessError):
            continue
    return 0


def _quality_to_ffmpeg_qscale(quality: int) -> int:
    quality = max(1, min(100, int(quality)))
    return max(2, min(31, round(2 + (100 - quality) * 0.29)))


def _frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= max_frames:
        return list(range(frame_count))
    if max_frames == 1:
        return [0]
    return sorted({
        round(index * (frame_count - 1) / (max_frames - 1))
        for index in range(max_frames)
    })


def extract_frame_data_uris(
    video_path: Path,
    max_frames: int = 32,
    quality: int = 90,
) -> tuple[list[str], int]:
    """Extract at most ``max_frames`` evenly spaced JPEG data URIs.

    The actual frame count is checked before selecting frames. If ffprobe
    cannot report it, all decoded frames are temporarily materialized so the
    selection still cannot request more frames than the video contains.
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    actual_count = count_video_frames(video_path)
    qscale = str(_quality_to_ffmpeg_qscale(quality))

    with tempfile.TemporaryDirectory(prefix="phase_frames_") as temp_dir:
        temp_path = Path(temp_dir)
        if actual_count > 0:
            indices = _frame_indices(actual_count, max_frames)
            select = "+".join(f"eq(n\\,{index})" for index in indices)
            output_pattern = temp_path / "frame_%06d.jpg"
            command = [
                "ffmpeg", "-v", "error", "-i", str(video_path),
                "-vf", f"select={select}", "-vsync", "vfr",
                "-q:v", qscale, str(output_pattern),
            ]
            subprocess.run(command, check=True)
            frame_paths = sorted(temp_path.glob("frame_*.jpg"))
        else:
            # This is the exact-count fallback when container metadata is
            # incomplete. The decoded files are then counted before selection.
            output_pattern = temp_path / "all_%08d.jpg"
            command = [
                "ffmpeg", "-v", "error", "-i", str(video_path),
                "-vsync", "0", "-q:v", qscale, str(output_pattern),
            ]
            subprocess.run(command, check=True)
            all_paths = sorted(temp_path.glob("all_*.jpg"))
            actual_count = len(all_paths)
            if not actual_count:
                raise ValueError(f"No decodable frames found in {video_path}")
            selected = _frame_indices(actual_count, max_frames)
            frame_paths = [all_paths[index] for index in selected]

        if not frame_paths:
            raise ValueError(f"No frames extracted from {video_path}")

        data_uris = []
        for frame_path in frame_paths:
            encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
            data_uris.append(f"data:image/jpeg;base64,{encoded}")
        return data_uris, actual_count

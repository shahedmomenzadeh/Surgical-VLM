# `video_pipeline/common/utils.py` — Shared Utilities

Small, dependency-light helpers used across the pipeline.

## Functions

| Function | Purpose |
|----------|---------|
| `time_str_to_seconds(ts)` | `MM:SS` / `HH:MM:SS` → seconds; `None` on unparsable input or `?` placeholders |
| `seconds_to_mm_ss(seconds)` | inverse; auto-uses `HH:MM:SS` above one hour |
| `extract_youtube_id(url)` | parse video ID from `youtu.be/…` or `watch?v=…` URLs |
| `normalize_id(name)` | strip all non-alphanumeric chars + lowercase — used for robust filename matching (handles `™` → `__` sanitization) |
| `is_ffmpeg_installed()` | `shutil.which("ffmpeg")` check |

## Why it exists

Previously each stage reimplemented its own timestamp parser
(`time_str_to_seconds`, `timestamp_to_seconds`, `parse_timestamp` were
duplicated 4× across the codebase). Everything now uses this module.

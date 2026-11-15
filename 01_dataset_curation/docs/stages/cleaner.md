# `video_pipeline/stages/cleaner.py` — Cleaner (Step 2)

## Purpose

Hygiene step: deletes videos (and their audio + metadata rows) that exceed
the configured duration threshold (`max_duration_seconds`), so very long
files never reach the expensive downstream stages.

## Flow

1. `find_videos_to_remove()` reads the metadata CSV and splits rows into
   keep / remove based on `duration_seconds`.
2. Optionally asks for confirmation (`auto_confirm=True` from the CLI).
3. `delete_videos()` removes the video and audio files from disk and
   rewrites the metadata CSV without the removed rows.

## Inputs / Outputs

| | Path |
|---|---|
| In | `videos/video_metadata.csv` |
| Out | same CSV (rows removed), deleted files |

## Configuration

- `download.max_duration_seconds` (default 1500 s)

## Notes / Quirks

- Rows with non-numeric durations are treated as 0 (kept).
- `duration_numeric` is a temporary column dropped before saving.
- If the CSV is missing, the stage reports an error and exits safely.

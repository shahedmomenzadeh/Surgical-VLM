# `video_pipeline/stages/splitter.py` — Splitter (Step 9)

## Purpose

Cuts each full-length surgery video into step-level clips using the VLM
annotation timestamps. Pure FFmpeg work, no LLM involved.

## Flow

1. For each `vlm_dataset/{video_id}.jsonl` (aggregate excluded):
2. `find_video()` locates the source video — exact `original_filename` in
   `videos/`, then `{video_id}.ext` fallback.
3. For each annotation step:
   - `time_str_to_seconds()` parses `MM:SS` / `HH:MM:SS` (skips invalid or
     inverted ranges);
   - `extract_clip()` runs FFmpeg (`-ss` start, `-t` duration, libx264 CRF 22
     re-encode, audio copied);
   - writes `clip_dataset/{video_id}/clip_XX.jsonl` with flat clip metadata.
4. Logs per-video clip counts to `split_log.csv`.

## Inputs / Outputs

| | Path | Notes |
|---|---|---|
| In | `vlm_dataset/{video_id}.jsonl` | annotations |
| In | `videos/{video_id}.mp4` | source video |
| Out | `clip_dataset/{video_id}/clip_XX.mp4` | step clip |
| Out | `clip_dataset/{video_id}/clip_XX.jsonl` | `clip_id, parent_video_id, clip_filename, parent_video_title, parent_video_url, step_number, clip_duration_seconds, timestamp_start_in_parent, timestamp_end_in_parent, step_title, visual_description, transcript_context, instruments[], anatomy[]` |
| Out | `clip_dataset/split_log.csv` | `video_id, clips_generated, errors, timestamp` |

## Configuration

- `splitter.log_file`
- `directories.clips` (default `./clip_dataset`)

## Notes / Quirks

- Resumable: a clip is skipped when both its `.mp4` and `.jsonl` already
  exist (counted as created).
- Clips are named by step number (`clip_01.mp4` …), so a re-run after
  annotation changes may mix stale clips.
- `clip_id` = `{video_id}_clip_{step_number:02d}`.

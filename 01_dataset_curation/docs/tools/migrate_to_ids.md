# `tools/migrate_to_ids.py` — One-Time Migration Script

Migrates artifacts from the old **title-based naming** (`{video_title}.mp4`)
to **YouTube-ID naming** (`{youtube_id}.mp4`).

## Background

Older downloads saved videos using the YouTube title, which contains
characters that filesystems sanitize differently (`™`, `｜`, `：`, `＂` …).
This made downstream stages fragile. New downloads (via
`stages/downloader.py`) already use YouTube IDs; this script fixes the
existing data.

## What it does

1. **Builds the mapping** — reads `videos/video_metadata.csv` (fallback:
   `dataset_info.csv`), deriving each old filename → YouTube ID from the
   `video_id` column or the URL.
2. **Renames media files** in `videos/`, `audio/`, `transcripts/`,
   `refined_transcripts/` (including `full_responses/`).
3. **Rewrites CSVs** — `video_metadata.csv` (adds `video_id`, updates
   `filename`/`audio_filename`), `refinement_log.csv` (`video_name`,
   `transcript_name`).
4. **Updates VLM data** — renames `vlm_dataset/*.jsonl` and rewrites
   `video_id` / `original_filename` fields.
5. **Updates clip data** — renames `clip_dataset/{folder}/` and rewrites
   `parent_video_id` / `clip_id` in every clip JSONL.
6. **Verifies** — reports rename count and warnings.

## Matching

File names on disk are matched via `normalize_id()` (strip all
non-alphanumeric chars + lowercase), so sanitized names still match:
`CataractCoach™ 2653：…` ↔ `CataractCoach__ 2653：…` ↔ `cataractcoach2653…`.
Unmatched files are listed and left untouched.

## Usage

```bash
# Point at the old project being migrated
python -m video_pipeline.tools.migrate_to_ids --config /path/to/old/config.yaml
python -m video_pipeline.tools.migrate_to_ids --project-root /path/to/old/project
python -m video_pipeline.tools.migrate_to_ids --apply   # perform the migration
```

Options:
- `--dry-run` (default) — preview only, zero filesystem changes.
- `--apply` — perform the migration.
- `--project-root DIR` — root of the old project (default: the active
  config's workspace root).
- `--config <path>` — config whose workspace hosts the old data.

## Safety notes

- Always back up first (the `dataset/` folder is **not** touched — it
  already uses YouTube IDs).
- If a target name already exists, the file is skipped and reported as a
  warning (no overwrite).

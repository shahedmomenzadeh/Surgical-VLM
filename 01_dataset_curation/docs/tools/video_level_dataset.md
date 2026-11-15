# `video_level_dataset.py` — Full-Video Narration Generation

Second dataset track: instead of clip-level MCQs, generates a **full-video**
timestamped narration SFT record per whole surgery video.

## What it generates (per parent folder in `dataset/`)

1. **`full_video_sft.jsonl`** — one record:
   - narration: `video` + "Provide a detailed timestamped narration…" →
     assistant = programmatic `[{MM:SS} - {MM:SS}] step: description` text.

## Key logic

- Clips are read from `clip_*.jsonl` and sorted chronologically by
  `timestamp_start_in_parent`.
- The narration is built deterministically from the core clip metadata
  (`step_title`, `visual_description`, timestamps) — no LLM is used.
- Resume: skips folders where the file already exists and is non-empty.

## Configuration

- Dataset directory defaults to `directories.dataset` in the active config
  (override with `--dataset-dir`).

## Usage

```bash
python -m video_pipeline.tools.video_level_dataset
```

## Notes

- Requires `dataset/` to already contain core clip JSONLs per folder
  (output of `merge_dataset.py` + `rename_to_youtubeid.py`).
- Intended as a complementary track to the clip-level dataset from
  `dataset_formatter` (step 9).

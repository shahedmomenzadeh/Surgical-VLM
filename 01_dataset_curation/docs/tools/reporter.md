# `reporter.py` — Console Dataset Report

Prints summary statistics about the VLM dataset directly to the console
(older, text-only alternative to `generate_report.py`).

## What it reports

- Total video files and total clips.
- Average / max / min clips per video (with video names).
- Average / max / min clip length in seconds (with video names).

## Usage

```bash
python -m video_pipeline.tools.reporter
```

Directory is hardcoded: `VLM_DATASET_DIR = "./vlm_dataset"`. Individual
JSONL files are analyzed (aggregate `*all.jsonl` excluded).

## Notes

- Reads `vlm_annotations[]` from each per-video JSONL; timestamps parsed
  from `MM:SS` / `HH:MM:SS`.

# `video_annotation.py` — Annotated Step Clips (moviepy)

Burns the VLM `visual_description` text into each step clip of selected
videos and exports annotated clips to `annotated_videos/{video_id}/`.

## Behavior

- Reads per-video JSONL files from `VLM_DATASET_DIR` (`./vlm_dataset`),
  skipping the aggregate.
- Only processes videos listed in the hardcoded `VIDEOS_TO_ANNOTATE` list
  (video IDs = filename stems).
- For each annotation step:
  - cuts the sub-clip with `VideoFileClip.subclipped(start, end)`;
  - creates a caption `TextClip` (font 40, white text, black stroke,
    centered at the top) using the `visual_description`;
  - composites and writes `annotated_videos/{video_id}/clip_{N}.mp4`
    (libx264 + AAC audio).

## Usage

```bash
python -m video_pipeline.tools.video_annotation --all
```

## Notes

- `parse_timestamp` accepts `MM:SS` (and `HH:MM:SS`); invalid ranges are
  skipped.
- Requires `moviepy`.

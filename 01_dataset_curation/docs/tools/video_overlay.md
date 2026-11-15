# `video_overlay.py` — Description Overlay on Full Videos (moviepy)

Burns `visual_description` overlays onto a *full-length* video (unlike
`video_annotation.py`, which cuts clips). Output written to `output_videos/`.

## Behavior

- Loads one JSONL entry from a hardcoded path
  (`./vlm_dataset/…{video_title}.jsonl`).
- For each annotation, creates a caption `TextClip` (font 40, white, black
  stroke, centered near the top) and places it with `.with_start(start)`
  and `.with_duration(duration)` on top of the full video.
- Writes `output_videos/labeled_{original_filename}` (libx264 + AAC).

## Usage

```bash
python -m video_pipeline.tools.video_overlay --jsonl <file>
```

## Notes

- Older, experimental helper; `video_annotation.py` is the maintained
  variant.
- Missing source videos are skipped with a warning.

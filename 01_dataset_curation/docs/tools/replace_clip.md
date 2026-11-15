# `replace_clip.py` — Resolve `.ref.txt` Placeholders

On filesystems without symlink support (e.g. Windows without the symlink
privilege), `apply_mapping_all.py` writes `clip_XX.mp4.ref.txt` placeholders
containing the source video path. This script copies the real video files
over the placeholders.

## Behavior

- Recursively finds `*.ref.txt` files under `clip_dataset_mapped/`.
- Reads the source path (second line of the placeholder).
- Copies the source video to the destination name (`clip_XX.mp4.ref.txt` →
  `clip_XX.mp4`) and removes the placeholder.
- Skips when the destination already exists.

## Usage

```bash
python -m video_pipeline.tools.replace_clip
python -m video_pipeline.tools.replace_clip --mapped-dir ./clip_dataset_mapped --dry_run
```

## Options

- `--mapped-dir` (default `./clip_dataset_mapped`)
- `--dry_run` — preview only.

## Notes

- Companion to `apply_mapping_all.py` (which creates the placeholders).

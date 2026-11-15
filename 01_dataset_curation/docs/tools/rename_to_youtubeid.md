# `rename_to_youtubeid.py` — Rename Dataset Folders to YouTube IDs

Renames `dataset/Train|Validation|Test/{old_name}/` folders to
`{youtube_id}/` and rewrites all internal references so everything stays
consistent.

## Flow — Phase 1: rename & rewrite

For each parent folder:

1. Reads the YouTube ID from `parent_video_url` in the first core
   `clip_*.jsonl`.
2. Copies all files into `dataset_new/{split}/{youtube_id}/`:
   - **core clip JSONL** — `parent_video_id` → ID, `clip_id` →
     `{id}_{clip_base}`;
   - **SFT JSONL** — rewrites every `video` reference to `{id}/{filename}`;
   - **GRPO JSONL** — same for the `prompt` content;
   - **MP4s** — copied directly.

## Flow — Phase 2: two-way sync verification

Deep-checks every JSONL line of the output: `parent_video_id` and `clip_id`
must match the folder name, and SFT/GRPO `video` paths must start with
`{folder}/`.

## Usage

```bash
python -m video_pipeline.tools.rename_to_youtubeid
```

Paths are hardcoded: input `./dataset`, output `./dataset_new`.

## Notes

- Input folders whose URL yields no ID are skipped with a warning.
- Only needed for data produced before ID-based naming; new pipeline output
  already uses YouTube IDs.

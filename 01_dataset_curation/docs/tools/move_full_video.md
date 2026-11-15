# `move_full_video.py` — Copy Full Videos into the Dataset

Adds `full_video.mp4` to every parent folder of the final dataset so each
video is available in one place (used by the video-level dataset tooling).

## Flow — Phase 1: matching & copying

1. Catalogs all source videos in `videos/` (by original name + normalized
   name).
2. For every `dataset/{Train,Validation,Test}/{folder}/`:
   - reads `parent_video_title` from the first core `clip_*.jsonl`;
   - locates the source video:
     - Strategy 1: YouTube ID contained in the filename;
     - Strategy 2: normalized title ↔ normalized filename containment;
   - copies it to `{folder}/full_video.mp4` (skips existing non-empty files).

## Flow — Phase 2: verification

Checks every folder with JSONL files has a valid `full_video.mp4`
(existence + size > 1 MB) and reports missing/corrupt entries.

## Usage

```bash
python -m video_pipeline.tools.move_full_video
```

Paths default to the active config: source videos come from
`splitter.videos_dir` (default `videos_down/`, the downsampled 10 fps set)
and the target from `directories.dataset` (overridable with `--videos-dir`,
`--dataset-dir`). This is also pipeline **step 12**
(`python -m video_pipeline --step move_full_video`).

## Notes

- Designed for the dataset layout where folders are named by YouTube ID
  (see `rename_to_youtubeid.py`).
- Verification ignores folders without any `.jsonl` files.

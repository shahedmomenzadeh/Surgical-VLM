# `downsample_videos.py` — Video Downsampling (FPS)

Re-samples every video in `videos/` to a target frame rate, writing to
`videos_down/`. Useful for reducing file size / compute for experiments
(e.g. lower-fps inputs for model inference).

## Behavior

- Filters for `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`.
- Runs `ffmpeg -vf fps=10 -c:a copy` with live console progress
  (`-loglevel info`).
- Output keeps the same filename in `OUTPUT_DIR`.

## Usage

```bash
python -m video_pipeline.tools.downsample_videos
```

Defaults come from the `downsample` config section (`target_fps`,
`input_dir`, `output_dir`); overridable with `--fps`, `--in`, `--out`.
This is also pipeline **step 8** (`python -m video_pipeline --step downsample`).

## Notes

- Audio stream is copied unchanged to preserve sync/duration.
- Verbose mode prints every FFmpeg command and its progress.

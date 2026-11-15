# `remove_sound.py` — Mute Clip Videos

Strips the audio stream from every video in a directory tree, preserving the
folder structure. Used to produce a muted variant of the clip dataset
(`clip_dataset` → `clip_dataset_muted`).

## Behavior

- Recursively scans the input folder for video files (`.mp4`, `.avi`,
  `.mkv`, `.mov`, `.webm`).
- Runs `ffmpeg -c:v copy -an` — video stream is copied (no re-encode),
  audio removed, output overwritten.
- Maintains the same relative subfolder layout in the output directory.

## Usage

```bash
python -m video_pipeline.tools.remove_sound
```

Paths are hardcoded: `input_directory = 'clip_dataset'`,
`output_directory = 'clip_dataset_muted'`.

## Notes

- Copying the video stream keeps quality and speed; container-level
  remux only.
- Corrupted files are reported and skipped.

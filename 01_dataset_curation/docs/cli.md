# `video_pipeline/cli.py` — CLI & Stage Dispatch

Command-line front-end: `python -m video_pipeline` or via an external
entry point (`main.py` in the original project).

## Interface

```
python -m video_pipeline [--step {all,download,clean,transcribe,refine,summarize,vlm,adverse_event,downsample,split,format,merge,move_full_video,narration,refine_instrument_options,generate_missing_grpo_sft}] [--config <path>]
```

- Default step is `all` (runs steps 1–15 in order).
- `--config` overrides the config file (default: the bundled
  `video_pipeline/config.yaml` or `$VIDEO_PIPELINE_CONFIG`).
- `run_step(step, settings)` dispatches to the matching stage with the
  correct configuration from the settings object.

## Responsibilities

- Parsing arguments and printing the banner.
- Mapping every stage's parameters onto the `DatasetSettings` object built
  by `config.build_settings()` (e.g. refiner delay/limit come from
  `refiner.extra`, VLM gatekeeper settings from `vlm.gatekeeper`, etc.).
- Step numbering in the log output (`[Step X/15]`).
- `main(argv=None, config=None)` — the `config` keyword is prepended as a
  `--config` argument, so external wrappers can pin their own config file
  (the original project's `main.py` does exactly this).

## The 15 steps

| # | Step | Runs | Notes |
|---|------|------|-------|
| 1 | `download` | `stages/downloader` | yt-dlp + audio + metadata CSV |
| 2 | `clean` | `stages/cleaner` | drop too-long videos |
| 3 | `transcribe` | `stages/transcriber` | Whisper ASR |
| 4 | `refine` | `stages/refiner` | LLM transcript editing |
| 5 | `summarize` | `stages/summarizer` | `dataset_info.csv` |
| 6 | `vlm` | `stages/vlm_generator` | Gemini video analysis |
| 7 | `adverse_event` | `stages/adverse_event_detector` | safety analysis |
| 8 | `downsample` | `tools/downsample_videos` | 10 fps → `videos_down/` |
| 9 | `split` | `stages/splitter` | clips cut from `videos_down/` |
| 10 | `format` | `stages/dataset_formatter` | clip-level SFT/GRPO + split file |
| 11 | `merge` | `tools/merge_dataset` | assemble `dataset/{Train,Validation,Test}/` |
| 12 | `move_full_video` | `tools/move_full_video` | copy `full_video.mp4` per video |
| 13 | `narration` | `tools/video_level_dataset` | `full_video_sft.jsonl` narration |
| 14 | `refine_instrument_options` | `tools/refine_instrument_options` | LLM fix of instrument distractors |
| 15 | `generate_missing_grpo_sft` | `tools/generate_missing_grpo_sft` | regenerate missing per-clip SFT/GRPO |

## Notes

- Stages are imported eagerly at module load; heavy optional dependencies
  (whisper, torch) are imported lazily inside their stages, so the CLI runs
  without them installed.
- Each stage handles missing keys / files gracefully (prints an error and
  returns) rather than crashing the whole run.
- LLM-backed steps (`refine_instrument_options`, `generate_missing_grpo_sft`)
  are resumable and skip already-processed clips.

# Video Pipeline — Documentation

Complete documentation for the **Video Processing Pipeline**: a modular
pipeline that builds Vision-Language Model (VLM) training datasets
(SFT & GRPO) for cataract surgery from YouTube videos.

## Architecture Overview

```
youtube_links.txt
      │  Step 1-2
      ▼
videos/{youtube_id}.mp4 ──► audio/{youtube_id}.wav ──► transcripts/{youtube_id}.json
                                                              │  Step 4 (LLM refine)
                                                              ▼
                                                   refined_transcripts/{youtube_id}.txt
                                                              │  Step 5
                                                              ▼
                                                   dataset_info.csv
                                                              │  Step 6 (Gemini analyst)
                                                              ▼
vlm_dataset/{youtube_id}.jsonl ──► adverse_events/{youtube_id}.jsonl   (Step 7)
                                                              │  Step 8 (downsample → 10 fps)
                                                              ▼
                                                   videos_down/{youtube_id}.mp4
                                                              │  Step 9 (FFmpeg)
                                                              ▼
                                                   clip_dataset/{youtube_id}/clip_XX.{mp4,jsonl}
                                                              │  Step 10 (LLM QA)
                                                              ▼
                                   training_datasets/{train,val,test}_{sft,grpo}.jsonl
                                                              │  Step 11 (merge)
                                                              ▼
                                   dataset/{Train,Validation,Test}/{youtube_id}/…
                                              │  Step 12 (full_video.mp4)
                                              │  Step 13 (narration SFT)
                                              │  Step 14 (instrument distractors)
                                              ▼  Step 15 (missing SFT/GRPO)
                                   final dataset (as shipped)
```

### The 15 pipeline steps

| # | Step | Doc | What it does |
|---|------|-----|--------------|
| 1 | `download` | [stages/downloader.md](stages/downloader.md) | yt-dlp download as `{youtube_id}.mp4`, FFmpeg audio extraction, metadata CSV |
| 2 | `clean` | [stages/cleaner.md](stages/cleaner.md) | delete videos longer than `max_duration_seconds` |
| 3 | `transcribe` | [stages/transcriber.md](stages/transcriber.md) | Whisper ASR → timestamped JSON segments |
| 4 | `refine` | [stages/refiner.md](stages/refiner.md) | LLM medical editing of transcripts |
| 5 | `summarize` | [stages/summarizer.md](stages/summarizer.md) | merge logs → `dataset_info.csv` |
| 6 | `vlm` | [stages/vlm_generator.md](stages/vlm_generator.md) | gatekeeper filter + Gemini video analysis |
| 7 | `adverse_event` | [stages/adverse_event_detector.md](stages/adverse_event_detector.md) | LLM safety analysis of annotations |
| 8 | `downsample` | [tools/downsample_videos.md](tools/downsample_videos.md) | re-sample videos to `downsample.target_fps` → `videos_down/` |
| 9 | `split` | [stages/splitter.md](stages/splitter.md) | FFmpeg clip extraction from the downsampled videos |
| 10 | `format` | [stages/dataset_formatter.md](stages/dataset_formatter.md) | LLM MCQ generation → SFT/GRPO files |
| 11 | `merge` | [tools/merge_dataset.md](tools/merge_dataset.md) | assemble `dataset/{Train,Validation,Test}/` |
| 12 | `move_full_video` | [tools/move_full_video.md](tools/move_full_video.md) | copy `full_video.mp4` into each dataset folder |
| 13 | `narration` | [tools/video_level_dataset.md](tools/video_level_dataset.md) | full-video narration SFT (`full_video_sft.jsonl`) |
| 14 | `refine_instrument_options` | [tools/dataset_helpers.md](tools/dataset_helpers.md) | LLM judge fixing too-similar instrument distractors |
| 15 | `generate_missing_grpo_sft` | [tools/dataset_helpers.md](tools/dataset_helpers.md) | regenerate missing per-clip SFT/GRPO files |

### Supporting code

| File | Doc | Role |
|------|-----|------|
| `video_pipeline/cli.py` | [cli.md](cli.md) | `--step` argument parsing and stage dispatch |
| `video_pipeline/config.py` | [config.md](config.md) | config.yaml + .env loading, per-stage settings |
| `video_pipeline/prompts.py` | [prompts.md](prompts.md) | all LLM prompts (single source of truth) |
| `video_pipeline/common/utils.py` | [common/utils.md](common/utils.md) | timestamps, YouTube IDs, name normalization, FFmpeg check |
| `video_pipeline/common/io.py` | [common/io.md](common/io.md) | JSONL helpers + CSV progress-log class (resumability) |
| `video_pipeline/common/llm.py` | [common/llm.md](common/llm.md) | OpenAI-compatible clients + JSON retry helper |
| `video_pipeline/common/gemini.py` | [common/gemini.md](common/gemini.md) | Google GenAI client (analyst stage only) |
| `main.py` (original project) | [main.md](main.md) | external entry point pinning the old project's config |

## Utility / One-off Tools (`video_pipeline/tools/`)

Run any tool with `python -m video_pipeline.tools.<name> [options]`.
All tools accept `--config <path>` (default: bundled config /
`$VIDEO_PIPELINE_CONFIG`).

| Tool | Doc | Purpose |
|------|-----|---------|
| `apply_mapping_all` | [tools/apply_mapping_all.md](tools/apply_mapping_all.md) | canonicalize step/instrument labels via mapping JSONs |
| `merge_dataset` | [tools/merge_dataset.md](tools/merge_dataset.md) | distribute clips + SFT/GRPO into `dataset/Train/Validation/Test/` (pipeline step 11) |
| `move_full_video` | [tools/move_full_video.md](tools/move_full_video.md) | copy `full_video.mp4` into dataset folders (pipeline step 12) |
| `rename_to_youtubeid` | [tools/rename_to_youtubeid.md](tools/rename_to_youtubeid.md) | rename dataset folders to YouTube IDs + sync verify |
| `video_level_dataset` | [tools/video_level_dataset.md](tools/video_level_dataset.md) | full-video narration SFT (pipeline step 13) |
| `generate_report` | [tools/generate_report.md](tools/generate_report.md) | HTML analytics report (Chart.js) |
| `remove_sound` | [tools/remove_sound.md](tools/remove_sound.md) | strip audio from clip videos |
| `downsample_videos` | [tools/downsample_videos.md](tools/downsample_videos.md) | re-sample videos to a target FPS (pipeline step 8) |
| `video_annotation` | [tools/video_annotation.md](tools/video_annotation.md) | burn visual descriptions into step clips (moviepy) |
| `video_overlay` | [tools/video_overlay.md](tools/video_overlay.md) | burn descriptions as overlay on full videos |
| `clean_log` | [tools/clean_log.md](tools/clean_log.md) | remove `ERROR_GENERATION` rows from process log |
| `replace_clip` | [tools/replace_clip.md](tools/replace_clip.md) | resolve `.ref.txt` symlink placeholders |
| `reporter` | [tools/reporter.md](tools/reporter.md) | console statistics of the VLM dataset |
| `apitest` | [tools/apitest.md](tools/apitest.md) | Gemini API connectivity test |
| `migrate_to_ids` | [tools/migrate_to_ids.md](tools/migrate_to_ids.md) | one-time migration: title names → YouTube IDs |
| `generate_missing_grpo_sft` | [tools/dataset_helpers.md](tools/dataset_helpers.md) | regenerate missing per-clip SFT/GRPO files (pipeline step 15) |
| `refine_instrument_options` | [tools/dataset_helpers.md](tools/dataset_helpers.md) | LLM judge that fixes too-similar instrument distractors (pipeline step 14) |
| `phase_sft_dataset` | [tools/phase_sft_dataset.md](tools/phase_sft_dataset.md) | teacher/critic grounded SFT generation from phase-recognition clips |
| `phase_grpo_dataset` | [tools/phase_grpo_dataset.md](tools/phase_grpo_dataset.md) | deterministic phase-ontology GRPO generation from accepted phase SFT clips |
| `merge_phase_dataset` | [tools/merge_phase_dataset.md](tools/merge_phase_dataset.md) | add accepted phase SFT/GRPO samples to the final dataset |

## Design Conventions

1. **YouTube-ID naming** — every artifact is keyed by `{youtube_id}` from
   download onward; no fragile title-based filename matching.
2. **OpenAI-compatible LLMs** — all text-only LLM stages use
   `base_url + model + api_key`. Only the VLM analyst (YouTube URL video
   analysis) uses the Google GenAI SDK.
3. **Secrets in `.env`** — one API key placeholder per pipeline section
   (see [.env.example](../.env.example)). Never hardcode keys.
4. **Resumability** — every stage logs progress to a CSV (via
   `common/io.py::ProgressLog`) and skips already-processed items.
5. **Data-leakage-free splits** — train/val/test split is done at the
   *parent video* level (80/10/10, seed 42).
6. **Output stability** — SFT/GRPO JSONL schemas, split file, clip metadata
   and log formats are frozen; new code must not change them.

## Configuration & Secrets

- `config.yaml` — paths, base URLs, model names (see [config.md](config.md)).
- `.env` — API keys: `REFINER_API_KEY`, `VLM_GATEKEEPER_API_KEY`,
  `GEMINI_API_KEY`, `ADVERSE_EVENT_API_KEY`, `FORMATTER_API_KEY`,
  `PHASE_SFT_TEACHER_API_KEY`, `PHASE_SFT_CRITIC_API_KEY`.

## Running the Pipeline

```bash
python main.py                 # all 15 steps
python main.py --step download # single step
```

See the [package README](../README.md) for setup and usage details.

## Phase-Dataset SFT Gate

The phase-recognition SFT generator runs independently of the 15-step
YouTube pipeline:

```bash
python -m video_pipeline.tools.phase_sft_dataset --config video_pipeline/config.yaml
```

It reads `phase_dataset/manifest.csv`, sends each clip as a base64 video to
the configured OpenAI-compatible teacher and critic, and writes only
critic-approved records to `phase_vlm_dataset/`. A full video request is
attempted first. Upstream or payload failures fall back to at most 32 evenly
spaced base64 JPEG frames, capped by the actual decoded frame count. Candidate
Idle clips containing visible surgery are rejected and do not produce SFT
files.

The stage is resumable through `phase_sft_log.csv` and per-clip review
checkpoints under `phase_vlm_dataset/.work/`. Use `--dry-run`, `--split`, and
`--limit` to validate or test a subset before making API calls.

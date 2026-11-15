# `video_pipeline/stages/transcriber.py` — Transcriber (Step 3)

## Purpose

Runs OpenAI Whisper speech-to-text on every audio file and stores
timestamped segment JSON: `transcripts/{video_id}.json`.

## Flow

1. Resolves the device (`cuda` if available, else `cpu`) and loads the
   Whisper model.
2. Compares the set of `.wav` files in `audio/` with existing `.json` files
   in `transcripts/`; only unprocessed files are transcribed (resumable).
3. For each file, `model.transcribe()` is called and the segments are saved
   as a JSON array of `{"start", "end", "text"}` (rounded to 0.1 s).

## Inputs / Outputs

| | Path | Notes |
|---|---|---|
| In | `audio/{video_id}.wav` | 16 kHz mono |
| Out | `transcripts/{video_id}.json` | `[{start, end, text}, ...]` |

## Configuration

- `whisper.model_size` (default `large`), `whisper.device` (default `cuda`)

## Dependencies

`whisper` (installed from GitHub source, see `requirements.txt`), `torch`.
These are **imported lazily** inside the stage so the rest of the pipeline
runs without them.

## Notes / Quirks

- Transcription uses `fp16=torch.cuda.is_available()`.
- A `.json` file existing is enough for the stage to skip the audio (there
  is no validity check on the JSON content).

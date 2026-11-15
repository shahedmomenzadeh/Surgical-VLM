# `video_pipeline/stages/vlm_generator.py` — VLM Generator (Step 6)

## Purpose

Creates the per-video surgical annotations used by every downstream stage.
Two-stage design:

1. **Gatekeeper** — a fast OpenAI-compatible text model decides whether the
   transcript is worth processing (`YES`/`NO` + confidence + reasoning).
2. **Analyst** — Google Gemini (genai SDK) *watches the video* via its
   YouTube URL and produces step-by-step surgical annotations.

## Flow

1. Loads `dataset_info.csv`; resumes by skipping videos whose
   `{video_id}.jsonl` exists on disk or whose log status is `ACCEPTED` /
   `REJECTED`.
2. Locates the refined transcript (several naming conventions tried).
3. **Gatekeeper call** (transcript truncated to 25 000 chars; JSON mode).
   A `NO` decision logs the video as `REJECTED`.
4. **Analyst call** — the YouTube URL is passed to Gemini as a `file_data`
   part along with the analyst prompt (JSON mode). A 60 s sleep throttles
   rate limits.
5. On success: writes `vlm_dataset/{video_id}.jsonl` (individual), appends
   to `vlm_dataset_all.jsonl` (aggregate), logs `ACCEPTED`.

## Inputs / Outputs

| | Path | Notes |
|---|---|---|
| In | `dataset_info.csv` | summary from step 5 |
| In | `refined_transcripts/*.txt` | transcript text |
| Out | `vlm_dataset/{video_id}.jsonl` | one record: `video_id, original_filename, status, video_url, video_title, download_date, transcript_quality_check, vlm_annotations[]` |
| Out | `vlm_dataset/vlm_dataset_all.jsonl` | aggregate of all accepted records |
| Out | `vlm_dataset/process_log.csv` | `video_id, original_filename, status, decision, confidence, reasoning, video_title, url, download_date, timestamp` |

Each `vlm_annotations[]` step: `step_number, timestamp_start, timestamp_end,
step_title, visual_description, transcript_context, instruments[], anatomy[]`.

## Configuration / Keys

- `vlm.gatekeeper` — `base_url`, `model`, `api_key_env`
  (default `VLM_GATEKEEPER_API_KEY`)
- `vlm.analyst` — `model` (default `gemini-2.5-pro`), `api_key_env`
  (default `GEMINI_API_KEY`)
- `vlm.aggregate_file`, `vlm.log_file`

## Notes / Quirks

- The analyst **must** use the google-genai SDK: sending a YouTube URL as
  `file_data` is Gemini-specific and unavailable through OpenAI-compatible
  endpoints.
- Only `ACCEPTED`/`REJECTED` log entries count as done; `ERROR_GENERATION`
  entries are retried on the next run.

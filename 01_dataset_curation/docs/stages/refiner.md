# `video_pipeline/stages/refiner.py` — Refiner (Step 4)

## Purpose

Cleans raw Whisper transcripts with an LLM (medical editor prompt): fixes
typos, punctuation and medical terminology while preserving the exact JSON
structure and timestamps. Uses an **OpenAI-compatible** endpoint (Cerebras
by default).

## Flow

1. Discovers `.json` transcripts that have no corresponding `.txt` output.
2. For each transcript:
   - sends the JSON payload to the LLM with
     `MEDICAL_EDITOR_SYSTEM_PROMPT`;
   - retries indefinitely on transient API errors (10 s backoff); context
     overflows return a marker payload instead;
   - saves the raw LLM response to
     `refined_transcripts/full_responses/{id}_full_response.txt`;
   - `parse_llm_json_output()` strips `<think>` blocks and extracts the
     JSON array;
   - `format_segments_to_txt()` renders `[MM:SS - MM:SS]: text` lines into
     `refined_transcripts/{id}.txt`;
   - appends a row to `refinement_log.csv`.
3. `max_files_per_run` limits how many files are processed per invocation;
   `api_call_delay_seconds` throttles the API.

## Inputs / Outputs

| | Path |
|---|---|
| In | `transcripts/{video_id}.json` |
| Out | `refined_transcripts/{video_id}.txt` |
| Out | `refined_transcripts/full_responses/{video_id}_full_response.txt` |
| Out | `refined_transcripts/refinement_log.csv` |

Log columns: `video_name, transcript_name, total_characters, total_words,
video_found, success`.

## Configuration / Keys

- `refiner.base_url` (default `https://api.cerebras.ai/v1`)
- `refiner.model`, `refiner.api_call_delay_seconds`, `refiner.max_files_per_run`
- API key: `REFINER_API_KEY`

## Notes / Quirks

- `find_matching_video()` checks for `{video_id}.mp4` in `videos/` — with
  YouTube-ID naming this is a simple existence check.
- `video_name` in the log is the matched video filename (or `NOT_FOUND`);
  the summarizer joins on this column.

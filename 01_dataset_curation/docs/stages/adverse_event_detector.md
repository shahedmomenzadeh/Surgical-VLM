# `video_pipeline/stages/adverse_event_detector.py` — Adverse Event Detector (Step 7)

## Purpose

Analyzes the VLM step annotations for intraoperative complications
(e.g. posterior capsule rupture, iris prolapse, zonular dialysis, vitreous
loss, nucleus/IOL drop). Text-only analysis, therefore OpenAI-compatible.

## Flow

1. Reads every `vlm_dataset/{video_id}.jsonl` (skipping the aggregate file),
   resuming via the log (status `DETECTED`/`NO_EVENT` are done).
2. `extract_visual_steps()` formats the annotations into a chronological
   "Surgical Steps Timeline" text block.
3. The LLM is asked (with `SAFETY_ANALYST_PROMPT`, JSON mode) to return a
   list of `adverse_events`; 30 s sleep between calls for rate limits.
4. If events were found: writes `adverse_events/{video_id}.jsonl` + appends
   to the aggregate, logs `DETECTED`. Otherwise logs `NO_EVENT`.

## Inputs / Outputs

| | Path | Notes |
|---|---|---|
| In | `vlm_dataset/{video_id}.jsonl` | annotations from step 6 |
| Out | `adverse_events/{video_id}.jsonl` | only when events detected |
| Out | `adverse_events/adverse_events_all.jsonl` | aggregate |
| Out | `adverse_events/adverse_event_log.csv` | `video_id, status, event_count, timestamp` |

Detected event object: `event_name, timestamp_start, timestamp_end, reason`.

## Configuration / Keys

- `adverse_event.base_url`, `adverse_event.model`, `adverse_event.api_key_env`
  (default `ADVERSE_EVENT_API_KEY`)
- `adverse_event.aggregate_file`, `adverse_event.log_file`

## Notes / Quirks

- The prompt instructs the model to *infer* complications from management
  actions (e.g. "anterior vitrectomy" implies PCR/vitreous loss).
- Videos with no visual steps are skipped without logging.

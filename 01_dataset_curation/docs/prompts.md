# `video_pipeline/prompts.py` — LLM Prompts

Single source of truth for every LLM prompt in the pipeline. Content is
preserved **verbatim** from the original modules so behavior stays
identical.

## Prompts

| Constant / function | Used by | Role |
|---------------------|---------|------|
| `MEDICAL_EDITOR_SYSTEM_PROMPT` | refiner (step 4) | correct typos/grammar/medical terms in transcript JSON, preserving structure |
| `GATEKEEPER_PROMPT` | vlm_generator (step 6) | decide `YES`/`NO` whether a transcript belongs in a surgical dataset (contains `{transcript_text}` placeholder) |
| `ANALYST_PROMPT` | vlm_generator (step 6) | segment the video into surgical steps with instruments/anatomy (contains `{transcript_text}`) |
| `SAFETY_ANALYST_PROMPT` | adverse_event_detector (step 7) | detect intraoperative complications from the step timeline |
| `QA_SYSTEM_PROMPT` | dataset_formatter (step 9) | generate 3 MCQ QA pairs per clip |
| `build_qa_user_prompt(clip_data)` | dataset_formatter (step 9) | builds the user message from clip metadata (step title, description, instruments, anatomy) |

## Conventions

- Prompts request strict JSON output; stages enforce parsing via
  `response_format={"type": "json_object"}` (OpenAI-compatible) or
  `response_mime_type="application/json"` (GenAI) plus Pydantic validation
  where schemas matter (dataset formatter).
- `ANALYST_PROMPT` instructs the model to prioritize *visual* information
  over the transcript and to use `MM:SS` / `HH:MM:SS` timestamps.

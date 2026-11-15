# `video_pipeline/stages/dataset_formatter.py` — Dataset Formatter (Step 10)

## Purpose

Final stage: converts step clips into training-ready **SFT** and **GRPO**
JSONL files. For each clip an OpenAI-compatible LLM generates exactly 3
multiple-choice QA pairs (step identification, visual observation, instrument
identification), validated with Pydantic before being written.

**Output schemas are frozen** — they must stay byte-compatible with existing
datasets.

## Flow

1. `manage_train_val_test_split()` creates or loads the parent-video-level
   split (`dataset_split.json`): 80/10/10 with `seed(42)` so reruns and
   recomputations are reproducible.
2. Resumes via the formatter log (`SUCCESS` clip ids).
3. For each `clip_dataset/{parent_id}/clip_XX.jsonl`:
   - builds the QA user prompt from clip metadata;
   - calls the LLM with `QA_SYSTEM_PROMPT` in JSON mode, validates with
     `QAList`/`QAPair` Pydantic models;
   - writes to `{split}_sft.jsonl` and `{split}_grpo.jsonl` (append mode).

## SFT output — 4 records per clip

1. Description task: `messages` with `video` + text
   "Describe what is happening…" → assistant = `visual_description`.
2–4. QA pairs: `video` + MCQ text → assistant =
   `{reference_reasoning} Therefore the answer is {letter}) {option}.`

## GRPO output — 3 records per clip

```json
{
  "prompt": [{"role": "user", "content": [
      {"type": "video", "video": "{parent_id}/clip_XX.mp4"},
      {"type": "text", "text": "..."}]}],
  "correct_answer": "A",
  "question_type": "step_identification | visual_observation | instrument_identification",
  "reference_reasoning": "...",
  "reward_type": "deterministic | llm_judge"
}
```

`reward_type` is `deterministic` for step/instrument identification and
`llm_judge` for visual observation.

## Inputs / Outputs

| | Path |
|---|---|
| In | `clip_dataset/{parent_id}/clip_XX.jsonl` |
| Out | `training_datasets/{train,val,test}_{sft,grpo}.jsonl` |
| Out | `training_datasets/dataset_split.json` |
| Out | `training_datasets/dataset_formatter_log.csv` (`clip_id, status, timestamp`) |

## Configuration / Keys

- `dataset_formatter.base_url`, `model`, `max_retries`, `output_dir`,
  `log_file`, `split_file`
- API key: `FORMATTER_API_KEY`

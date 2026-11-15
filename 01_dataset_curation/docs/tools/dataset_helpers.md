# `dataset/` Helper Scripts

Two scripts that operate on the final `dataset/` layout
(`dataset/{Train,Validation,Test}/{video_id}/…`) to fix or complete
per-clip SFT/GRPO files after generation. Both are integrated into the
main pipeline as **step 14** (`refine_instrument_options`) and
**step 15** (`generate_missing_grpo_sft`), and use the
`refine_instrument_options` / `generate_missing_grpo_sft` config sections
(endpoint, model, API key, retries) when run standalone.

---

## `generate_missing_grpo_sft.py` — Regenerate Missing SFT/GRPO

Scans all `clip_*.mp4` files across the splits and regenerates
`clip_XX_sft.jsonl` / `clip_XX_grpo.jsonl` for any clip missing them.

- Reuses the same QA-generation prompt/Pydantic schemas as
  `stages/dataset_formatter.py` (3 MCQs per clip).
- Only writes the files that are missing (union of clips needing either).
- `--dry-run` lists what's missing without generating.

```bash
python -m video_pipeline.tools.generate_missing_grpo_sft ./dataset
python -m video_pipeline.tools.generate_missing_grpo_sft ./dataset --dry-run
```

Defaults: OpenAI-compatible endpoint `https://api.gapgpt.app/v1`, model
`gemini-3-flash-preview`, API key from `FORMATTER_API_KEY` env var
(overridable with `--base-url`, `--model`, `--api-key`, `--retries`).

---

## `refine_instrument_options.py` — Fix Similar Instrument Distractors

An LLM **judge** that inspects the 4 options of every
`instrument_identification` question in the SFT/GRPO files and replaces
distractors that are too visually/functionally similar to the correct
answer (e.g. two hook-like instruments).

- Reads SFT line 4 (the instrument question) via regex parsing; writes back
  the SFT and GRPO entries when options are replaced.
- Resume via `dataset/processed_clips.txt` (appended per processed clip).
- `--dry-run` reports what would change without writing.

```bash
python -m video_pipeline.tools.refine_instrument_options ./dataset
python -m video_pipeline.tools.refine_instrument_options ./dataset --dry-run
```

Defaults: endpoint `https://api.gapgpt.app/v1`, model `deepseek-v4-flash`,
API key from `FORMATTER_API_KEY`.

---

Both scripts expect `clip_XX_sft.jsonl` to contain exactly 4 lines
(1 description + 3 QA) and `clip_XX_grpo.jsonl` exactly 3 lines.

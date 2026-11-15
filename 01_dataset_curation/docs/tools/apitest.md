# `apitest.py` — Gemini API Connectivity Test

Minimal script that verifies the Google GenAI API key works by asking the
model to echo a phrase using streaming output.

## Behavior

- Reads `GEMINI_API_KEY` from the environment / project `.env` (raises if
  missing).
- Calls `gemini-2.5-flash` with `generate_content_stream` and prints the
  response chunks.
- Uses `thinking_config` with an unlimited thinking budget.

## Usage

```bash
python -m video_pipeline.tools.apitest
```

## Notes

- Historically contained a hardcoded key; now reads from `.env` only.

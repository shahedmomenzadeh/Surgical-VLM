# `video_pipeline/common/llm.py` — OpenAI-Compatible LLM Client

Single place for all **text-only** LLM calls (refiner, VLM gatekeeper,
adverse-event detector, dataset formatter).

## Functions

| Function | Purpose |
|----------|---------|
| `_resolve_api_key(env_var, colab_secret=None)` | key resolution: Google Colab secrets → environment → project `.env` |
| `get_openai_client(api_key, base_url=None)` | raw client factory |
| `resolve_and_get_client(env_var, base_url=None)` | resolve key from env and build client; prints an error and returns `None` if missing |
| `chat_json_with_retry(client, model_name, messages, max_retries, temperature, max_tokens, retry_delay, response_format, on_retry)` | chat completion with retry; parses JSON when `response_format` is set; `None` on total failure |

## Design decisions

- Every stage configures `base_url` + `model` + `api_key_env` in
  `config.yaml`; the key itself lives in `.env`.
- `.env` is loaded from the **project root** (not the CWD) so the pipeline
  works when invoked from any directory.
- Retry policy is centralized (backoff + optional `on_retry` callback for
  per-attempt logging).

## Notes

- The VLM analyst does **not** use this module — it needs the Google GenAI
  SDK to send YouTube URLs (see `common/gemini.md`).

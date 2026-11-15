# `video_pipeline/common/gemini.py` — Google GenAI Client

Factory for the `google-genai` SDK client.

## Why only here

Only the **VLM analyst** stage uses this client. It sends the *YouTube URL*
of a video as a `file_data` part so Gemini can watch the video directly —
a feature unavailable through OpenAI-compatible endpoints. All other LLM
stages use `common/llm.py`.

## Functions

| Function | Purpose |
|----------|---------|
| `get_gemini_client(env_var="GEMINI_API_KEY")` | build `genai.Client`; key from Colab secrets → environment → project `.env`; `None` (with error message) if missing |

## Notes

- Colab support is preserved: when running inside Google Colab with the
  secret named after `env_var`, that key wins.
- `.env` is loaded from the project root via `config.load_env()`.

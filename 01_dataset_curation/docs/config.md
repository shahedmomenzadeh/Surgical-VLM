# `video_pipeline/config.py` — Configuration Loading

Loads `config.yaml` (non-secret settings) and `.env` files (API keys), and
exposes typed per-stage settings.

## Config resolution order

1. `--config <path>` CLI argument (see [cli.md](cli.md))
2. `VIDEO_PIPELINE_CONFIG` environment variable
3. `video_pipeline/config.yaml` — the **bundled default**

The package is fully self-contained: it never reads configs from any parent
project. The active config file is recorded (`set_active_config`) so that
`.env` loading and relative-path resolution always target the correct
workspace, no matter which module calls them.

## Key functions

| Function | Purpose |
|----------|---------|
| `load_config(config_path=None)` | parse YAML; exits with an error if missing |
| `find_config(config_path=None)` | resolve the config path (override → env → bundled) |
| `set_active_config(config_path=None)` | record the active config file |
| `get_active_config()` | the recorded config path (falls back to bundled) |
| `get_workspace_root(config_path=None)` | folder containing the active config |
| `load_env()` | load workspace `.env` then package `.env` (workspace wins) |
| `resolve_path(raw)` | make config paths absolute relative to the workspace root |
| `load_video_links(file_path)` | read URLs file (skips blanks and `#` comments) |
| `build_settings(config=None, config_path=None)` | build the `DatasetSettings` dataclass |

## Settings dataclasses

- `DownloadSettings` — `max_duration_seconds`
- `WhisperSettings` — `model_size`, `device`
- `DownsampleSettings` — `target_fps`, `input_dir`, `output_dir` (resolved)
- `LLMSettings` — `base_url`, `model`, `api_key_env`, `extra` (remaining
  section keys, e.g. delays, log/aggregate filenames)
- `VlmSettings` — `gatekeeper` (LLMSettings), `analyst_model`,
  `analyst_api_key_env`, `aggregate_file`, `log_file`
- `DatasetSettings` — fully-resolved absolute paths for every directory and
  log file, plus the above per-stage settings and `config_path`.

## Config sections for the added pipeline steps (8, 11–15)

| Section | Fields | Used by |
|---------|--------|---------|
| `directories.videos_down` | path to the 10 fps videos | `downsample`, `split`, `move_full_video` |
| `directories.dataset` | final dataset root | `merge`, `move_full_video`, `narration`, steps 14–15 |
| `downsample` | `target_fps`, `input_dir`, `output_dir` | step 8 (`tools/downsample_videos`) |
| `splitter.videos_dir` | source videos for clipping (default: `videos_down`) | step 9 |
| `refine_instrument_options` | `base_url`, `model` (`deepseek-v4-flash`), `api_key_env` (`FORMATTER_API_KEY`), `max_retries` | step 14 |
| `generate_missing_grpo_sft` | `base_url`, `model` (`gemini-3-flash-preview`), `api_key_env` (`FORMATTER_API_KEY`), `max_retries` | step 15 |

`refine_instrument_options` and `generate_missing_grpo_sft` fall back to
their bundled defaults when their sections are absent.

## Conventions

- **Paths** are resolved relative to the folder containing the *active*
  config file, so the pipeline works from any working directory.
- **Secrets never live in config.yaml** — each section names its env var
  via `api_key_env`; the value is read from `.env` at runtime.
- Missing sections fall back to sensible defaults rather than crashing.

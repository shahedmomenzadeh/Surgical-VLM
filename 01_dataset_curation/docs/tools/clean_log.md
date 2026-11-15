# `clean_log.py` — Clean Process Log

One-off maintenance script: removes `ERROR_GENERATION` rows from
`vlm_dataset/process_log.csv`.

## Why

The VLM generator logs failed generations as `ERROR_GENERATION` and the
resume logic already retries those, so keeping them in the log is harmless.
This script was used to purge them when a clean history was wanted.

## Usage

```bash
python -m video_pipeline.tools.clean_log
```

Path is hardcoded: `vlm_dataset/process_log.csv`. The file is overwritten
with the filtered rows.

## Notes

- Reports original/kept/removed counts; no-op if no error rows exist.

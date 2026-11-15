# `video_pipeline/common/io.py` — JSONL / CSV I/O

Centralized file I/O for the pipeline: JSONL reading/writing plus the
append-only CSV progress log used by every stage for resumability.

## JSONL helpers

| Function | Purpose |
|----------|---------|
| `load_jsonl(path)` | all records, skipping empty/malformed lines |
| `load_single_jsonl_record(path)` | first record (per-video / per-clip files) |
| `write_jsonl(path, records, append=False)` | bulk write |
| `write_single_jsonl_record(path, record)` | overwrite single record |
| `append_jsonl_record(path, record)` | append one record |

All writes are UTF-8 and `ensure_ascii=False`.

## `ProgressLog` class

Append-only CSV log:

- `ProgressLog(path, headers)` — creates the file with headers on first use.
- `.append(**fields)` — appends a row (missing fields become `""`).
- `.read_processed(statuses)` — returns the set of ids (first column) whose
  **last** logged status is in `statuses`.

Example usage (adverse event detector):

```python
log = ProgressLog("adverse_event_log.csv", ["video_id", "status", "event_count", "timestamp"])
log.append(video_id="abc", status="DETECTED", event_count=2, timestamp=now_str())
done = log.read_processed(["DETECTED", "NO_EVENT"])
```

This replaces five near-identical CSV helper implementations that previously
existed in refiner / vlm_generator / adverse_event_detector / splitter /
dataset_formatter.

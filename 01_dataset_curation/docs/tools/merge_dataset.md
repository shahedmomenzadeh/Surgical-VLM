# `merge_dataset.py` — Merge Clips & Training Data into `dataset/`

Builds the final self-contained dataset layout:

```
dataset/
├── Train/Validation/Test/
│   └── {video_id}/
│       ├── clip_01.mp4 … clip_XX.mp4     (copied from clip_dataset)
│       ├── clip_01.jsonl …               (core clip metadata)
│       ├── clip_01_sft.jsonl …           (SFT records)
│       └── clip_01_grpo.jsonl …          (GRPO records)
```

## Flow

1. Loads and **inverts** `training_datasets/dataset_split.json`
   (`train/val/test` → `Train/Validation/Test`).
2. Copies every clip folder from `clip_dataset/` into its split folder.
   Folder names are matched with `normalize_id()` (case-insensitive,
   non-alphanumeric-stripped), so sanitized names still route correctly.
   Videos missing from the split map default to `Train`.
3. Distributes the SFT/GRPO lines from
   `training_datasets/{split}_{sft|grpo}.jsonl` into per-clip files
   (`clip_XX_sft.jsonl`, `clip_XX_grpo.jsonl`), resolving the `video`
   reference to the actual split/folder and **verifying the source clip
   mp4 exists** before writing.
4. Reports per-file count verification (lines in source vs. writes).

## Usage

```bash
python -m video_pipeline.tools.merge_dataset
```

Paths default to the active config: `clips` → `clip_dataset/`,
`dataset_formatter.output_dir` → `training_datasets/`,
`directories.dataset` → `dataset/` (overridable with `--clips-dir`,
`--training-dir`, `--out`). This is also pipeline **step 11**
(`python -m video_pipeline --step merge`).

## Notes

- Uses normalized ID matching for both splits and physical folder names.
- Safe to re-run: files are only copied when missing; JSONL writes append.

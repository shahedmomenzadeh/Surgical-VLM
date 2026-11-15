# `generate_report.py` — HTML Dataset Analytics Report

Generates a self-contained `dataset_report.html` (Chart.js) summarizing the
final dataset: clips, SFT/GRPO records, splits, steps, instruments, anatomy
and text-length distributions.

## Inputs

| Flag | Default | What it reads |
|------|---------|---------------|
| `--clips_dir` | `./clip_dataset` | per-clip JSONL metadata |
| `--dataset_dir` | `./training_datasets` | `{train,val,test}_{sft,grpo}.jsonl` |
| `--split_file` | `./training_datasets/dataset_split.json` | parent-video splits |
| `--report` | `dataset_report.html` | output path |

## Report contents

- Summary cards: total clips, SFT/GRPO record counts, unique instruments /
  steps (≥3 appearances), avg clips per video, avg description/reasoning
  word counts.
- Charts: parent videos per split, SFT vs GRPO per split, step / instrument
  / anatomy frequency (horizontal bars), word-count histograms, question
  type & reward type doughnuts, reasoning length by question type, clips
  per video distribution.
- Dark/light mode via `prefers-color-scheme`.

## Usage

```bash
python -m video_pipeline.tools.generate_report
python -m video_pipeline.tools.generate_report --clips-dir ./clip_dataset_mapped --report ./reports/report.html
```

## Notes

- Only instruments/steps/anatomy with ≥3 instances are charted.
- Falls back to empty split stats when the split file is missing.

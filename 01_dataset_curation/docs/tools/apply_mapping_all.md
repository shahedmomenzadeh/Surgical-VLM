# `apply_mapping_all.py` — Label Canonicalization

Applies `step_mapping.json` and `instrument_mapping.json` to normalize the
free-form LLM labels (`step_title`, `instruments`) into canonical values.

## What it does

- Builds reverse lookups: `{variant: canonical}` from the mapping files
  (case-insensitive).
- **Pipeline A (VLM dataset)** — maps `vlm_dataset/` records (nested
  `vlm_annotations[]` structure) → `vlm_dataset_mapped/`, plus a mapped
  aggregate `vlm_dataset_all_mapped.jsonl`.
- **Pipeline B (clip dataset)** — maps `clip_dataset/` clip records (flat
  structure) → `clip_dataset_mapped/`. Video files are symlinked (or
  `.ref.txt` placeholders on filesystems without symlink support).
- Generates reports into `reports_vlm_mapped/` and `reports_clip_mapped/`
  (`report_summary.txt`, `per_video_report.csv`, counter CSVs,
  `unmapped_values.txt`).
- Unmapped values are kept as-is and reported.

## Usage

```bash
python -m video_pipeline.tools.apply_mapping_all
python -m video_pipeline.tools.apply_mapping_all --skip_vlm --clip-out ./clip_mapped
python -m video_pipeline.tools.apply_mapping_all --step-map my_steps.json --instrument-map my_instr.json
```

## Options

`--vlm_dir`, `--clip_dir`, `--step_map`, `--instrument_map`,
`--aggregate_in`, `--vlm_out`, `--clip_out`, `--aggregate_out`,
`--vlm_report_dir`, `--clip_report_dir`, `--skip_vlm`, `--skip_clip`.

## Related

- `replace_clip.py` resolves the `.ref.txt` placeholders created on Windows.
- `generate_report.py` is the newer HTML report generator; this script
  writes the older text/CSV report format.

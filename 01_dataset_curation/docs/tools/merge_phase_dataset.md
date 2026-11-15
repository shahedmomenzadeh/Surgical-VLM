# `merge_phase_dataset.py` - Add Phase Samples to the Final Dataset

Adds accepted phase VLM samples from `phase_vlm_dataset/` into the existing
final `dataset/` directory. The existing split assignments and procedure
folders are preserved.

The tool does not:

- Re-split procedures
- Make LLM/VLM/API calls
- Overwrite existing procedure folders
- Copy rejected or errored phase clips

## Usage

Inspect the merge first:

```bash
cd /mnt/d/programming/MSc_project
video_pipeline/.venv/bin/python -m video_pipeline.tools.merge_phase_dataset \
  --config video_pipeline/config.yaml \
  --dry-run
```

Perform the merge using independent file copies:

```bash
video_pipeline/.venv/bin/python -m video_pipeline.tools.merge_phase_dataset \
  --config video_pipeline/config.yaml
```

Optional arguments:

| Option | Description |
|---|---|
| `--config PATH` | Configuration file to use |
| `--source PATH` | Override the phase VLM source directory |
| `--destination PATH` | Override the final dataset directory |
| `--materialization copy` | Copy files; default and safest final-dataset mode |
| `--materialization hardlink` | Use hardlinks to avoid additional disk usage |
| `--dry-run` | Report procedures, clips, and files without writing |

Defaults are:

```text
source:      phase_vlm_dataset/
destination: dataset/
```

## Merge Layout

The source folders are added to matching split directories:

```text
phase_vlm_dataset/Train/PH_0004_2934_S2/
                       │
                       ▼
dataset/Train/PH_0004_2934_S2/
```

Each accepted phase procedure contributes:

```text
clip_XX.mp4
clip_XX.jsonl
clip_XX_sft.jsonl
clip_XX_grpo.jsonl
```

The relative video paths inside SFT and GRPO records already use the neutral
procedure/clip layout, so they remain valid after the merge.

## Safety Checks

Before writing, the tool checks for procedure-folder collisions. If a phase
procedure folder already exists in the destination, the merge stops without
overwriting it.

After writing, it verifies:

- Every source file exists in the destination
- Source and destination file sizes match
- No copy errors occurred

Files are copied through temporary paths and atomically renamed. The default
`copy` mode makes the final dataset independent of `phase_vlm_dataset`.

The optional `hardlink` mode uses less disk space but depends on the source
files continuing to exist.

## Current Expected Result

The current phase SFT/GRPO output contains 1,184 accepted clips across 150
procedure folders:

| Split | Phase Procedures | Phase Clips |
|---|---:|---:|
| Train | 105 | 925 |
| Validation | 22 | 128 |
| Test | 23 | 131 |
| **Total** | **150** | **1,184** |

The existing dataset contains 1,440 clips. After this merge, the expected
combined dataset contains:

| Split | Existing Clips | Added Phase Clips | Combined Clips |
|---|---:|---:|---:|
| Train | 1,141 | 925 | 2,066 |
| Validation | 137 | 128 | 265 |
| Test | 162 | 131 | 293 |
| **Total** | **1,440** | **1,184** | **2,624** |

These counts reflect the current accepted phase snapshot and will increase if
additional SFT clips pass review later.

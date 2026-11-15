# `phase_grpo_dataset.py` - Deterministic Phase GRPO Generation

Generates one phase-identification GRPO record for every complete,
critic-approved phase SFT clip. This stage is deterministic and makes no
LLM, VLM, or API calls.

The generator reads:

```text
phase_vlm_dataset/{Train,Validation,Test}/{procedure}/
├── clip_XX.mp4
├── clip_XX.jsonl
└── clip_XX_sft.jsonl
```

and creates:

```text
clip_XX_grpo.jsonl
```

Rejected Idle clips and SFT clips with errors are not eligible for GRPO.

## Usage

Run from the project parent directory:

```bash
cd /mnt/d/programming/MSc_project
video_pipeline/.venv/bin/python -m video_pipeline.tools.phase_grpo_dataset \
  --config video_pipeline/config.yaml
```

Process a single split or a small subset:

```bash
video_pipeline/.venv/bin/python -m video_pipeline.tools.phase_grpo_dataset \
  --config video_pipeline/config.yaml \
  --split Validation \
  --limit 10
```

Inspect inputs without writing files:

```bash
video_pipeline/.venv/bin/python -m video_pipeline.tools.phase_grpo_dataset \
  --config video_pipeline/config.yaml \
  --dry-run
```

Options:

| Option | Description |
|---|---|
| `--config PATH` | Configuration file to use |
| `--split SPLIT` | Process only `Train`, `Validation`, or `Test` |
| `--limit N` | Process at most `N` eligible clips after split filtering |
| `--dry-run` | Report eligible and existing files without writing |

## Configuration

The configuration is under `phase_grpo` in `config.yaml`:

```yaml
phase_grpo:
  input_dir: "./phase_vlm_dataset"
  log_file: "phase_grpo_log.csv"
```

No model name, API key, endpoint, or retry setting is used by this stage.

## Eligibility Rules

A clip is eligible only when all of the following exist and are non-empty:

```text
clip_XX.mp4
clip_XX.jsonl
clip_XX_sft.jsonl
```

The core metadata must contain a valid phase ID from `P01` through `P13` and a
non-empty critic-approved `visual_description`.

The generator uses the phase split and procedure folder already assigned by
`phase_dataset`. It does not create a new split and does not mix procedures
between Train, Validation, and Test.

## Ontology Prompt

Every GRPO prompt presents the complete cataract-surgery ontology:

```text
P01 - Incision: Initial corneal incision
P02 - Viscoelastic: Injection of viscoelastic material
P03 - Capsulorhexis: Continuous circular anterior capsulotomy
P04 - Hydrodissection: Fluid separation and lens mobilization
P05 - Phacoemulsification: Ultrasonic lens-nucleus fragmentation and removal
P06 - Irrigation-Aspiration: Removal of cortex or residual lens material
P07 - Capsule Polishing: Cleaning or polishing the lens capsule
P08 - Lens Implantation: Insertion of the intraocular lens
P09 - Lens Positioning: Final positioning of the intraocular lens
P10 - Viscoelastic Suction: Removal of viscoelastic material
P11 - Anterior Chamber Flushing: Flushing or irrigating the anterior chamber
P12 - Tonifying-Antibiotics: Medication or antibiotic administration
P13 - Idle: No active surgical action or inactive time
```

The model is asked to reason from the video and finish with exactly one
canonical ontology ID, for example:

```text
Final answer: P09
```

The phase ID is used instead of a randomized A-D option because it is the gold
label already stored in the phase metadata.

## GRPO Record Schema

Each `clip_XX_grpo.jsonl` contains exactly one JSON object with the same
training-facing field names used by the existing dataset:

```text
prompt
correct_answer
question_type
reference_reasoning
reward_type
```

The record uses:

```text
question_type: "phase_identification"
reward_type: "deterministic"
correct_answer: "P01" through "P13"
```

The video reference preserves the existing multimodal structure:

```text
prompt[0].content[0].type: "video"
prompt[0].content[0].video: "PH_0004_2934_S2/clip_01.mp4"
prompt[0].content[1].type: "text"
prompt[0].content[1].text: ontology question
```

`reference_reasoning` is copied from the critic-approved
`visual_description` in the core metadata. This avoids another model call and
keeps the GRPO rationale consistent with the SFT record.

## Deterministic Reward

The future reward function should extract the final ontology ID from the model
response, normalize case and whitespace, and compare it with `correct_answer`.
The reward should evaluate the final answer rather than compare the entire
reasoning string.

## Resume Behavior

Progress is tracked in:

```text
phase_vlm_dataset/phase_grpo_log.csv
```

The log contains:

```text
record_id
split
parent_video_id
clip_filename
phase_id
status
error
timestamp
```

`SUCCESS` is terminal. Existing valid GRPO files are also detected even when
the log entry is missing, then recorded as successful skips. Invalid or
partial GRPO files are regenerated.

Writes use a temporary file followed by an atomic rename. The operation can be
interrupted and safely rerun.

## Expected Counts

The number of generated phase GRPO records equals the number of accepted phase
SFT clips, not the original 1,300 source clips.

For the current phase SFT snapshot:

| Split | Accepted SFT Clips | Expected GRPO Records |
|---|---:|---:|
| Train | 925 | 925 |
| Validation | 128 | 128 |
| Test | 131 | 131 |
| **Total** | **1,184** | **1,184** |

These counts will change after rejected Idle and error records are rerun.

## Verification

After generation, verify that:

1. Every eligible SFT clip has one `_grpo.jsonl` file.
2. Every GRPO file contains exactly one JSON object.
3. Every `correct_answer` is a valid phase ID.
4. Every `question_type` is `phase_identification`.
5. Every `reward_type` is `deterministic`.
6. Every video path points to the neutral `procedure/clip_XX.mp4` path.
7. No GRPO files were created for rejected or errored SFT clips.

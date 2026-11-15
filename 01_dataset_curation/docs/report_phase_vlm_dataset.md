# Phase-Recognition VLM Dataset — Implementation Report

## 1. Overview

This report describes the construction of a phase-recognition Vision-Language
Model (VLM) training dataset derived from a manually labeled cataract-surgery
phase-recognition corpus. The goal is to produce two complementary training
tracks per video sub-clip:

1. **SFT (Supervised Fine-Tuning)** — a teacher VLM produces a grounded,
   phase-relevant visual observation of each clip; a critic VLM verifies it
   against the video to remove hallucinations and phase-inconsistent claims.
2. **GRPO (Group Relative Policy Optimization)** — a deterministic
   phase-classification sample per clip, where the manually verified phase
   label is the gold answer and the reward is exact match on the canonical
   phase ID.

The implementation currently covers the complete **SFT generation stage**
(Stage 1), which also acts as the quality gate for the dataset: clips with
incorrect Idle boundaries are detected and removed before any training
records are produced. The GRPO stage (Stage 2) and the merge with the
existing YouTube-derived dataset (Stage 3) are designed but not yet
implemented.

All code lives in the self-contained `video_pipeline` package:

| Component | Role |
|---|---|
| `tools/phase_sft_dataset.py` | Teacher/critic SFT generation pipeline |
| `tools/phase_media.py` | Video base64 encoding and frame fallback |
| `config.py` / `config.yaml` | `phase_sft` configuration section |
| `docs/tools/phase_sft_dataset.md` | Detailed tool documentation |

## 2. Source Data

### 2.1 Collection

The source collection is the Phase Recognition sub-clip dataset:

```text
hf_zips/1_Phase_Recognition/annotations_sub_clips/
```

- 150 source surgical procedures (129 from site `S1`, 21 from site `S2`)
- 5,335 phase sub-clips covering 13 surgical phases of cataract surgery
- Videos are 5 FPS, 480p height, H.264; total source duration 33 h 38 m 52 s
- Each sub-clip carries a **manually verified phase label** from the 13-phase
  ontology

### 2.2 The 13-phase surgical ontology

| ID | Phase | Description |
|---|---|---|
| `P01` | Incision | Initial corneal incision |
| `P02` | Viscoelastic | Injection of viscoelastic material |
| `P03` | Capsulorhexis | Continuous circular anterior capsulotomy |
| `P04` | Hydrodissection | Fluid separation and lens mobilization |
| `P05` | Phacoemulsification | Ultrasonic lens-nucleus fragmentation and removal |
| `P06` | Irrigation-Aspiration | Removal of cortex or residual lens material |
| `P07` | Capsule Polishing | Cleaning or polishing the lens capsule |
| `P08` | Lens Implantation | Insertion of the intraocular lens |
| `P09` | Lens Positioning | Final positioning of the intraocular lens |
| `P10` | Viscoelastic Suction | Removal of viscoelastic material |
| `P11` | Anterior Chamber Flushing | Flushing or irrigating the anterior chamber |
| `P12` | Tonifying-Antibiotics | Medication or antibiotic administration |
| `P13` | Idle | No active surgical action or inactive time |

`P13` is a real phase, not missing data. `P01`–`P12` are active phases.

### 2.3 Per-clip information in the manifest

The selection result is recorded in `manifest.csv` with one row per sample:

| Column | Meaning |
|---|---|
| `split` | Train / Validation / Test |
| `sample_id` | Original sub-clip filename without extension |
| `source_procedure` | Source procedure directory |
| `site` | S1 or S2 |
| `phase_id` | P01–P13 |
| `phase_name` | Phase name |
| `subclip_order` | Temporal order in the source procedure |
| `phase_occurrence` | Occurrence number of that phase in the procedure |
| `duration_sec` | Measured duration in seconds |
| `previous_phase` | Phase of the previous sub-clip (empty at procedure start) |
| `next_phase` | Phase of the next sub-clip (empty at procedure end) |
| `context_bucket` | Position-in-timeline bucket (see below) |
| `source_path` | Path in the source dataset |
| `dataset_path` | Path inside the selected dataset |
| `materialization` | hardlink / copy / symlink |

The **original filenames encode the ground truth**: e.g.
`PH_0004_2934_S2_0002_P01_01.mp4` means procedure `PH_0004_2934_S2`,
sub-clip order 2, phase `P01`, occurrence 1. For this reason the phase VLM
dataset materializes clips under neutral names (`clip_XX.mp4`); the phase
label must not leak through the file path presented to a student model.

### 2.4 Context buckets

Each sample is tagged with its position inside the source procedure timeline:

| Bucket | Meaning |
|---|---|
| `transition_adjacent` | Active clip bordered by at least one different phase |
| `between_different_active` | Idle between two different active phases |
| `before_active` | Idle directly before the first active phase |
| `after_active` | Idle directly after the last active phase |
| `between_same_active` | Idle between two occurrences of the same active phase |
| `long_idle` | Idle clip of 30 s or longer |
| `procedure_edge` | Active clip at the start or end of the procedure |

This context is valuable: it allows Idle samples to be balanced across
temporal contexts and, in the GRPO design, supports choosing clinically
plausible distractor phases.

### 2.5 Selection design

The selected dataset contains **1,300 samples**:

| Split | Samples | Sampled procedures | Assigned procedures |
|---|---:|---:|---:|
| Train | 1000 | 105 | 105 |
| Validation | 150 | 22 | 22 |
| Test | 150 | 23 | 23 |

Design decisions:

- **Procedure-level splits**: complete procedures are assigned to one split
  before selection, preventing data leakage across clips of the same surgery.
- **Phase balance**: about 75 samples per active phase and 150 Idle samples in
  Train (15% Idle), scaled down in Validation/Test. `P11` is the rarest phase
  in the source data, so its target could not be fully reached.
- **Site oversampling**: site `S2` was intentionally oversampled to ~30% of
  each phase sample (its natural share is only 14%).
- **Duration preference**: clips of 8–45 s score higher in the quality
  heuristic; sub-clips shorter than 3.0 s are excluded entirely.
- **Deterministic selection**: seed `20260807`; samples are picked
  round-robin across procedures before taking a second sample from the same
  procedure.
- **Materialization**: videos are hard links to the source files (no extra
  disk usage).

### 2.6 Selected dataset statistics

Total duration ≈ 11 h 14 m. Duration statistics (seconds):

| Split | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Train | 31.3 | 19.6 | 3.0 | 530.0 |
| Validation | 31.4 | 20.1 | 3.5 | 236.1 |
| Test | 29.4 | 20.5 | 4.0 | 184.1 |

Phase distribution (all splits):

| Phase | Samples | Total | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| P01 Incision | 99 | 32 m 29 s | 19.7 | 17.6 | 6.0 | 46.8 |
| P02 Viscoelastic | 99 | 23 m 33 s | 14.3 | 12.7 | 3.5 | 61.8 |
| P03 Capsulorhexis | 97 | 1 h 20 m 33 s | 49.8 | 47.0 | 16.6 | 118.1 |
| P04 Hydrodissection | 97 | 39 m 06 s | 24.2 | 21.9 | 4.9 | 70.8 |
| P05 Phacoemulsification | 97 | 3 h 02 m 04 s | 112.6 | 104.2 | 8.0 | 530.0 |
| P06 Irrigation-Aspiration | 97 | 1 h 12 m 14 s | 44.7 | 35.6 | 9.6 | 236.1 |
| P07 Capsule Polishing | 97 | 18 m 53 s | 11.7 | 10.8 | 3.4 | 40.8 |
| P08 Lens Implantation | 97 | 25 m 49 s | 16.0 | 14.4 | 3.0 | 48.9 |
| P09 Lens Positioning | 97 | 34 m 45 s | 21.5 | 18.5 | 4.4 | 123.5 |
| P10 Viscoelastic Suction | 97 | 1 h 03 m 21 s | 39.2 | 37.2 | 9.1 | 154.5 |
| P11 Anterior Chamber Flushing | 36 | 7 m 05 s | 11.8 | 9.3 | 3.4 | 52.8 |
| P12 Tonifying-Antibiotics | 96 | 38 m 34 s | 24.1 | 20.8 | 3.3 | 106.6 |
| P13 Idle | 194 | 55 m 38 s | 17.2 | 13.7 | 3.5 | 64.1 |

## 3. Intuition Behind the Phase VLM Dataset

### 3.1 Gold labels versus free-form annotations

The existing training dataset (`dataset/`) is built from YouTube surgical
videos and LLM-generated step annotations. The step titles there are
free-form (e.g. "main corneal incision") and are not standardized against a
fixed ontology. The phase-recognition corpus is fundamentally different: it
provides **manually verified labels** against a well-defined 13-phase
ontology, with a rigorous procedure-level train/validation/test split.

This makes the phase dataset the natural source of **gold deterministic
supervision** for surgical-phase understanding.

### 3.2 Two complementary training tracks

For each sub-clip we can create two samples that exercise different skills:

1. **GRPO — phase identification**: given the clip and the full 13-phase
   ontology, the model must reason and output the correct phase ID
   (`P01`–`P13`). Because the label is human-verified, the reward can be
   deterministic exact-match. This is a classification-style task that
   grounds the model in the temporal structure of cataract surgery.
2. **SFT — grounded visual observation**: a strong teacher VLM watches the
   same clip and writes a *visual* description (instruments, anatomy,
   actions, observable changes), conditioned on the hidden gold phase label.
   A critic VLM then validates it against the video. This produces rich
   visual-language data without requiring manual captioning, while the gold
   label keeps the description phase-relevant.

The two tracks share the video and the gold label but ask for complementary
outputs: a short deterministic decision (GRPO) and a detailed grounded
description (SFT).

### 3.3 Teacher/critic design rationale

- The phase label alone cannot teach the model *why* a phase looks the way it
  does; the teacher converts the label into visual evidence.
- Teachers hallucinate; a critic that re-inspects the video and the proposal
  removes unsupported instruments, anatomy, actions, or complications before
  anything enters the dataset.
- The teacher must not echo the label into the description, otherwise the SFT
  target would leak the answer. The phase label is conditioning context for
  the teacher/critic, but is **never** present in the student-facing prompt.
- Both models run on the same OpenAI-compatible endpoint
  (`http://localhost:8000/v1`, model `qwen3.8-max`), which accepts full
  videos as base64 data URIs.

### 3.4 Idle boundary quality and the SFT-first gate

A known quality defect exists in the source data: **some Idle clips
(`P13`) have imperfect boundaries** — active surgery can appear inside a clip
labeled Idle. If such clips were used as-is, the model would learn to call
active surgery "Idle", corrupting both the GRPO targets and the SFT
descriptions.

The solution is to run **SFT generation first** as the quality gate: the
teacher and critic are instructed that if any recognizable surgical maneuver
is visible in a candidate Idle clip, they must **reject/skip** the clip. Only
clips that pass this gate are materialized; the GRPO stage will later consume
only accepted clips. A slightly smaller final count is acceptable — data
quality trumps sample count.

### 3.5 Neutral naming against label leakage

The source filenames encode the phase (`..._0002_P01_01.mp4`). If these
filenames were used as training video paths, a model (or a file-based
loader) could read the answer from the path. Accepted clips are therefore
materialized under procedure folders with neutral names:

```text
PH_0004_2934_S2/clip_01.mp4
```

The phase label lives only in the metadata JSONL, never in the path.

## 4. Pipeline Flow

### 4.1 Overall flow

```text
phase_dataset (1,300 manually labeled sub-clips)
        │
        ▼
 Stage 1: SFT GENERATION (implemented, quality gate)
   per clip: full-video teacher  →  critic validation
        │  (Idle clips with visible surgery are rejected)
        ▼
 phase_vlm_dataset/{Train,Validation,Test}/{procedure}/
        │        clip_XX.mp4 + clip_XX.jsonl + clip_XX_sft.jsonl
        ▼
 Stage 2: GRPO GENERATION (designed, next)
   per accepted clip: deterministic phase-identification record
        │
        ▼
        │  clip_XX_grpo.jsonl
        ▼
 Stage 3: MERGE (planned)
   union with dataset/ by split → ~2,740 clips total
```

### 4.2 Stage 1 — SFT generation (implemented)

**Inputs**

- `phase_dataset/manifest.csv` (the source of truth)
- `phase_dataset/{split}/{procedure}/{sample_id}.mp4` (resolved via
  `dataset_path` with a fallback)
- `config.yaml` section `phase_sft`

**Per-clip processing loop**

1. **Enumerate**: read the manifest, sort records by split, procedure, and
   `subclip_order`, and assign stable neutral clip numbers
   (`clip_01.mp4`, `clip_02.mp4`, ...). Rejected clips can leave gaps in the
   numbering, which keeps paths stable across resumed runs.
2. **Resume check**: skip records whose latest status in `phase_sft_log.csv`
   is terminal (`SUCCESS`, `REJECTED_IDLE_BOUNDARY`), unless
   `--retry-rejected` is passed. Completed review checkpoints under
   `phase_vlm_dataset/.work/reviews/` are reused so teacher/critic API calls
   are never repeated after an interruption.
3. **Media preparation**: encode the full MP4 as a base64 `video/mp4` data
   URI and send it as a `video_url` content part. Full video is always
   attempted first — it preserves motion and timing, which matter for
   surgical-phase understanding.
4. **Fallback**: if the endpoint returns an upstream/payload error (HTTP
   502/503/504/413, "upstream error", "gateway timeout", ...), the clip is
   re-requested as **at most 32 evenly spaced base64 JPEG frames**. Before
   extraction, `ffprobe` counts the actual decoded frames, so the number
   sent is `min(actual_frame_count, 32)` — a 15-frame clip sends 15 frames.
5. **Teacher call**: the teacher receives the video (or frames) plus the
   gold `phase_id`/`phase_name` as hidden conditioning. It must produce a
   strictly visual observation: visible instruments, anatomy, actions, and
   observable changes — not a textbook definition of the phase. For
   candidate `P13` clips it must return `SKIP` if any active surgical
   operation is visible.
6. **Critic call**: the critic receives the same video, the gold phase, and
   the teacher's proposal. It checks visual grounding and phase consistency,
   corrects over-claims, and returns `ACCEPT`, `SKIP`, or `REJECT`. The
   critic-approved text becomes the final description.
7. **Idle rejection**: if the phase is `P13` and either model reports active
   surgery, the clip is logged as `REJECTED_IDLE_BOUNDARY` and skipped. No
   MP4, metadata, or SFT files are written; the count simply decreases.
8. **Materialization**: on acceptance the video is copied/hard-linked to
   `clip_XX.mp4`, rich metadata is written to `clip_XX.jsonl`, and exactly
   one SFT record is written to `clip_XX_sft.jsonl`. All writes are atomic
   (temporary file + rename).
9. **Logging**: every decision is appended to `phase_sft_log.csv` with
   status, attempts, media input modes (`full_video`/`frames`), frame
   counts, and error text.

**Robustness**

- Exponential retries with backoff per request
- `response_format` fallback for gateways that reject `json_object`
- Markdown-fence/thinking-tag stripping before JSON parsing
- Per-clip exception isolation (one bad clip never aborts the run)
- Terminal-state checkpoints for resumability
- Optional `--split`, `--limit`, `--dry-run` for controlled execution

**Outputs per accepted clip**

```text
phase_vlm_dataset/{split}/{procedure}/
├── clip_XX.mp4          # neutral-named video
├── clip_XX.jsonl        # rich phase metadata
└── clip_XX_sft.jsonl    # one SFT record
```

The metadata contains `clip_id`, `phase_clip_id`, `parent_video_id`,
`clip_filename`, `video_path`, `phase_id`, `phase_name`, `step_number`,
`step_title`, durations, `previous_phase`, `next_phase`, `context_bucket`,
`site`, `subclip_order`, `phase_occurrence`, the final
`visual_description`, `instruments`, `anatomy`, provenance
(`source_path`, `dataset_path`, `materialization`), and model audit fields
(`teacher_model`, `critic_model`, `quality_status`).

The SFT record preserves the existing training schema
(`messages`/`content`/`type:"video"`/`video`/`text`) so it can be merged
with the current dataset and consumed by the same training loaders.

**Verified behavior (real run)**

- A `P13` Idle clip (`PH_0001_2931_S2_0001_P13_01`) was correctly rejected:
  the model reported active surgical manipulation entering the field in the
  latter half of the clip; no training files were produced.
- A `P01` Incision clip (`PH_0001_2931_S2_0002_P01_01`) passed teacher and
  critic and produced `clip_02.mp4`, `clip_02.jsonl`, and
  `clip_02_sft.jsonl` with a grounded description (keratome incision,
  speculum, forceps, conjunctival bleeding at contact points).
- Full-video requests were used (no fallback needed); the fallback path was
  additionally validated with a simulated `503 upstream error` and a
  real 15-frame clip (15 frames sent, not 32).

### 4.3 Stage 2 — GRPO generation (designed)

The next stage reads **only accepted clips** from `phase_vlm_dataset` and
creates exactly one GRPO record per clip:

```text
prompt:  "Given this video, identify the surgical phase from the surgical
          ontology." + the full 13-phase list (P01...P13 with names)
correct_answer: "P01" ... "P13"        # the gold label
question_type:  "phase_identification"
reward_type:    "deterministic"
```

The student-facing prompt shows the complete ontology and never reveals the
correct phase. Reasoning is requested before the final phase ID. Because
the answer is the canonical ID, there is no ambiguity between alternative
phase names (I/A vs Irrigation-Aspiration). The output field names match the
existing GRPO schema, enabling the later merge.

### 4.4 Stage 3 — Merge with the existing dataset (planned)

The phase records will be merged into the existing `dataset/` without
re-splitting: phase `Train` → `dataset/Train`, and likewise for Validation
and Test. The procedure IDs of the two sources are distinct, so no folder
collisions occur. Expected totals:

| Split | Existing clips | Phase clips | Merged |
|---|---:|---:|---:|
| Train | 1,141 | 1,000 | 2,141 |
| Validation | 137 | 150 | 287 |
| Test | 162 | 150 | 312 |
| Total | 1,440 | 1,300 | 2,740 |

The ~2,600 new *training records* (1,300 SFT + 1,300 GRPO) correspond to
1,300 new *video clips*.

## 5. Configuration and Infrastructure

The `phase_sft` section of `config.yaml`:

| Setting | Value |
|---|---|
| Endpoint | `http://localhost:8000/v1` (OpenAI-compatible) |
| Model | `qwen3.8-max` |
| API key | literal `none` (local server); env fallback `PHASE_SFT_*_API_KEY` |
| Media | full-video base64 first; ≤32 evenly spaced JPEG frames fallback |
| Materialization | hardlink (fallback to copy) |
| Resume | `phase_sft_log.csv` + `.work/reviews/` checkpoints |

Runtime dependencies: `openai`, `pydantic`, `pyyaml`, `tqdm`,
`python-dotenv`, `ffmpeg`/`ffprobe` (fallback only). All installed in the
project's uv-managed `.venv`.

## 6. Next Steps

1. Implement the GRPO generation stage over accepted clips.
2. Implement the split-preserving merge with `dataset/`.
3. Run the full 1,300-clip SFT generation (Validation-limited runs already
   verified end-to-end).
4. Optionally, replace rejected Idle clips from the source pool if the final
   count must be closer to 1,300.

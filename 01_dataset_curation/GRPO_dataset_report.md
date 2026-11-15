# 3. Methodology — GRPO Dataset Report (Verified Against `dataset_grpo`)

> **Scope.** This report is a verified rewrite of `academic_report.md` §3 against the
> *training* GRPO view at `dataset_grpo/` (mirrored from `dataset/` via
> `tools/build_separated_datasets.py`), the SFT view at `dataset_sft/`, the monolith
> at `dataset/`, and the generator at `phase_grpo_dataset/`. All counts below were
> obtained by sampling `*_grpo.jsonl`/`*_sft.jsonl` and are reproducible with the
> `Verify` snippets in §3.9. The monolith `dataset/` is retained for provenance;
> `dataset_grpo/` is the canonical GRPO training set.

## 3.1 Dataset Design and Sources

Developing vision-language models (VLMs) for complex surgical domains such as cataract
surgery (phacoemulsification) requires representations that capture multiple levels of
visual perception, clinical reasoning, and temporal understanding. Existing surgical vision
datasets have predominantly focused on frame-level recognition, phase recognition, or short
video classification, providing comparatively limited supervision for open-ended visual
description, procedural reasoning, and reinforcement-learning-based alignment.

To address these limitations, we constructed a **dual-source, dual-resolution surgical
dataset** that combines complementary forms of supervision from in-the-wild surgical
recordings and an annotated clinical phase-recognition corpus. The collection is
materialized in three views:

* `dataset/` — monolith (active + `archived/` provenance);
* `dataset_sft/` — **SFT-only** mirrored view (hardlinked videos);
* `dataset_grpo/` — **GRPO-only** mirrored view (hardlinked videos, this report).

The combined inventory spans **286 parent procedures** and **136 full-length surgical
procedures** (YouTube track), with **3,824 video clips** in the GRPO view
(`2,624` clip-level `clip_*.mp4` + `1,200` phase-localization `grpo_*.mp4`; see §3.8):

| View | Parent procedures | Clip `clip_*.mp4` | GRPO-track `grpo_*.mp4` | Full-length `full_video.mp4` | SFT files | GRPO records |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dataset` (monolith, active) | 286 | 2,624 | 0 (1,200 live in `dataset_grpo`; 1,184 + 136 archived in `dataset/archived/`) | 136 | 2,624 files / 6,944 lines* | 4,321 active (YouTube); 1,184 + 1,200 + 136 archived |
| `dataset_sft` (SFT-only) | 286 | 2,624 | — | 136 | 2,624 files / 6,944 lines + 136 full-video | — |
| **`dataset_grpo` (GRPO-only, canonical)** | **286** | **2,624** | **1,200** | **136** | — | **5,521** |

\*Each YouTube `clip_*_sft.jsonl` contains **4 lines** (1 description + 3 MCQ with
CoT); each phase `clip_*_sft.jsonl` contains **1 line**. `dataset_sft/README.md` counts
files; line-level SFT supervision is 6,944 + 136. `dataset_grpo` carries no `*_sft.jsonl`.

For GRPO training, `dataset_grpo` provides **5,521 records**, all
`reward_type="deterministic"` (see §3.6). The complementary SFT supervision remains
available in `dataset_sft` (and in `dataset/`). The former per-clip
`phase_identification` GRPO (1,184 records, one per phase clip) is preserved only in
`dataset/archived/` and is **not** part of the active GRPO training set.

The first source, the **In-the-Wild Surgical Corpus** (YouTube, `YouTube ID` folders),
consists of publicly available cataract surgery recordings. These recordings provide
visual and procedural diversity — microscope systems, illumination, image quality,
techniques, instrumentation, anatomy, and intraoperative events — plus surgeon commentary
where available (transcribed with Whisper large at 16 kHz mono, §3.3.1), supporting
learning from heterogeneous, naturally occurring observations.

The second source, the **Clinical Phase Corpus** (procedure folders `PH_*`, derived from
`phase_dataset/` via `phase_vlm_dataset/` and `phase_grpo_dataset/`), consists of
annotated clips against a standardized **13-phase cataract surgery ontology** (P01–P13).
It supplies structured, verifiable phase supervision both for SFT clip descriptions and
for temporal GRPO tasks extracted from the stitched full procedures in
`full_surgical_videos/` (110 procedures).

| Dimension | In-the-Wild Surgical Corpus | Clinical Phase Corpus | Combined Benefit |
| :--- | :--- | :--- | :--- |
| Primary video format | Full procedures (10–25 min) + step clips (8–15 s, 10 fps, 480p) | Phase sub-clips (SFT, ~14 s) + phase-localization clips (10–41 s, ≤50 frames, adaptive 2.4–5 fps, mean 25.8 s) | Dual-resolution temporal representation |
| Visual context | Diverse microscopes, image quality, illumination, viewpoints | Controlled clinical acquisition | Robustness to visual variability |
| Audio / narration | Surgeon commentary, transcribed & LLM-refined where available | No narration required | Linguistic grounding + visual verification |
| Taxonomy | Open-vocabulary steps / instruments / anatomy | Closed 13-phase ontology (P01–P13) | Open-ended reasoning + standardized classification |
| Supervision | Step / instrument / visual-observation MCQ + full-video narration | Grounded visual description (SFT) + 4 temporal phase tasks (GRPO) | Complementary SFT and deterministic GRPO supervision |

*Verification:* `dataset_grpo/README.md:30-42`, `phase_grpo_dataset/stats.json`,
`dataset/README.md`, `dataset_sft/README.md`, `config.yaml:directories`.

## 3.2 Dual-Resolution Temporal Representation

Surgical procedures exhibit hierarchical temporal structure. Fine-grained actions occur over
seconds; clinically meaningful transitions and procedural context span minutes. The dataset
represents procedures at two resolutions; the GRPO view inherits both.

### 3.2.1 Micro-Resolution: Step / Phase Clips

At micro-resolution, short clips are extracted with frame-accurate segmentation:

* **YouTube step clips** (`clip_*.mp4`): mean 8–15 s per the monolith metadata; 10 FPS
  after downsampling (`config.yaml:downsample.target_fps: 10`). Each carries structured
  metadata (`clip_*.jsonl`) with `step_title`, `visual_description`, `instruments`,
  `anatomy`, and transcript context.
* **Phase clips** (SFT, `PH_*/clip_*.mp4`): 1,184 clips labeled with gold `phase_id`
  (P01–P13) and a teacher/critic-approved `visual_description` (`PH_*/clip_*.jsonl`,
  32 keys). SFT supervision is 1 line per clip.
* **Phase-localization GRPO clips** (`PH_*/grpo_*_*.mp4`, only in `dataset_grpo`):
  1,200 clips extracted from `full_surgical_videos/` stitched procedures; duration
  11.1–41.1 s, mean 25.8 s, ≤50 frames, adaptive FPS 2.4–5.0 (mean 3.93),
  480p H.264 (`phase_grpo_dataset/stats.json`). Each carries a prompt embedding the
  fixed 13-phase legend and a strict JSON output schema (see §3.9.2).

The clip-level data support the following question types in the **GRPO view**:

1. **Step identification** — operative maneuver (YouTube).
2. **Instrument identification** — actively used surgical instrument (YouTube).
3. **Visual observation** — visible anatomical / dynamic surgical change (YouTube).
4. **Contextual phase recognition** — phase in the target portion after an inactive
   interval (phase-localization).
5. **Timestamp-to-phase** — phase at a given clip-local timestamp (phase-localization).
6. **Boundary detection** — start timestamp of a specified phase (phase-localization).
7. **Temporal localization** — start/end interval of a specified phase (phase-localization).

The former `phase_identification` (single-label 13-way per phase clip, 1,184 records) is
archived and excluded from `dataset_grpo`.

### 3.2.2 Macro-Resolution: Full Procedures

At macro-resolution, 136 complete YouTube procedures are retained as `full_video.mp4`
with one timestamped narration line per video (`full_video_sft.jsonl`,
`[MM:SS - MM:SS] step: description`). Full videos are mirrored into both
`dataset_sft/` and `dataset_grpo/` (hardlinked; 108 Train / 13 Val / 15 Test).
The narration provides long-context procedural supervision; the 1,200
phase-localization GRPO clips are the only GRPO samples derived from full-procedure
temporal structure (global → clip-local coordinates stored in `phase_grpo_dataset`
provenance, see `dataset/README.md:§4`).

The resulting hierarchy spans localized actions over seconds and complete procedures over
minutes, providing complementary supervision for fine-grained perception and long-range
procedural reasoning.

## 3.3 Data Curation and Preprocessing

The pipeline converts raw recordings into temporally aligned, quality-controlled
multimodal samples: video acquisition and filtering, speech transcription, medical
lexical refinement, multimodal surgical annotation, temporal segmentation, sample
formatting, and quality control.

### 3.3.1 Speech Transcription and Medical Lexical Refinement

For recordings with surgeon commentary, audio is extracted as single-channel 16-bit PCM
WAV at 16 kHz. **Whisper large** (`config.yaml:whisper.model_size: large`) generates
timestamped segments; an OpenAI-compatible LLM (`config.yaml:refiner`) performs medical
lexical normalization while preserving temporal boundaries (e.g. *capsule rexis* →
*capsulorhexis*, *faco* → *phacoemulsification*, *keratom* → *keratome*). This stage
applies only to the YouTube corpus.

### 3.3.2 Multimodal Surgical Annotation

In-the-wild recordings undergo textual screening (gatekeeper, `prompts.py:GATEKEEPER_PROMPT`)
and multimodal analysis (analyst, `prompts.py:ANALYST_PROMPT`) to produce structured
event descriptions: step boundaries, visual descriptions, instruments, and anatomical
structures. Procedural descriptions are subsequently screened for intraoperative events
(posterior capsule rupture, vitreous loss, iris prolapse, zonular dialysis, IFIS, nucleus
drop, IOL drop) when supported by visual evidence (`prompts.py:SAFETY_ANALYST_PROMPT`).

### 3.3.3 Teacher-Critic Quality Control for Phase Data

The clinical phase corpus undergoes independent QC before SFT/GRPO generation
(`config.yaml:phase_sft`, `phase_vlm_dataset/`). A **Teacher-Critic** framework generates a
grounded visual observation (Teacher) and an independent visual review (Critic); samples
with unsupported anatomical, instrumental, or procedural claims are rejected. Provenance
for accepted clips is retained in `phase_dataset/manifest.csv` and
`phase_vlm_dataset/` reviews. Special handling for **P13 (Idle)**: passive support
devices (specula, retractors) are distinguished from active intraocular manipulation;
Idle clips showing active action are rejected to reduce boundary ambiguity.
Approved `visual_description` fields are reused as SFT targets and as
`reference_reasoning` in GRPO.

## 3.4 Data Partitioning and Leakage Prevention

To prevent temporal or procedural leakage, partitioning is performed at the
**parent-procedure level** — all clips (and any derived GRPO samples) from a given
procedure inherit the same split assignment (`dataset/grpo_procedure_split_map.json`,
`dataset_grpo/procedure_split_map.json`).

The final inventory contains **213 training procedures, 35 validation procedures, and
38 test procedures** (74.5% / 12.2% / 13.3% of the 286 parents), consistent across
`dataset`, `dataset_sft`, and `dataset_grpo` (verified: folder counts identical across
views). A fixed pseudo-random mapping determines the assignment; the mapping files store
the result. For the phase-localization GRPO track, the generator's internal
`phase_grpo_dataset/splits.json` assignment (`train:959 / val:130 / test:111` over 110
procedures) is **ignored** — each of the 1,200 samples is reassigned by its parent
`PH_*` procedure folder, giving the training distribution **Train 828 / Val 162 /
Test 210** (`dataset/README.md:§Split Policy`; `dataset_grpo/README.md:25`).

This procedure-level policy ensures no clips from the same surgery appear in multiple
splits, even across the YouTube and phase corpora.

## 3.5 SFT Dataset Construction (Context for GRPO)

The SFT dataset (`dataset_sft/`, and equivalently `dataset/` active) provides direct
multimodal supervision: per-clip instruction + target response linking visual evidence to
a clinical concept, plus per-full-video timestamped narration. GRPO training is designed
to follow SFT.

| SFT artifact | Files | Lines / records | Location |
| :--- | ---: | ---: | :--- |
| YouTube clip SFT (`clip_*_sft.jsonl`) | 1,440 | 5,760 lines (1 description `Describe what is happening...` + 3 MCQ with CoT per clip) | `dataset_sft/{split}/{YT_ID}/` |
| Phase clip SFT (`clip_*_sft.jsonl`) | 1,184 | 1,184 lines (teacher/critic description `Describe the visually observable...`) | `dataset_sft/{split}/PH_*/` |
| Full-procedure narration (`full_video_sft.jsonl`) | 136 | 136 lines (`Provide a detailed timestamped narration...` → `[MM:SS - MM:SS] step: ...`) | `dataset_sft/{split}/{YT_ID}/` |
| **Total SFT** | **2,760 files** | **7,080 lines** | — |

`dataset_sft/README.md` reports file counts (`2,624` clip SFT files + `136` full-video
files = `2,760`). The line-level total (`6,944` clip lines + `136` full-video lines =
`7,080`) is the effective SFT sample count for training. The earlier report's
"`2,624 clip-level samples`" conflates files with lines.

## 3.6 GRPO Dataset Construction and Verifiable Rewards

GRPO samples provide verifiable objectives for RL-based alignment. Each record in
`dataset_grpo` contains a multimodal `prompt`, a typed `correct_answer`, a
`question_type`, a `reference_reasoning` string, and `reward_type="deterministic"`.
No `llm_judge` records remain in the GRPO view.

### 3.6.0 Canonical GRPO Inventory (Corrected)

| # | Question Type | Source | Videos | GRPO Files | Records | Reward |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| 1 | `step_identification` | YouTube | 1,440 `clip_*.mp4` | 1,440* | 1,441 | Deterministic (MCQ letter) |
| 2 | `instrument_identification` | YouTube | — | — | 1,440 | Deterministic (MCQ letter) |
| 3 | `visual_observation` | YouTube | — | — | 1,440 | Deterministic (MCQ letter) |
| | **YouTube subtotal** | | **1,440** | **1,440** | **4,321** (3 lines in most `clip_*_grpo.jsonl`) | **100% deterministic** |
| 4 | `temporal_localization` | Clinical phase-localization (from `full_surgical_videos/`, via `phase_grpo_dataset/`) | 1,200 `grpo_*.mp4` | 1,200 | 300 | Deterministic (tIoU) |
| 5 | `contextual_phase_recognition` | — | — | — | 300 | Deterministic (phase exact-match) |
| 6 | `boundary_detection` | — | — | — | 300 | Deterministic (exponential decay) |
| 7 | `timestamp_to_phase` | — | — | — | 300 | Deterministic (phase exact-match) |
| | **Phase-localization subtotal** | | **1,200** | **1,200** | **1,200** | **100% deterministic** |
| | **Active GRPO total (`dataset_grpo`)** | | **2,640** | **2,640** | **5,521** | **100% deterministic** |

\*`YouTube GRPO` is stored as **3 lines per `clip_*_grpo.jsonl`** (one per question type;
a handful of clips contribute a fourth/duplicate line accounting for the 1,441 vs 1,440
offset). `grep -r llm_judge dataset_grpo --include="*.jsonl" | wc -l` → **0**.
The monolith `dataset/` retains 1,440 `llm_judge` records in its **archived** visual-observation
history and in the pre-migration prompt, but the migrated `dataset_grpo` view is
fully deterministic (`prompts.py:GRPO_JSON_INSTRUCTION`).

For completeness, two **archived (inactive)** GRPO tracks remain in `dataset/archived/`:

* **Phase clip `phase_identification`** — 1,184 records (one deterministic 13-way per
  `PH_*/clip_*.mp4`, `Final answer: PXX` format); clips/SFT remain active, GRPO is
  archived (`dataset/archived/{split}/PH_*/clip_*_grpo.jsonl`).
* **Full-video reordering QA** — 136 records (one `sequence_ordering` per
  `full_video.mp4`); archived at `dataset/archived/{split}/{YT_ID}/full_video_grpo.jsonl`.

If these are added to the active counts, the monolith lineage totals 6,841 (5,521 +
1,184 + 136) records; `dataset/README.md:§Archived Data` is authoritative for the
archived inventory.

### 3.6.1 Reward Definitions (All Deterministic)

All rewards are rule-based and require no LLM judge. Every prompt enforces a **strict
JSON output** with exactly two keys `{"explanation", "answer"}` — `explanation` is
1–3 sentences of natural-language evidence; `answer` is task-typed.

#### YouTube Track (MCQ) — `step_identification` / `visual_observation` / `instrument_identification`

Expected model output:

```json
{"explanation": "In the video the surgeon uses a keratome to create a corneal incision.", "answer": "B"}
```

Task reward (`dataset_grpo/README.md:§Reward Functions`):

```
R_task = 1.0 if json.loads(pred)["answer"] == gold["correct_answer"] else 0.0
# strict: pred must be valid JSON with exactly {explanation, answer}, answer ∈ {A,B,C,D}
```

#### Phase-Localization Track — 4 Tasks (proposed in `dataset_grpo/README.md:§2`)

Expected output per task (same outer JSON, typed `answer`):

| Task | `correct_answer` shape | `R_task` |
| :--- | :--- | :--- |
| `boundary_detection` | `{"timestamp": float}` — clip-local seconds of the phase start | Exponential decay on absolute error: `R_task = exp(-|t_pred − t_gt| / 1.5)` with τ=1.5 s. Fallback regex extraction if JSON is malformed; exact match →1.0, 0.5 s error →0.717, 1.5 s →0.368 |
| `temporal_localization` | `{"start": float, "end": float}` — clip-local interval of the target occurrence | Temporal IoU: `inter = max(0, min(pred_end, gt_end) − max(pred_start, gt_start))`, `union = (pred_end−pred_start)+(gt_end−gt_start)−inter`, `R_task = inter/union` |
| `contextual_phase_recognition` | `{"phase_id": "P0X", "phase_name": "..."}` | Normalized exact match: `R_task = 1.0 if normalize(pred_phase)==gt_phase else 0.0` (aliases normalized, e.g. `P06` ≡ Irrigation-Aspiration) |
| `timestamp_to_phase` | `{"phase_id": "P0X", "phase_name": "..."}` or `{"phase": "P0X"}` | Same normalized exact match as above (queried timestamp ∈ clip) |

#### Format Reward (Phase-Localization Tasks)

`R_fmt = 1.0` if `is_strict_json_format(pred)` with exactly the task schema, else `0.0`.
Reported total in `dataset_grpo/README.md:§3`:

```
R_total = R_task + 0.05 * R_fmt
```

| Outcome | `R_task` | `R_fmt` | `R_total` |
| :--- | :---: | :---: | :---: |
| Correct + valid JSON | 1.0 | 1.0 | 1.05 |
| Correct + malformed (fallback) | 1.0 | 0.0 | 1.00 |
| Wrong + valid JSON | 0.0 | 1.0 | 0.05 |
| Wrong + malformed | 0.0 | 0.0 | 0.00 |

Reference implementation noted in `dataset_grpo/README.md` as
`phase_grpo_dataset/rewards.py:reward_phase/reward_localization/reward_boundary/reward_format/compute_reward`
(archived generator path; current `dataset_grpo` records are self-contained).

> **Contrast with the prior report.** The prior report described `visual_observation` as
> LLM-judge graded (`R_judge(ŷ, y*_reason) ∈ [0,1]`, 78.5% deterministic / 21.5%
> judge). That description matches the **pre-migration** `dataset/` monolith
> (`dataset/**/clip_*_grpo.jsonl`: 1,440 `llm_judge` visual-observation records,
> `Provide your reasoning first, then state your answer.` suffix). The migrated
> `dataset_grpo/` view replaced it with the deterministic JSON MCQ
> (`prompts.py:GRPO_JSON_INSTRUCTION`, `Respond ONLY with a JSON object...`), so the
> active GRPO training set is **100% deterministic** and judge-free.

### 3.6.2 Prompt Construction and Distractor Refinement

Multiple-choice distractors remain plausible while avoiding ambiguous alternatives.
An automated judge is retained **only at curation time** (during distractor generation /
refinement) to identify functionally overlapping or visually confusable choices; ambiguous
distractors are replaced from sufficiently distinct functional/visual categories
(`tools/refine_instrument_options.py`, `tools/generate_missing_grpo_sft.py` handle
`--grpo-dataset-dir` / `--target` for the separated layout). At **training time**, no
judge is required — the reward is the deterministic letter match described above.

All GRPO prompts are generated by the formatter (`stages/dataset_formatter.py:133`) and
share the 5-key schema `prompt / correct_answer / question_type / reference_reasoning /
reward_type`. `tools/merge_dataset.py --out-sft/--out-grpo` routes `*_sft.jsonl` →
`dataset_sft/` and `*_grpo.jsonl` → `dataset_grpo/` (hardlink, fallback copy).

## 3.7 Clinical Phase Ontology

The clinical phase corpus uses the same 13-class cataract ontology across SFT and GRPO
— labels are P01–P13; phase-localization prompts embed the full legend:

```
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

### 3.7.1 SFT Phase Clip Distribution (1,184 Clips; Ground Truth for SFT)

Counted from `dataset_sft/**/PH_*/clip_*.jsonl` (`phase_id`):

| Phase | Canonical Name | Clinical Description | SFT Clips |
| :--- | :--- | :--- | ---: |
| P01 | Incision | Clear corneal incision / paracentesis | 99 |
| P02 | Viscoelastic | OVD injection | 99 |
| P03 | Capsulorhexis | Continuous curvilinear anterior capsulotomy | 97 |
| P04 | Hydrodissection | Fluid wave dissection of cortical-capsular adhesions | 97 |
| P05 | Phacoemulsification | Ultrasonic sculpting/chopping, nuclear removal | 96 |
| P06 | Irrigation-Aspiration | Cortical cleanup / epinucleus removal | 96 |
| P07 | Capsule Polishing | Posterior capsule scraping / polishing | 92 |
| P08 | Lens Implantation | Foldable IOL injector delivery | 87 |
| P09 | Lens Positioning | Dialing / centering IOL in capsular bag | 97 |
| P10 | Viscoelastic Suction | Residual viscoelastic evacuation | 95 |
| P11 | AC Flushing | Hydro-reformation / AC flushing | 36 |
| P12 | Tonifying-Antibiotics | Intracameral antibiotics / stromal hydration | 91 |
| P13 | Idle | Inactive interval | 102 |
| **Total** | | | **1,184** |

This distribution was reproduced in the prior report with different per-phase counts;
the current counts are authoritative for the delivered SFT artifacts.

### 3.7.2 GRPO Phase-Localization Distribution (1,200 Samples; Active GRPO)

From `phase_grpo_dataset/stats.json` (the 1,200-sample generator), remapped by parent
procedure into `dataset_grpo` splits:

| Phase | Samples |
| :--- | ---: |
| P01 | 133 |
| P02 | 206 |
| P03 | 104 |
| P04 | 75 |
| P05 | 79 |
| P06 | 136 |
| P07 | 83 |
| P08 | 90 |
| P09 | 81 |
| P10 | 92 |
| P11 | 20 |
| P12 | 101 |
| P13 | — (Idle serves as window context, never the target) |
| **Total** | **1,200** (= 300 × 4 tasks) |

Task × phase cross-counts (`task_phase_counts` in `stats.json`) range 5–55 per cell;
window modes are `start_boundary` 550 / `interior` 461 / `end_context` 189.

> The prior report's Table 3.7 conflated the SFT clip ontology with the GRPO
> phase-localization counts and listed the now-archived `phase_identification` as the
> phase GRPO contribution.

## 3.8 Dataset Statistics

### 3.8.1 Global Dataset Composition — GRPO View (`dataset_grpo`, Canonical for GRPO)

| Partition | Parent Procedures | Clip `clip_*.mp4` | GRPO-Track `grpo_*.mp4` | Full `full_video.mp4` | `*_grpo.jsonl` Files | GRPO Records | Reward |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Train** | 213 (74.5%) | 2,066 (78.7%) | 828 | 108 (79.4%) | 1,969 (1,141 YT + 828 phase) | 4,252 (3,424 + 828) | 100% deterministic |
| **Validation** | 35 (12.2%) | 265 (10.1%) | 162 | 13 (9.6%) | 299 (137 + 162) | 573 (411 + 162) | 100% deterministic |
| **Test** | 38 (13.3%) | 293 (11.2%) | 210 | 15 (11.0%) | 372 (162 + 210) | 696 (486 + 210) | 100% deterministic |
| **Total** | **286** | **2,624** | **1,200** | **136** | **2,640** | **5,521** | **100% deterministic** |

Procedure breakdown per split (GRPO folders mirror SFT/dataset):

| Partition | YouTube Procedures | Phase Procedures (`PH_*`) | Total Folders |
| :--- | ---: | ---: | ---: |
| Train | 108 | 105 | 213 |
| Validation | 13 | 22 | 35 |
| Test | 15 | 23 | 38 |
| **Total** | **136** | **150** | **286** |

For the phase-localization contribution, only 110 of the 150 phase procedures carry
GRPO-track samples (the generator used 110 stitched videos from
`full_surgical_videos/`). All three views agree on the 286-procedure folder inventory.

The prior report's Table 3.8.1 reported Train 2,066 clips / 4,328 GRPO records and a
total of 5,505 GRPO records — that table mixed active YT counts with archived
`phase_identification` and omitted the 1,200 localization samples, and understated
the GRPO-track `grpo_*.mp4` inventory (reported as 0).

### 3.8.2 Source Breakdown — GRPO View

| Metric | In-the-Wild Surgical Corpus | Clinical Phase Corpus | `dataset_grpo` Combined |
| :--- | ---: | ---: | ---: |
| Parent procedures | 136 | 150 (110 contribute GRPO-track samples) | **286** |
| Video clips (`clip_*.mp4`) | 1,440 | 1,184 | **2,624** |
| GRPO-track videos (`grpo_*.mp4`) | — | 1,200 | **1,200** |
| Full-length videos | 136 | 0 | **136** |
| Video files total in `dataset_grpo` | 1,576 | 2,384 | **3,960** |
| GRPO files | 1,440 `clip_*_grpo.jsonl` | 1,200 `grpo_*_grpo.jsonl` | **2,640** |
| GRPO records | 4,321 (`step` 1,441 / `visual` 1,440 / `instrument` 1,440) | 1,200 (300 × 4 temporal tasks) | **5,521** |
| Question types (active) | `step_identification`, `visual_observation`, `instrument_identification` | `temporal_localization`, `contextual_phase_recognition`, `boundary_detection`, `timestamp_to_phase` | **7 active** (+ 1 archived `phase_identification`) |
| Mean clip duration | 8–15 s (step clips, 10 FPS) | SFT phase clips ~14 s; GRPO phase-local mean 25.8 s (11.1–41.1 s, ≤50 frames, 3.93 FPS mean) | — |
| Reward | 100% deterministic (MCQ letter) | 100% deterministic (exact-match / tIoU / exp-decay + optional `R_fmt`) | **100% deterministic** |

The prior report's §3.8.2 listed mean durations `12.4±4.2 s / 14.1±6.8 s / 13.2±5.6 s`
and total durations `≈24.6 h / ≈4.6 h / ≈29.2 h` without distinguishing
`grpo_*.mp4` from `clip_*.mp4`; those figures should not be cited without
re-derivation from the current video files. The structure above is verified from file
counts; `phase_grpo_dataset/stats.json:clip_duration/fps/num_frames` gives the
phase-local timing.

For the per-split GRPO task split remapped to procedure folders (authoritative for
`dataset_grpo`; cf. `dataset/README.md:GRPO phase-localization track`):

| `question_type` | Train | Validation | Test | Total |
| :--- | ---: | ---: | ---: | ---: |
| `temporal_localization` | 214 | 32 | 54 | 300 |
| `contextual_phase_recognition` | 201 | 39 | 60 | 300 |
| `boundary_detection` | 210 | 41 | 49 | 300 |
| `timestamp_to_phase` | 203 | 50 | 47 | 300 |
| **Total** | **828** | **162** | **210** | **1,200** |

These counts coincide with `dataset/README.md` Table `GRPO phase-localization track`
(now described as archived because that README documents the monolith), but now live
in `dataset_grpo/{split}/PH_*/grpo_*_grpo.jsonl`.

## 3.9 Data Record Structure

The dataset uses three principal record families: **SFT records** (`*_sft.jsonl`),
**YouTube GRPO records** (`clip_*_grpo.jsonl`), and **phase-localization GRPO records**
(`grpo_*_grpo.jsonl`). All share relative `video` paths resolved against the split
folder (e.g. `dataset_grpo/Train/{procedure}/{clip}.mp4`).

### 3.9.1 Clip-Level SFT Records (`dataset_sft/`)

Unchanged in structure; retained for context (GRPO follows SFT).

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "video", "video": "PH_0004_2934_S2/clip_02.mp4"},
      {"type": "text", "text": "Describe the visually observable surgical activity in this cataract surgery video clip. Focus on visible instruments, anatomy, actions, and observable changes. Do not infer details that cannot be seen."}
    ]},
    {"role": "assistant", "content": "A magnified view of an eye held open by metal lid speculums shows a keratome blade approaching and penetrating the cornea at the inferior limbus. ..."}
  ]
}
```

* YouTube `clip_*_sft.jsonl`: 4 lines per file — line 1 pure visual description; lines 2–4
  MCQ with CoT (`Therefore the answer is X) ...`).
* Phase `clip_*_sft.jsonl`: 1 line per file — teacher/critic-approved description.
* Full-video SFT: 1 line per `full_video_sft.jsonl` — timestamped narration
  (`Provide a detailed timestamped narration of this surgical procedure.` →
  `[MM:SS - MM:SS] step: ...`).

### 3.9.2 GRPO Records — YouTube Track (`dataset_grpo/{split}/{YT_ID}/clip_*_grpo.jsonl`)

One record per line, **3 lines per file** (one per question type). Output is **strict JSON**.

```json
{
  "prompt": [{"role": "user", "content": [
    {"type": "video", "video": "-Q4uQ6rEExs/clip_01.mp4"},
    {"type": "text", "text": "What primary surgical step is being demonstrated in this video clip?\nA) Hydrodissection\nB) Main corneal incision\nC) Capsulorhexis\nD) Lens fragmentation\n\nRespond ONLY with a JSON object with two keys: \"explanation\" and \"answer\". \"explanation\" is 1-3 sentences of reasoning describing what is visible (natural language, no field names). \"answer\" is the single letter A, B, C or D (nothing more). Example: {\"explanation\": \"In the video the surgeon uses a keratome to create a corneal incision.\", \"answer\": \"B\"}"}
  ]}],
  "correct_answer": "B",
  "question_type": "step_identification",
  "reference_reasoning": "In the video, the surgeon is seen using a sharp blade to penetrate the clear cornea ...",
  "reward_type": "deterministic"
}
```

Reward (see §3.6.1):

```python
import json
pred = json.loads(model_output)  # must be {"explanation": str, "answer": "A|B|C|D"}
reward = 1.0 if pred["answer"] == gold["correct_answer"] and set(pred) == {"explanation", "answer"} else 0.0
```

> The prior report's example GRPO record omitted the JSON constraint and implied
> free-text answer extraction; current records require JSON parsing. The monolith's
> `Provide your reasoning first, then state your answer.` suffix is replaced by
> `prompts.py:GRPO_JSON_INSTRUCTION`.

### 3.9.3 GRPO Records — Phase-Localization Track (`dataset_grpo/{split}/PH_*/grpo_*_grpo.jsonl`)

One record per file (1,200 total), one video per file (10–41 s). Prompt embeds the
13-phase legend and a task-specific `answer` schema.

```json
{
  "prompt": [{"role": "user", "content": [
    {"type": "video", "video": "PH_0004_2934_S2/grpo_0384_temporal_localization.mp4"},
    {"type": "text", "text": "Locate the Capsule Polishing phase that follows the inactive interval in this clip. Give its start and end timestamps.\n\nThe available surgical phases are:\nP01 - Incision: Initial corneal incision\n...P13 - Idle: No active surgical action or inactive time\n\nRespond with a single JSON object with exactly two keys: \"explanation\" and \"answer\".\n\"explanation\" must be a brief reasoning in 1-3 sentences. \"answer\" is task-specific and described below.\n\nExample output:\n{\n  \"explanation\": \"Brief reasoning in 1-3 sentences based on the video.\",\n  \"answer\": {\n    \"start\": 5.0,\n    \"end\": 18.0\n  }\n}\n\nOutput ONLY the JSON object, no extra text, no markdown fences."}
  ]}],
  "correct_answer": {"start": 9.4, "end": 13.8},
  "question_type": "temporal_localization",
  "reference_reasoning": "Capsule Polishing (P07) occupies clip-local 9.4-13.8 s in this clip.",
  "reward_type": "deterministic"
}
```

Other `answer` shapes in this track:

* `boundary_detection` → `{"timestamp": 7.7}`
* `contextual_phase_recognition` / `timestamp_to_phase` → `{"phase_id": "P07", "phase_name": "Capsule Polishing"}` (aliases normalized)

Materialization note: `prompt[0].content[0].video` is split-relative
(`PH_*/grpo_*.mp4`); resolve against `dataset_grpo/{split}/`. No `metadata` block is
stored in `dataset_grpo` (full provenance — `source` global seconds, `target`
global/clip-local intervals, `window.mode`, `frames{fps,num_frames,frame_times_clip}`,
`grpo_split_ignored` — is retained only in the generator output at
`phase_grpo_dataset/annotations.jsonl` and `dataset/README.md` §4).

### 3.9.4 Full-Procedure SFT Records

Unchanged; mirrored into both `dataset_sft/` and `dataset_grpo/` as
`{YT_ID}/full_video_sft.jsonl`:

```
[00:07 - 00:15] viscoelastic injection: A cannula is introduced ...
[00:16 - 00:45] capsulorhexis: Capsulorhexis forceps create a circular opening ...
[00:46 - 01:12] hydrodissection: Fluid is injected beneath the capsule ...
[01:13 - 03:40] phacoemulsification: The phaco probe emulsifies and aspirates ...
```

## 3.10 Summary of the Dataset Design

The verified dataset provides four complementary properties supporting surgical VLM
development:

1. **Hierarchical temporal supervision.** Localized actions (seconds) and complete
   procedure narratives (minutes), with the GRPO view adding explicit temporal
   localization/boundary tasks over stitched full procedures.
2. **Complementary data sources.** In-the-wild recordings (diversity + narration) and the
   annotated phase corpus (controlled 13-phase ontology, teacher/critic QC).
3. **Fully verifiable RL objectives.** `dataset_grpo` is **100% deterministic** — MCQ
   letter exact-match for the YouTube track and exact-match / tIoU / exponential-decay
   (+ optional format) rewards for the phase-localization track. No LLM judge at
   training time; distractor curation is the only judge use.
4. **Procedure-level data integrity.** Split assignment at the parent-procedure level
   across all views, including remapped phase-localization samples; archived
   `phase_identification` and `full_video_grpo` are retained for provenance but excluded
   from active GRPO.

Together, these design choices establish a unified dataset for supervised multimodal
learning and verifiable GRPO alignment in cataract surgery — with `dataset_sft` for SFT
and `dataset_grpo` (5,521 deterministic records) for GRPO.

## 3.11 Training and Evaluation

The construction above is the foundation for the subsequent training and evaluation
procedures, described separately: base VLM, parameter-efficient fine-tuning, SFT stage
(`dataset_sft`), GRPO optimization (`dataset_grpo` with the rewards in §3.6.1), and
training configuration; followed by the held-out test protocol, baselines, tasks, and
quantitative metrics.

---

## Appendix — Verification Notes and Corrections vs. `academic_report.md`

The following discrepancies were confirmed by sampling `dataset_grpo/`,
`dataset_sft/`, `dataset/`, `phase_grpo_dataset/`, `prompts.py`, and `config.yaml`.

| Prior Report Claim | Verified Status / Correction | Evidence |
| :--- | :--- | :--- |
| §3.1: "2,624 clips from 286 procedures incl. 136 full procedures; 2,760 SFT samples; 5,505 GRPO samples" | **Partially stale.** 2,624 `clip_*.mp4` + 136 `full_video.mp4` describes the monolith/`dataset_sft` clip inventory only. GRPO view has **3,824 video clips** (2,624 `clip_*.mp4` + 1,200 `grpo_*.mp4`). SFT "2,760 samples" conflates **files** with **lines** — true SFT lines are 6,944 + 136 = 7,080 (see §3.5). GRPO "5,505" = 4,321 (YouTube) + 1,184 (archived `phase_identification`); **active GRPO** is **5,521 = 4,321 + 1,200** (YouTube + phase-localization). | `dataset_grpo/README.md:30-42`, `dataset_sft/README.md`, `dataset/README.md:§Statistics` |
| §3.2: "Mean duration 13.2±5.6 s at 10 FPS; 4 question types incl. phase_identification" | **Conflated.** 10 FPS applies to YT downsampling; phase-localization GRPO clips are ≤50 frames, adaptive 2.4–5.0 FPS, mean 25.8 s (§3.2.1). Active GRPO has **7 question types** (3 YT + 4 phase-local); `phase_identification` is archived, not active. | `config.yaml:downsample`, `phase_grpo_dataset/stats.json:clip_duration/fps/num_frames` |
| §3.6 Table: `phase_identification` 1,184 as the phase GRPO contribution | **Superseded.** Phase GRPO is the **4-task localization suite** (300 each, 1,200 total; §3.6.0). `phase_identification` is archived at `dataset/archived/.../clip_*_grpo.jsonl` (1,184) and excluded from `dataset_grpo`. | `dataset_grpo/**/grpo_*_grpo.jsonl` counts; `dataset/README.md:§Archived Data`; `phase_grpo_dataset/stats.json` |
| §3.6: "GRPO: 5,505 records; 78.5% deterministic / 21.5% LLM judge (`R_judge`)" | **Incorrect for the training set.** `dataset_grpo` is **100% deterministic**, `R_judge` not used. `grep -r llm_judge dataset_grpo` → 0; `dataset/**/clip_*_grpo.jsonl` → 1,440 `llm_judge` matches the pre-migration monolith (visual_observation). | `dataset_grpo/README.md:41`, `prompts.py:GRPO_JSON_INSTRUCTION`, sampling 5,521 records |
| §3.6 reward: `R_det = 1(ExtractAnswer(ŷ)=y*)` free-text, and `R_judge ∈ [0,1]` | **Format changed.** All records require strict JSON `{"explanation","answer"}`; rewards are letter exact-match, phase normalized exact-match, `exp(-|err|/1.5)`, tIoU, plus optional `R_fmt` (§3.6.1). | `prompts.py:145-150`, `dataset_grpo/README.md:§Reward Functions` |
| §3.7 Phase ontology counts (P01 112 ... P13 52, total 1,184) | **Distribution updated.** Current SFT clip distribution is P01 99 / P02 99 / P03 97 / P04 97 / P05 96 / P06 96 / P07 92 / P08 87 / P09 97 / P10 95 / P11 36 / P12 91 / P13 102 (§3.7.1). GRPO phase-local distribution is separate (1,200, §3.7.2). | Counts from `dataset_sft/**/PH_*/clip_*.jsonl`; `phase_grpo_dataset/stats.json:phase_counts` |
| §3.8.1 Global composition: Train GRPO 4,328 / Val 557 / Test 620 | **Updated to remapped phase-local splits.** `dataset_grpo` actual: Train 4,252 (3,424+828) / Val 573 (411+162) / Test 696 (486+210) (§3.8.1). Prior table omitted `grpo_*.mp4` as a separate video class. | `dataset_grpo/**/ *_grpo.jsonl` per-split counts (§3.8.1) |
| §3.8.2 Source breakdown: GRPO 4,321 + 1,184 = 5,505 | **Updated.** `dataset_grpo` GRPO = 4,321 + 1,200 = **5,521**; source tables now distinguish `clip_*.mp4` vs `grpo_*.mp4` (§3.8.2). | `dataset_grpo/README.md:37-40` |
| §3.9 GRPO record example (free-text MCQ, `Provide your reasoning...`) | **Prompt suffix replaced.** `dataset_grpo` uses `Respond ONLY with a JSON object...` with typed `answer` (§3.9.2/3.9.3). | `prompts.py:GRPO_JSON_INSTRUCTION`, sampled `dataset_grpo` prompts |
| §3.9.3 Full-procedure phrasing | Unchanged, retained verbatim. | `dataset_sft/**/full_video_sft.jsonl` |

### Integrity Checks Performed

* Counted `*_grpo.jsonl` lines per `question_type` and `reward_type` across
  `dataset/`, `dataset_sft/`, `dataset_grpo/`, `phase_grpo_dataset/`.
* Sampled prompts for the JSON instruction vs legacy suffix.
* Compared `phase_grpo_dataset/stats.json` (`task_counts`, `phase_counts`,
  `clip_duration`, `fps`, `mode_counts`) and `splits.json` vs remapped
  `dataset_grpo/{split}/PH_*/` distribution.
* Verified folder-level split parity across the three dataset views.
* Read `prompts.py`, `config.yaml`, `dataset/README.md`, `dataset_sft/README.md`,
  `dataset_grpo/README.md`, `phase_grpo_dataset/README.md`.

### How to Re-Verify

```bash
# GRPO inventory (canonical)
grep -r '"question_type"' dataset_grpo --include="*.jsonl" | cut -d: -f2 | sort | uniq -c
grep -r llm_judge dataset_grpo --include="*.jsonl" | wc -l   # → 0
grep -r "Respond ONLY with a JSON" dataset_grpo --include="*.jsonl" | wc -l  # → 4321 (YT) + phase prompts
wc -l dataset_grpo/Train/*/*_grpo.jsonl dataset_grpo/Validation/*/*_grpo.jsonl dataset_grpo/Test/*/*_grpo.jsonl

# SFT phase ontology
python3 -c "import glob,json,collections; c=collections.Counter(); [c.update([json.load(open(f))['phase_id']]) for f in glob.glob('dataset_sft/**/PH_*/clip_*.jsonl',recursive=True)]; print(c)"

# Phase-local stats
cat phase_grpo_dataset/stats.json | python3 -m json.tool | head -80
```

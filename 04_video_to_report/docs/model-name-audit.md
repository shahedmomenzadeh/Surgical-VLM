# Model-name audit — `qwen3vl-4b-instruct` vs `qwen3vl-2b-grpo`

Date: 2026-10-28
Scope: the final 25-video `v21full` evaluation, the working repo, the HF dataset
repo, and the manuscript folder `/mnt/d/VLM paper/video-to-report/`.

## 1. Why this audit exists

The requested correction is:

> where the results say `qwen3vl-4b-instruct`, they must say `qwen3vl-2b-grpo`.

Both tags currently exist as **complete, independent chains**. A one-way rename
would collide with the existing `qwen3vl-2b-grpo` chain, so the intended mapping
must be confirmed before any bulk rename.

## 2. Inventory — every place the names occur

### 2.1 Working repo (local)

| Area | `qwen3vl-4b-instruct` | `qwen3vl-2b-grpo` |
|---|---:|---:|
| `output-candidates/<tag>/` | 3,496 files / 16 MB | 3,496 files / 14 MB |
| `output-evaluation/v21full/judgments/` | 50 files | 50 files |
| `output-evaluation/v21full/judge_video_scores.jsonl` | 50 rows | 50 rows |
| `output-evaluation/v21full/video_scores.jsonl` | 50 rows | 50 rows |
| `output-evaluation/v21full/{comparisons.json,summary.md,README.md}` | model keys / rows | model keys / rows |
| `output-evaluation/full25/` (legacy run) | 150 files | 150 files |
| `output-evaluation/v2slim/` (legacy pilot) | 1 file | 11 files |
| `output-readable/` | 25 files | 78 files |
| `kaggle_run/` | 3 files | 4 files |
| `src/` | 2 files | 5 files |
| `docs/` + root `README.md` | 2 files | 2 files |
| `archives/` (historical) | 2 paths | 5 paths |

The 50 judgment files per tag are the 25 videos × 2 settings. Each judgment
file also stores the tag inside every JSONL row.

The candidate chain per tag is complete: 25 video masters + 25 memory files +
1,698 chunk files, per setting, for both `s1` and `s3`.

### 2.2 Hugging Face dataset repo

`shahedm2001/cataract-video-to-report` (27,530 files total):

| Path | `qwen3vl-4b-instruct` | `qwen3vl-2b-grpo` |
|---|---:|---:|
| `output-candidates/<tag>/` | 3,496 files | 3,496 files |
| `output-evaluation/v21full/judgments/*<tag>*` | 50 files | 50 files |

The HF dataset card (`README.md`) and the leaderboard/tables embedded in it also
name both models.

### 2.3 Manuscript folder

`/mnt/d/VLM paper/video-to-report/`:

- `README.md` — headline, leaderboard, contrasts
- `RESULTS_AND_PERFORMANCE_ANALYSIS.md` — 8 mentions; headline, table rows,
  bootstrap contrasts, per-model discussion (§5.1), qualitative examples
- `data/comparisons.json`, `data/summary.md`
- `data/video_scores.jsonl`, `data/judge_video_scores.jsonl` — 50 rows each
- `tables/tab_video_to_report_leaderboard.tex` — display name
  `Qwen3-VL-4B-Instruct`, rank 1
- `tables/tab_video_to_report_contrasts.tex` — display name
  `Qwen3-VL-4B-Instruct`
- `tables/tables_summary.md`
- `generate_paper_tables.py` — display-name map

Also containing the model names but **from the older 9-model benchmark**, not
the current 25-video run:

- `/mnt/d/VLM paper/evaluation-results/` — separate evaluation with its own
  4B-Instruct and GRPO rows (Sept 2026 data)
- `/mnt/d/VLM paper/generate_evaluation_tables.py`,
  `/mnt/d/VLM paper/generate_interactive_plots.py`,
  `/mnt/d/VLM paper/evaluation-results/tables/*.tex`
- `/mnt/d/VLM paper/presentation/index.html` — contains the **current**
  25-video leaderboard (4B rank 1, GRPO rank 6) plus the older benchmark.

## 3. The conflict

A one-way rename `qwen3vl-4b-instruct` → `qwen3vl-2b-grpo` cannot be applied
to the current data without deciding what happens to the existing GRPO chain.
Both chains already cover all 25 videos, both settings, all 849 chunks.

Pre-correction `v21full` leaderboard (Tier J, s3):

| Rank | Model | Accuracy | Coverage | Material error |
|---:|---|---:|---:|---:|
| 1 | `qwen3vl-4b-instruct.s3` | 1.625 | 1.708 | 43.6% |
| 2 | `hulumed-4b.s3` | 1.616 | 1.553 | 48.5% |
| 3 | `hulumed-7b.s3` | 1.555 | 1.557 | 52.2% |
| 4 | `lingshu-7b.s3` | 1.331 | 1.400 | 65.0% |
| 5 | `qwen3vl-2b-sft-stage2.s3` | 1.308 | 1.445 | 69.8% |
| 6 | `qwen3vl-2b-grpo.s3` | 1.205 | 1.351 | 76.3% |
| 7 | `qwen3vl-2b-instruct.s3` | 1.087 | 1.214 | 88.9% |

So the choice changes the paper's headline:

- **swap** → GRPO becomes rank 1 (1.625); 4B drops to rank 6 (1.205)
- **rename + archive old GRPO** → GRPO rank 1; 4B row disappears (6 models)
- **drop 4B, keep old GRPO** → Hulu-Med-4B becomes rank 1 (1.616); GRPO stays 1.205
- **text-only change** → raw results and leaderboard stay as they are

## 4. What the data says about the physical model behind each chain

The Kaggle zips are still in `/mnt/c/Users/shahed/Downloads/` and were inspected.
Both archives contain only `output-candidates/<tag>/` paths — no model ID, no
run log, no config. The zip filenames and internal timestamps are consistent
with two concurrent runs started ~9 minutes apart, but they cannot prove which
checkpoint each notebook actually loaded.

Behavioural fingerprints of the two chains (all 25 videos, both settings):

| Signal | 4B-tagged chain | GRPO-tagged chain | SFT-stage2 | Base 2B |
|---|---:|---:|---:|---:|
| Median latency, s1 / s3 | 16.0 / 18.0 s | 9.7 / 9.7 s | 9.1 / 10.3 s | 9.8 / 10.3 s |
| Median report length | 596 chars | 429 chars | 418 chars | 461 chars |
| Generic/hallucinated anatomy terms | 11.5% | 1.2% | 5.4% | 0.1% |
| Similarity to archived local GRPO pilot | 0.323 | 0.488 | — | — |
| s1 stated-phase accuracy | 5.1% | 12.2% | 17.0% | 6.7% |
| Zip internal run window | 14:39–22:47 | 14:30–19:16 | — | — |

Interpretation: the `qwen3vl-4b-instruct` chain behaves like a larger **general**
model (1.6–1.8× slower, 39% longer, far more generic anatomy terms, less
teacher-like). The `qwen3vl-2b-grpo` chain behaves like the 2B fine-tune and
matches the archived local GRPO pilot much more closely. On the evidence, the
tags are already attached to the right physical models.

This is why the rename cannot be done silently: the raw data does not support
relabelling the verbose chain as GRPO. If the Kaggle notebook for the 4B slot
was launched with the GRPO checkpoint, the notebook log is the authority — but
the log is not in the repo or the zips. Please confirm the mapping.

## 5. Options and exact consequences

### Option A — swap the two chains (recommended if the two zips were mixed up)

- `output-candidates/qwen3vl-4b-instruct/` → `.../qwen3vl-2b-grpo/`
- `output-candidates/qwen3vl-2b-grpo/` → `.../qwen3vl-4b-instruct/`
- Swap the tag strings inside every candidate/readable/judgment/score file,
  swap the two directories, swap the rows in all tables and prose.
- Result: 7 models preserved; GRPO becomes rank 1 (1.625), 4B rank 6 (1.205).
- Paper narrative: “our 2B GRPO model beats 4B/7B general and medical models.”
- Fully reversible: keep a manifest of old→new paths and the original zips.

### Option B — rename 4B → GRPO, archive the existing GRPO chain

- Rename the 4B chain to `qwen3vl-2b-grpo`.
- Move the current GRPO chain to
  `archives/qwen3vl-2b-grpo-run1-2026-10-18/` (with a note and checksums).
- Result: 6 models in the final run; GRPO rank 1 (1.625); no 4B row.
- Paper narrative: GRPO wins; 4B was not part of the final lineup.
- The archived chain remains available as a second GRPO run if needed.

### Option C — drop 4B, keep the existing GRPO chain

- Use the current GRPO chain as-is; remove the 4B chain from all final results
  and discussion (archive it, do not delete).
- Result: 6 models; Hulu-Med-4B becomes rank 1 (1.616); GRPO stays rank 6
  (1.205).
- Paper narrative: medical specialists lead; the 4B general model was an
  accidental extra run and is excluded.

### Option D — text-only correction

- Keep raw folders and scores unchanged; only change the display labels in
  prose/tables. This is not recommended because the numbers would no longer
  identify the chain that produced them.

## 6. Files that will be touched under A/B/C

1. Candidate folders + every `"model"` field and path string.
2. `output-evaluation/v21full/judgments/` filenames and JSONL tag fields;
   regenerate `judge_video_scores.jsonl`, `video_scores.jsonl`,
   `comparisons.json`, `summary.md`, `README.md`.
3. `output-readable/` folders and headers.
4. Legacy `output-evaluation/full25/`, `v2slim/`, `archives/` if in scope.
5. `src/` tag lists (`upload_hf_dataset.py`, `build_hf_dataset.py`,
   `evaluation/viewer.py`, docstrings/examples).
6. `kaggle_run/` scripts, `runs.txt`, `README.md` (note: folder is gitignored
   but still on disk).
7. Root `README.md`, `docs/methods-report.md`, `judgement_overview.md` if named.
8. `/mnt/d/VLM paper/video-to-report/` — all 8 files, both `.tex` tables,
   regenerated `tables_summary.md`, `generate_paper_tables.py`.
9. `/mnt/d/VLM paper/presentation/index.html` — current leaderboard rows.
10. HF dataset repo — re-upload the changed candidate/judgment/score files and
    regenerate the dataset card.
11. Older `/mnt/d/VLM paper/evaluation-results/` and its tables/presentation
    only if the same Kaggle mistake affected the older 9-model benchmark.

## 8. Resolution (2026-10-28)

Option **A (swap)** was confirmed and applied, scoped to `v21full` and the
current manuscript/presentation. The swap was performed with placeholders and
verified against a pre-swap snapshot:

- 9,977 affected files compared canonically: **0 mismatches, 0 missing**.
- `comparisons.json` panels and paired contrasts swapped exactly.
- Backup and restore commands: `archives/model-name-swap-2026-10-28/`.

Corrected leaderboard (Tier J, s3): `qwen3vl-2b-grpo.s3` is rank 1
(Acc 1.625, Cov 1.708, material 43.6%); `qwen3vl-4b-instruct.s3` is rank 6
(1.205, 1.351, 76.3%). The 4B generalist retains the highest lexical overlap
(ROUGE-L 0.239), which is now explicitly discussed as a lexical-metric
deception rather than evidence of clinical quality.

The quarantined `judgments-blind-visual1/` rows were included in the swap for
local consistency but remain invalid for visual claims; see their `NOTE.md`.
The Kaggle run scripts were **not** swapped — they correctly load
`Qwen/Qwen3-VL-4B-Instruct` and the GRPO checkpoint for future runs.

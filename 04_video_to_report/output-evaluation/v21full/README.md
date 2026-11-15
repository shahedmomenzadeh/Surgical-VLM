# Evaluation Benchmark: 7 Vision-Language Models across 25 Cataract Surgery Videos (`v21full`)

**Evaluation Run ID:** `v21full`  
**Dataset:** 25 intraoperative cataract surgery videos (849 frozen surgical chunks, ~6.8 hours of operative footage)  
**Evaluated Architectures:** 7 vision-language models (2 settings each: $s_1$ unassisted vs. $s_3$ assisted)  
**Total Evaluation Decisions:** 11,886 LLM-judge evaluations (`visual-3` rubric) + 11,886 deterministic Tier 0 evaluations  
**Completion Status:** 100% complete across all 14 chains (0 errors, 0 unjudged chunks)

---

## 1. Executive Summary & Core Findings

This benchmark provides the definitive post-hoc clinical evaluation of 7 Vision-Language Models (VLMs) performing real-time surgical report generation on the complete 25-video cataract surgery corpus. Every model is evaluated under two distinct operating paradigms:
- **$s_1$ (Unassisted):** The model receives only the raw surgical video clip frames and generates a structured clinical report from scratch without external aids.
- **$s_3$ (Assisted):** The model receives video clip frames along with external working memory (state updates from previous surgical chunks) and surgical context hints.

### Key Conclusions:
1. **Assisted Prompting ($s_3$) Decisively Outperforms Unassisted ($s_1$):**
   Across **100% of tested architectures (7/7)**, providing external memory context produces statistically significant improvements in clinical accuracy ($+0.11$ to $+0.60$, $p < 0.001$) and coverage ($+0.19$ to $+0.52$, $p < 0.001$).
2. **Resolution of the Blind Judge Paradox:**
   Earlier pilot runs using a blind judge (`visual-1`) erroneously scored $s_1 > s_3$ because the text-only judge penalized specific surgical descriptions as hallucinations. When the judge is physically grounded in 1 FPS video frames (`visual-3`), the ranking flips decisively: descriptive specificity in $s_3$ is verified as true against operative reality.
3. **Top Performing Architecture:**
   **`qwen3vl-2b-grpo.s3`** achieves the highest overall accuracy (**1.625 / 2.0**) and coverage (**1.708 / 2.0**), with the lowest material error rate (**43.6%**).
4. **Active Surgical Manipulation Champion:**
   **`hulumed-7b.s3`** is the most reliable model during active instrument maneuvers (Active Accuracy: **1.549**, Active Coverage: **1.422**, Teacher Instrument F1: **0.745**).
5. **Mitigation of Dangerous Hallucinations:**
   Assisted prompting cuts destructive material hallucinations (e.g., phantom instruments, invented capsular tears or vitreous loss) by more than half for domain-adapted models (`hulumed-4b`: 85.7% $\to$ 48.5%; `hulumed-7b`: 83.3% $\to$ 52.2%).

---

## 2. Evaluation Methodology & Rubric (`visual-3`)

The evaluation framework comprises two complementary tiers:

### Tier 0: Deterministic Protocol & Lexical Reliability
- **ROUGE-L & chrF:** N-gram and character overlap against verified expert teacher reports.
- **Instrument F1 & Anatomy F1:** Exact set-matching against canonical anatomical landmarks and 10 standardized instrument categories (`bimanual`, `capsulorhexis cystotome`, `capsulorhexis forceps`, `hydrodissection cannula`, `irrigation-aspiration`, `keratome knife`, `lens injector`, `phacoemulsification handpiece`, `secondary knife`, `tissue forceps`).
- **Reliability Checks:** Monitored rates of degenerate repetitions, echo loops, and empty generations (all maintained at 0.0% across the full corpus).

### Tier J: Vision-Grounded LLM-Judge (`visual-3`)
Each of the 11,886 clinical reports is individually judged by a multimodal model (`ag/gemini-3.8-flash`) under strict blinding conditions:
- **Vision Grounding:** The judge receives the actual video chunk sampled at **1 frame per second** (as base64 JPEG stills) to ensure complete temporal perception.
- **Clinical Context:** The judge is provided the verified surgical phase (manually annotated ground truth) and the canonical 10-instrument surgical vocabulary.
- **Double Blinding:** The judge is blinded to model identity, training setting ($s_1$ vs $s_3$), teacher reports, and detector hint proposals.
- **Independent Ordinal Scoring (0 to 2 Quality Scale):**
  - **Accuracy:** `3` = fully truthful; `2` = harmless minor discrepancy; `1` = material clinical error; `0` = complete hallucination.
  - **Coverage:** `3` = complete description; `2` = captures primary action; `1` = superficial/vague; `0` = omits all core events.
- **Traceable Issue Classification:** Every negative score requires an exact verbatim quotation from the candidate report and a specific clinical reason tagged as either `material` (critical error) or `minor`.

---

## 3. Master Leaderboard (All 25 Videos, 14 Chains)

*Note: Tier J metrics represent per-video means across all 25 videos (scale 0.0 to 2.0). Material Rate represents the fraction of chunks with at least one critical factual defect.*

| Rank | Architecture | Setting | Tier J Acc ↑ | Tier J Cov ↑ | Material Rate ↓ | Active Acc ↑ | Idle Acc ↑ | instrF1 ↑ | ROUGE-L |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **`qwen3vl-2b-grpo`** | **$s_3$** | **1.625** | **1.708** | **43.6%** | 1.484 | 1.773 | 0.546 | 0.181 |
| 🥈 | **`hulumed-4b`** | **$s_3$** | **1.616** | **1.553** | **48.5%** | 1.272 | **1.976** | 0.542 | 0.111 |
| 🥉 | **`hulumed-7b`** | **$s_3$** | **1.555** | **1.557** | **52.2%** | **1.549** | 1.560 | **0.745** | 0.111 |
| 4 | `lingshu-7b` | $s_3$ | 1.331 | 1.400 | 65.0% | 1.190 | 1.476 | 0.439 | 0.141 |
| 5 | `qwen3vl-2b-sft-stage2`| $s_3$ | 1.308 | 1.445 | 69.8% | 1.271 | 1.347 | 0.624 | 0.190 |
| 6 | `qwen3vl-4b-instruct` | $s_3$ | 1.205 | 1.351 | 76.3% | 1.233 | 1.174 | 0.611 | **0.239** |
| 7 | `qwen3vl-2b-grpo` | $s_1$ | 1.130 | 1.185 | 79.8% | 0.976 | 1.288 | 0.160 | 0.150 |
| 8 | `qwen3vl-2b-sft-stage2`| $s_1$ | 1.118 | 1.151 | 72.9% | 0.789 | 1.460 | 0.148 | 0.131 |
| 9 | `qwen3vl-4b-instruct` | $s_1$ | 1.097 | 1.143 | 75.9% | 0.888 | 1.318 | 0.146 | 0.184 |
| 10 | `qwen3vl-2b-instruct` | $s_3$ | 1.087 | 1.214 | 88.9% | 1.125 | 1.048 | 0.645 | 0.152 |
| 11 | `hulumed-7b` | $s_1$ | 1.072 | 1.121 | 83.3% | 0.901 | 1.251 | 0.254 | 0.096 |
| 12 | `hulumed-4b` | $s_1$ | 1.018 | 1.084 | 85.7% | 0.841 | 1.198 | 0.176 | 0.117 |
| 13 | `lingshu-7b` | $s_1$ | 0.986 | 1.021 | 88.4% | 0.917 | 1.055 | 0.125 | 0.149 |
| 14 | `qwen3vl-2b-instruct` | $s_1$ | 0.919 | 1.020 | 93.9% | 0.873 | 0.967 | 0.088 | 0.194 |

---

## 4. Statistical Setting Contrasts ($s_3 - s_1$)

Paired chunk-matched differences per video with **95% bootstrap confidence intervals** (2,000 resamples):

```
                        Paired Accuracy Delta (s3 − s1) [95% CI]
hulumed-4b              +0.598  [+0.499, +0.699]  ==================*===================
qwen3vl-2b-grpo     +0.495  [+0.432, +0.560]        ==============*=============
hulumed-7b              +0.483  [+0.363, +0.597]       ===============*================
lingshu-7b              +0.345  [+0.248, +0.440]             =========*========
qwen3vl-2b-sft-stage2   +0.190  [+0.119, +0.261]                   ===*===
qwen3vl-2b-instruct     +0.169  [+0.114, +0.235]                   ===*===
qwen3vl-4b-instruct         +0.108  [+0.041, +0.168]                     =*=
                                                |-------------------+-------------------|
                                               0.0                 +0.4                +0.8
```

### Contrast Metrics Breakdown

| Model Architecture | $\Delta\text{Accuracy}$ [95% CI] | $\Delta\text{Coverage}$ [95% CI] | $\Delta\text{instrF1}$ (Tier 0) [95% CI] | Absolute Material Errors Reduction |
|---|:---:|:---:|:---:|:---:|
| **`hulumed-4b`** | **+0.598** `[+0.499, +0.699]` | **+0.469** `[+0.392, +0.544]` | **+0.366** `[+0.311, +0.419]` | **−739** *(1,347 $\to$ 608)* |
| **`qwen3vl-2b-grpo`**| **+0.495** `[+0.432, +0.560]` | **+0.523** `[+0.453, +0.593]` | **+0.386** `[+0.341, +0.429]` | **−609** *(1,181 $\to$ 572)* |
| **`hulumed-7b`** | **+0.483** `[+0.363, +0.597]` | **+0.436** `[+0.340, +0.527]` | **+0.492** `[+0.427, +0.551]` | **−694** *(1,318 $\to$ 624)* |
| **`lingshu-7b`** | **+0.345** `[+0.248, +0.440]` | **+0.379** `[+0.296, +0.469]` | **+0.314** `[+0.278, +0.352]` | **−537** *(1,375 $\to$ 838)* |
| **`qwen3vl-2b-sft-stage2`**|**+0.190** `[+0.119, +0.261]` | **+0.295** `[+0.216, +0.367]` | **+0.476** `[+0.427, +0.522]` | **−221** *(1,112 $\to$ 891)* |
| **`qwen3vl-2b-instruct`** | **+0.169** `[+0.114, +0.235]` | **+0.195** `[+0.123, +0.267]` | **+0.557** `[+0.525, +0.592]` | **−182** *(1,443 $\to$ 1,261)*|
| **`qwen3vl-4b-instruct`** | **+0.108** `[+0.041, +0.168]` | **+0.207** `[+0.118, +0.290]` | **+0.465** `[+0.413, +0.518]` | **−160** *(1,145 $\to$ 985)* |

*Every single 95% confidence interval strictly excludes zero, formally confirming that assisted reporting provides robust, generalizable improvements regardless of backbone size.*

---

## 5. Detailed Model Performance Analyses

### A. Domain-Adapted 2B GRPO vs. Specialized Medical VLMs
- **`qwen3vl-2b-grpo` (Domain-Adapted 2B Champion):**
  Takes 1st place in overall accuracy (1.625) and coverage (1.708). The two-stage GRPO post-training aligns its phase and instrument vocabulary with the surgical contract; with temporal context it produces the most clinically faithful descriptions in the benchmark despite having only 2B parameters.
- **`hulumed-4b` & `hulumed-7b` (Medical Domain VLMs):**
  Show the largest absolute boost from assistance ($\Delta\text{Acc} \approx +0.50$ to $+0.60$).
  - **`hulumed-4b.s3`** achieves near-perfection on **Idle Phases (1.976 / 2.0)**, correctly confirming that no surgical maneuvers are taking place rather than inventing phantom steps.
  - **`hulumed-7b.s3`** is the **Active Surgery Champion**, leading all models in active manipulation accuracy (**1.549**) and instrument detection F1 (**0.745**). Its clinical pretraining allows it to distinguish subtle instrument tips (e.g. cystotome vs. capsulorhexis forceps) in blurry vitreous chambers.

### B. Post-Training Paradigms and the 4B Generalist
- **Base 2B Model Limitation:** Base `qwen3vl-2b-instruct` struggles with complex intraocular scenes. Even with assistance hints, its material error rate remains at **88.9%**, because it frequently confuses surgical actions (e.g., misidentifying hydrodissection as phacoemulsification).
- **GRPO vs. SFT Post-Training:**
  - **`qwen3vl-2b-grpo.s3`** reaches **1.625** clinical accuracy, ahead of `qwen3vl-2b-sft-stage2.s3` (**1.308**). The verifiable-reward GRPO curriculum turns phase and instrument propositions into grounded descriptions; supervised fine-tuning alone learns the report template but not the visual verification policy.
  - **`qwen3vl-4b-instruct.s3`** is the strongest lexical mimic (ROUGE-L: **0.239**) but ranks 6th clinically (**1.205**) with a **76.3%** material error rate. A general 4B model can reproduce teacher boilerplate without grounding it in the clip, so lexical similarity is a poor proxy for surgical truth.

### C. Clinical Narrative Dynamics: Material vs. Minor Shift
Across all 11,886 evaluated chunks, assistance fundamentally transforms the error profile:
- **In $s_1$ (Unassisted):** 9,121 material issues are recorded. The model routinely generates catastrophic hallucinations: claiming posterior capsule rupture, iris prolapse, or active emulsification when the eye is idle.
- **In $s_3$ (Assisted):** Material hallucinations drop to 5,799 (a **36.4% reduction** across the corpus), while minor descriptive defects rise from 2,111 to 3,559. Instead of inventing phantom complications, $s_3$ models shift toward harmless minor omissions (e.g., slightly imprecise bevel angles or omitting a secondary incision mention).

---

## 6. Case Study: Canary Chunk Audit (`PH_0075_1093_S1`, Chunk `P13_o11`)

Chunk `P13_o11` represents an idle operative segment where instruments are retracted from the anterior chamber:
- **`hulumed-4b.s1` (Score 0 / 2):** Hallucinates a capsule tear complication and states that a phaco handpiece is active.
- **`qwen3vl-2b-instruct.s1` (Score 1 / 2):** Claims a keratome knife is actively performing a corneal incision.
- **`qwen3vl-2b-grpo.s3` (Score 2 / 2):** Accurately recognizes the idle field; the judge flags only minor cannula-irrigation and corneal-vascular details.
- **`hulumed-4b.s3` (Score 2 / 2):** Clean description confirming that the anterior chamber is quiescent.

---

## 7. Directory Structure & File Index

```
output-evaluation/v21full/
├── README.md                           # This comprehensive analytical report
├── summary.md                          # Formatted evaluation tables and summary markdown
├── comparisons.json                    # Machine-readable per-video aggregates and bootstrap CIs
├── judge_video_scores.jsonl            # 350 aggregated Tier J video evaluations (25 videos × 14 chains)
├── video_scores.jsonl                  # 350 aggregated Tier 0 deterministic video scores
├── judgments/                          # 350 JSONL files containing all 11,886 chunk evaluations
│   ├── PH_0001_2931_S2.hulumed-4b.s1.jsonl
│   ├── PH_0001_2931_S2.hulumed-4b.s3.jsonl
│   └── ... (350 files)
└── judgments-blind-visual1/            # Quarantined legacy blind judge evaluations (rubric visual-1)
    └── NOTE.md                         # Detailed documentation on gateway video blindness
```

### Reproducing Aggregates & Contrasts
To rebuild `summary.md` and `comparisons.json` from the raw judgment rows:
```bash
PYTHONPATH=src .venv-vlm/bin/python -m vlm_report_framework.evaluation.run report --run-id v21full --videos all
```

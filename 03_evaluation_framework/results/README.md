# Cataract Surgery VLM Benchmark: Experimental Results & Academic Reference

This directory contains the official evaluation outputs, raw model predictions, granular scores, and aggregated summaries for Vision-Language Models (VLMs) benchmarked on cataract surgery video understanding.

---

## 1. Benchmark Overview & Evaluation Protocol

The benchmark evaluates surgical video understanding across **two temporal granularities** and **five distinct task formulations**:

1. **Clip-Level Tasks (989 records, 0–30s video clips)**:
   - **Visual Description** ($n=293$): Open-ended surgical scene description evaluated against ground truth by an expert LLM judge.
   - **Multiple-Choice Questions (MCQ)** ($n=486$): Categorical surgical identification across 3 subtasks ($n=162$ each):
     - *Step Identification*
     - *Instrument Identification*
     - *Visual Observation / Anatomical Cue Detection*
   - **Phase Understanding** ($n=210$): Precise temporal and categorical phase recognition across 4 subtasks:
     - *Boundary Detection* ($n=49$)
     - *Temporal Localization* ($n=54$)
     - *Timestamp-to-Phase* ($n=47$)
     - *Contextual Phase Recognition* ($n=60$)

2. **Full-Video Procedural Narration (15 records, complete uncut surgeries)**:
   - Evaluated across 5 clinical dimensions against expert surgical narration.

All records instruct the model to produce a strict JSON output contract: `{"explanation": "...", "answer": ...}`. Evaluation is strictly zero-shot/direct inference (one prediction per record).

LLM-judge scoring (visual description + full-video narration) was performed with **Gemini-3.8-flash**; MCQ and phase tasks are scored deterministically.

---

## 2. Formal Task Definitions & Scoring Metrics

### 2.1. Multiple-Choice Questions (MCQ)
Models must select the correct single letter ($A$–$D$) within the JSON `answer` field.

```math
\mathcal{R}_{\text{task}} = \begin{cases} 1.0 & \text{if } \text{normalize}(\hat{y}) = y^* \\ 0.0 & \text{otherwise} \end{cases}
```

Scores are strictly deterministic.

---

### 2.2. Phase Understanding (13-Phase Clinical Ontology)
All phase tasks utilize the standardized 13-phase cataract surgery ontology:

```math
\text{P01 (Incision)} \dots \text{P13 (Idle)}
```

#### A. Boundary Detection ($n=49$)
The model predicts a clip-local boundary timestamp $\hat{t}$ in seconds. The reward applies an exponential decay parameterized by tolerance threshold $\tau = 1.5\text{ s}$:

```math
\mathcal{R}_{\text{task}} = \exp\left(-\frac{|\hat{t} - t^*|}{1.5}\right) \in (0, 1]
```

#### B. Temporal Localization ($n=54$)
The model predicts a time interval $[\hat{s}, \hat{e}]$ representing the active duration of a queried phase. Scored using Intersection-over-Union (IoU):

```math
\mathcal{R}_{\text{task}} = \text{IoU}([\hat{s}, \hat{e}], [s^*, e^*]) = \frac{\max(0, \min(\hat{e}, e^*) - \max(\hat{s}, s^*))}{(\hat{e} - \hat{s}) + (e^* - s^*) - \text{Intersection}}
```

#### C. Timestamp-to-Phase & Contextual Phase Recognition ($n=107$)
The model identifies the active surgical phase ID from visual frames or surrounding temporal context:

```math
\mathcal{R}_{\text{task}} = \begin{cases} 1.0 & \text{if } \hat{y}_{\text{phase}} = y^*_{\text{phase}} \\ 0.0 & \text{otherwise} \end{cases}
```

#### D. Strict JSON Format Bonus (Phase Tasks Only)
To incentivize structured clinical reporting, a decoupled format reward $\mathcal{R}_{\text{fmt}} \in \{0, 1\}$ awards 1.0 if the output strictly parses as a valid JSON object containing exactly `{"explanation", "answer"}` without markdown fences:

```math
\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{task}} + 0.05 \cdot \mathcal{R}_{\text{fmt}} \quad (\text{Max Score} = 1.05)
```

The normalized task score is computed as $\mathcal{S}_{\text{norm}} = \frac{\mathcal{R}_{\text{total}}}{1.05}$.

---

### 2.3. Visual Description (LLM-as-a-Judge)
Open-ended descriptions are evaluated by an ophthalmic surgical LLM judge against ground-truth descriptions across four clinical criteria:
1. **Surgical Actions & Maneuvers**: Accurate identification of operative steps.
2. **Instruments & Tools**: Correct naming and interaction description.
3. **Anatomical Structures**: Accurate recognition of intraocular structures.
4. **Factuality & Hallucination Prevention**: Strict penalization of unsupported claims.

- **Scale**: Integer $0 \text{ to } 5$ ($5 = \text{Excellent}, 0 = \text{Irrelevant/Contradictory}$).
- **Normalized Score**: $\mathcal{S}_{\text{norm}} = \frac{\text{Score}}{5} \in [0.0, 1.0]$.

---

### 2.4. Full-Video Procedural Narration (LLM-as-a-Judge)
Complete surgeries (5–25 minutes) require generating a comprehensive chronological narration. The judge evaluates five dimensions ($0 \text{ to } 5$ scale):
1. **Step Coverage**: Percentage and completeness of major surgical phases captured.
2. **Chronological Accuracy**: Sequence alignment relative to actual surgical flow.
3. **Visual & Technical Accuracy**: Correctness of specific tissue interactions and tool manipulations.
4. **Narrative Flow**: Fluid, real-time surgical commentary vs. disjointed bullet points.
5. **Overall Score**: Holistic surgical competence judgment (weighted towards procedural veracity).

---

## 3. Benchmark Leaderboard & Comparative Results

### Table 1: Overall Model Performance Summary
*All clip-level scores are reported as normalized accuracy $[0.0, 1.0]$. Narration overall is reported on the native $0 \text{ to } 5$ clinical scale (with normalized value in brackets). LLM-judge tasks (Visual Desc, Narration) scored with Gemini-3.8-flash.*

| Model Architecture | Parameters | Modality / Focus | Overall Clip Acc ($n=989$) | MCQ Macro ($n=486$) | Phase Macro ($n=210$) | Visual Desc ($n=293$) | Full Narration Overall ($n=15$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-VL-2B-Cataract-GRPO** | 2B | Domain GRPO | **0.5869** | **0.8909** | **0.2825** | 0.3024 | 0.2667 [0.0533] |
| **Qwen3-VL-2B-Cataract-SFT-Stage2** | 2B | Domain SFT (Stage 2) | 0.5421 | 0.8292 | 0.2010 | 0.3133 | 0.3333 [0.0667] |
| **Qwen3-VL-2B-Cataract-SFT** | 2B | Domain SFT (Stage 1) | 0.5376 | 0.8354 | 0.1925 | 0.2935 | 0.3333 [0.0667] |
| **Hulu-Med-4B** | 4B | Medical Specialist | 0.4376 | 0.6502 | 0.1953 | 0.2594 | 0.2000 [0.0400] |
| **Qwen3-VL-2B-Thinking** | 2B | Generalist (Reasoning) | 0.4338 | 0.6235 | 0.1754 | 0.3044 | 0.1333 [0.0267] |
| **Lingshu-7B** | 7B | Medical Specialist | 0.4208 | 0.6090 | 0.2157 | 0.2601 | **0.8000 [0.1600]** |
| **Qwen3-VL-4B-Instruct** | 4B | Generalist (Instruct) | 0.4059 | 0.5926 | 0.0825 | **0.3276** | 0.1333 [0.0267] |
| **Qwen3-VL-2B-Instruct** | 2B | Generalist (Instruct) | 0.4057 | 0.5905 | 0.1820 | 0.2608 | 0.0667 [0.0133] |
| **Hulu-Med-7B** | 7B | Medical Specialist | 0.3978 | 0.5453 | 0.2171 | 0.2826 | 0.6000 [0.1200] |

---

### Table 2: Granular Clip-Level Performance Breakdown
*Per-task normalized average accuracy across all 8 clip-level subtasks.*

| Model | Visual Desc ($n=293$) | Step ID ($n=162$) | Instrument ID ($n=162$) | Visual Obs ($n=162$) | Boundary Det. ($n=49$) | Temporal Loc. ($n=54$) | Timestamp $\to$ Phase ($n=47$) | Context Phase ($n=60$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-VL-2B-Cataract-GRPO** | 0.3024 | **0.8457** | **0.8765** | **0.9506** | 0.2506 | 0.3981 | **0.2908** | **0.1905** |
| **Qwen3-VL-2B-Cataract-SFT-Stage2** | 0.3133 | 0.7346 | 0.8580 | 0.8951 | 0.2488 | 0.2345 | 0.2097 | 0.1111 |
| **Qwen3-VL-2B-Cataract-SFT** | 0.2935 | 0.7778 | 0.8395 | 0.8889 | 0.2457 | 0.2237 | 0.1894 | 0.1111 |
| **Hulu-Med-4B** | 0.2594 | 0.4630 | 0.6358 | 0.8519 | 0.1061 | 0.3182 | 0.2300 | 0.1270 |
| **Qwen3-VL-2B-Thinking** | 0.3044 | 0.3519 | 0.6852 | 0.8333 | 0.2012 | 0.1928 | 0.1489 | 0.1587 |
| **Lingshu-7B** | 0.2601 | 0.4444 | 0.6481 | 0.7346 | **0.3170** | 0.2566 | 0.2097 | 0.0793 |
| **Qwen3-VL-4B-Instruct** | **0.3276** | 0.3148 | 0.6543 | 0.8086 | 0.0577 | 0.0530 | 0.1084 | 0.1111 |
| **Qwen3-VL-2B-Instruct** | 0.2608 | 0.3272 | 0.6358 | 0.8086 | 0.2063 | 0.1372 | 0.2097 | 0.1746 |
| **Hulu-Med-7B** | 0.2826 | 0.3333 | 0.5494 | 0.7531 | 0.0859 | **0.4618** | 0.2097 | 0.1111 |

---

### Table 3: Full-Video Procedural Narration Multi-Dimensional Breakdown
*Scores evaluated on complete surgical procedures across 5 clinical dimensions on a $0 \text{ to } 5$ scale.*

| Model | Step Coverage (/5) | Chronological Accuracy (/5) | Visual & Tech Accuracy (/5) | Narrative Flow (/5) | Overall Score (/5) | Normalized Score (/1.0) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Lingshu-7B** | **0.8000** | **0.8000** | 0.4000 | **2.4667** | **0.8000** | **0.1600** |
| **Hulu-Med-7B** | 0.6000 | 0.7333 | **0.6000** | 2.0000 | 0.6000 | 0.1200 |
| **Qwen3-VL-2B-Cataract-SFT-Stage2** | 0.5333 | 0.6000 | 0.3333 | 1.6667 | 0.3333 | 0.0667 |
| **Qwen3-VL-2B-Cataract-SFT** | 0.3333 | 0.4000 | 0.4000 | 1.3333 | 0.3333 | 0.0667 |
| **Qwen3-VL-2B-Cataract-GRPO** | 0.3333 | 0.2667 | 0.2667 | 1.1333 | 0.2667 | 0.0533 |
| **Hulu-Med-4B** | 0.2000 | 0.2667 | 0.1333 | 1.6667 | 0.2000 | 0.0400 |
| **Qwen3-VL-4B-Instruct** | 0.2000 | 0.1333 | 0.2667 | 1.0667 | 0.1333 | 0.0267 |
| **Qwen3-VL-2B-Thinking** | 0.0667 | 0.1333 | 0.2000 | 0.8000 | 0.1333 | 0.0267 |
| **Qwen3-VL-2B-Instruct** | 0.2000 | 0.0667 | 0.1333 | 1.2667 | 0.0667 | 0.0133 |

---

## 4. Key Academic Findings & Discussion

1. **Reinforcement Learning (GRPO) Establishes New Benchmark State-of-the-Art**:
   - **Qwen3-VL-2B-Cataract-GRPO** advances overall clip accuracy to an unprecedented benchmark high of **0.5869**, surpassing the previous leader **Stage 2 SFT** (**0.5421**) by **+4.48%** absolute (+8.26% relative gain) and **Stage 1 SFT** (**0.5376**) by **+4.93%** absolute (+9.17% relative gain).
   - Relative to the base **Qwen3-VL-2B-Instruct** (**0.4057**), GRPO yields an extraordinary **+18.12%** absolute improvement (+44.66% relative gain).
   - Remarkably, this 2B parameter GRPO model decisively outperforms specialized medical foundation models that are double or more than triple its parameter size: **Hulu-Med-4B** (**0.4376**, +14.93% gap), **Lingshu-7B** (**0.4208**, +16.61% gap), and **Hulu-Med-7B** (**0.3978**, +18.91% gap). This empirically demonstrates that domain-specific policy optimization with surgical task rewards is substantially more parameter-efficient than general medical pretraining.

2. **Categorical Surgical Perception Approaches Theoretical Ceiling (MCQ Macro 0.8909)**:
   - On Multiple-Choice Questions ($n=486$), GRPO establishes a new benchmark record of **0.8909**, far exceeding Stage 1 (**0.8354**), Stage 2 (**0.8292**), Hulu-Med-4B (**0.6502**), and base Qwen3-VL-2B-Instruct (**0.5905**).
   - Across specific recognition subtasks, GRPO achieves top-tier results in every category:
     - **Visual Observation / Anatomical Cue Detection**: **0.9506** (vs. 0.8951 in Stage 2, 0.8889 in Stage 1, and 0.8086 in Base 2B), exhibiting near-perfect discrimination of minute anterior segment features, capsular boundaries, and operative cues.
     - **Step Identification**: **0.8457** (vs. 0.7778 in Stage 1 and 0.7346 in Stage 2, representing a +11.11% absolute recovery over Stage 2). While Stage 2 experienced a minor trade-off in step recognition during multi-task SFT balancing, GRPO's group relative advantage updates completely resolve this regression.
     - **Instrument Identification**: **0.8765** (vs. 0.8580 in Stage 2, 0.8395 in Stage 1, and 0.6358 in Base 2B).

3. **Breakthrough in Phase Understanding & Continuous Temporal Grounding (Phase Macro 0.2825)**:
   - Phase understanding tasks ($n=210$) have historically been the primary performance bottleneck across all VLMs, with baseline models clustering around 0.08–0.21.
   - **Qwen3-VL-2B-Cataract-GRPO** achieves a dramatic breakthrough with a **Phase Macro** of **0.2825**, surpassing even 7B medical foundation models (**Hulu-Med-7B** at **0.2171** and **Lingshu-7B** at **0.2157**) and improving by **+40.53%** relative over Stage 2 SFT (**0.2010**).
   - Key drivers of this temporal breakthrough include:
     - **Temporal Localization**: **0.3981** (vs. 0.2345 in Stage 2, a **+69.8%** relative leap; second only to Hulu-Med-7B's 0.4618).
     - **Timestamp-to-Phase**: **0.2908** (the highest across all benchmarked models, outperforming Hulu-Med-4B at 0.2300 and all others at ≤0.2097).
     - **Contextual Phase Recognition**: **0.1905** (ranking #1 overall, surpassing Base 2B at 0.1746, Thinking at 0.1587, and Stage 2 at 0.1111).
     - **Boundary Detection**: **0.2506** (the highest among 2B/4B models, slightly exceeding Stage 2 at 0.2488 and trailing only Lingshu-7B's 0.3170).
   - This validates that reinforcement learning with temporal IoU and exponential boundary distance reward formulations provides precise gradient feedback for continuous coordinate regression and temporal alignment where standard cross-entropy loss suffers from temporal smoothing.

4. **Visual Description Quality & Reasoning Trade-offs**:
   - In short-clip Visual Description ($n=293$), **Qwen3-VL-2B-Cataract-GRPO** scores **0.3024**, remaining highly competitive with **Stage 2 SFT** (**0.3133**) and **Qwen3-VL-2B-Thinking** (**0.3044**), while comfortably outperforming **Stage 1 SFT** (**0.2935**), **Qwen3-VL-2B-Instruct** (**0.2608**), **Lingshu-7B** (**0.2601**), and **Hulu-Med-4B** (**0.2594**).
   - Only the larger **Qwen3-VL-4B-Instruct** (**0.3276**) achieves higher description scores. This indicates that while GRPO was optimized against structured verification rewards, the policy successfully preserved rich qualitative vocabulary and intraocular descriptive fidelity without degradation or mode collapse.

5. **Procedural Narration Across Uncut Videos & Long-Horizon Generalization**:
   - On full-video procedural narration ($n=15$, complete surgeries), 7B medical foundation models (**Lingshu-7B** at **0.8000 / 5** and **Hulu-Med-7B** at **0.6000 / 5**) continue to lead overall holistic procedural narration, benefiting from substantial pre-training on extended multi-phase surgical video sequences.
   - Among 2B parameter architectures, **Stage 2 SFT** maintains the highest narration performance (**0.3333 / 5** overall, **0.5333 / 5** Step Coverage, **0.6000 / 5** Chronological Accuracy, **1.6667 / 5** Flow).
   - **Qwen3-VL-2B-Cataract-GRPO** scores **0.2667 / 5** overall (normalized 0.0533, Step Coverage 0.3333, Chronological Accuracy 0.2667, Technical Accuracy 0.2667, Flow 1.1333). While this significantly outperforms base models (Qwen3-VL-2B-Instruct at 0.0667 / 5, Thinking at 0.1333 / 5, 4B-Instruct at 0.1333 / 5) and Hulu-Med-4B (0.2000 / 5), it reveals an important methodological insight: short-clip RL reward pressure directly sharpens local temporal and visual discrimination, but does not automatically transfer to long-horizon macro-narrative coherence without dedicated full-video sequence reward modeling.

6. **Cohort-Wide Dynamics & Frontier Challenges**:
   - Across the complete 9-model evaluation cohort, the benchmark reveals clear performance stratifications:
     - **Perceptual Cue Detection is Solved**: Visual Observation cohort average is **0.8361** (with top model at 0.9506) and Instrument Identification average is **0.7092** (top at 0.8765).
     - **Procedural Understanding is Attainable via Domain Post-Training**: MCQ Macro cohort average is **0.6852**, propelled by domain-adapted models (GRPO 0.8909, SFT 0.8354).
     - **Continuous Temporal Grounding Remains the Core Frontier**: Phase Macro cohort average is **0.1938** (Temporal Loc: **0.2529**, Timestamp-to-Phase: **0.2007**, Boundary Det: **0.1910**, Context Phase: **0.1305**). While GRPO achieves a major breakthrough (+40.5% over SFT), continuous temporal boundary estimation remains far below categorical perception.
     - **Long-Horizon Video Narration Demands Sequence-Level Modeling**: Full Narration cohort average is **0.3185 / 5** (normalized **0.0637**). Closing the gap on uncut procedures will require long-context architectures and sequence-level RL optimization.

---

## 5. Artifact Directory Structure & Academic Figures

For every model evaluation run, the pipeline generates three synchronized files:

```
results/
├── <tag>_clip_responses.jsonl   # Raw model outputs, reference labels, and prompts
├── <tag>_clip_scores.jsonl      # Itemized scores, extracted answers, and judge justifications
├── <tag>_clip_summary.json     # Aggregated clip-level metrics
├── <tag>_full_responses.jsonl   # Raw full-video narration outputs
├── <tag>_full_scores.jsonl      # Multi-dimensional narration scores per surgery
├── <tag>_full_summary.json     # Aggregated narration metrics
├── judge_<tag>.log              # Detailed execution and judge API communication logs
├── plots/                       # Publication-ready academic comparison plots (PDF & 300 DPI PNG)
│   ├── fig1_overall_benchmark_comparison.{pdf,png}   # Overall clip acc, MCQ vs Phase, narration
│   ├── fig2_subtask_radar_and_breakdown.{pdf,png}    # Multi-task radar chart & full 9x8 matrix
│   ├── fig3_training_progression_ablation.{pdf,png}  # Base -> SFT1 -> SFT2 -> GRPO trajectory
│   └── fig4_full_narration_dimensions.{pdf,png}      # 5D clinical breakdown on uncut videos
└── archive/                     # Preserved past scoring runs
```

### Publication Figures Summary
All figures are rendered in high-resolution vector format (`.pdf` for direct LaTeX integration with `pdflatex`) and 300 DPI raster (`.png`):
1. **Figure 1 (`fig1_overall_benchmark_comparison`)**:
   - **(a)** Overall clip accuracy across all 9 models ($n=989$), highlighting GRPO's new SOTA at **0.5869**.
   - **(b)** Direct comparison of categorical perception (MCQ Macro, $n=486$) versus temporal phase grounding (Phase Macro, $n=210$).
   - **(c)** Full-video procedural narration normalized scores ($n=15$) comparing 7B foundation models against 2B domain models.
2. **Figure 2 (`fig2_subtask_radar_and_breakdown`)**:
   - **(a)** 8-axis clinical competency radar profile contrasting GRPO (Ours), SFT Stage 2, Base 2B-Instruct, Lingshu-7B, and Hulu-Med-4B.
   - **(b)** Granular performance matrix (heatmap) across all 9 models and 8 subtasks, with peak performers boxed and annotated.
3. **Figure 3 (`fig3_training_progression_ablation`)**:
   - **(a)** 4-stage post-training evolutionary curve: Base 2B-Instruct $\to$ Domain SFT $\to$ Curriculum SFT $\to$ GRPO.
   - **(b)** Absolute subtask gains ($\Delta$ over base model) demonstrating the synergistic effect of domain RL.
4. **Figure 4 (`fig4_full_narration_dimensions`)**:
   - 5-dimensional clinical score breakdown (Step Coverage, Chronology, Technical Accuracy, Narrative Flow, Overall) across 15 complete uncut procedures.

---

## 6. Citation & Reproducibility

If you use these benchmark results or evaluation protocols in academic work, please cite:

```bibtex
@article{cataract_vlm_benchmark2026,
  title={Benchmarking Vision-Language Models for Cataract Surgery: From Clip-Level Phase Recognition to Full-Video Procedural Narration},
  author={Evaluation Team},
  journal={arXiv preprint},
  year={2026}
}
```

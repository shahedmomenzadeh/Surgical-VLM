# 3. Methodology

## 3.1 Dataset Design and Sources

Developing vision-language models (VLMs) for complex surgical domains such as cataract surgery (phacoemulsification) requires representations that capture multiple levels of visual perception, clinical reasoning, and temporal understanding. Existing surgical vision datasets have predominantly focused on frame-level recognition, phase recognition, or short video classification, providing comparatively limited supervision for open-ended visual description, procedural reasoning, and reinforcement-learning-based alignment.

To address these limitations, we constructed a **dual-source, dual-resolution surgical dataset** that combines complementary forms of supervision from in-the-wild surgical recordings and an annotated clinical phase-recognition corpus. The resulting dataset contains **2,624 temporally segmented video clips from 286 parent procedures**, including **136 full-length surgical procedures**. It supports both **Supervised Fine-Tuning (SFT)** and **Group Relative Policy Optimization (GRPO)**, comprising **2,760 SFT samples** and **5,505 GRPO samples**.

The first source, referred to as the **In-the-Wild Surgical Corpus**, consists of publicly available cataract surgery recordings collected from YouTube. These recordings provide substantial visual and procedural diversity, including variations in microscope systems, illumination, image quality, surgical techniques, instrumentation, anatomical presentation, and intraoperative events. Where available, surgeon commentary provides additional linguistic context regarding surgical intent and intraoperative decision-making. The corpus therefore supports learning from heterogeneous, naturally occurring surgical observations rather than a single controlled acquisition setting.

The second source, referred to as the **Clinical Phase Dataset**, consists of annotated surgical clips associated with a standardized 13-phase cataract surgery ontology. In contrast to the open-vocabulary YouTube corpus, this dataset provides structured phase-level supervision and dense temporal coverage of the surgical workflow. It therefore provides a controlled source for phase recognition and verifiable reinforcement-learning objectives.

The two sources serve complementary purposes. The in-the-wild corpus provides broader visual and procedural variability, while the annotated phase corpus provides standardized and verifiable clinical labels. Their combination enables the model to learn both open-ended surgical understanding and structured clinical recognition.

| Dimension | In-the-Wild Surgical Corpus | Clinical Phase Dataset | Combined Benefit |
| :--- | :--- | :--- | :--- |
| Primary video format | Full procedures (10–25 min) and step clips (8–15 s) | Phase sub-clips (5–30 s) | Dual-resolution temporal representation |
| Visual context | Diverse microscopes, image quality, illumination, and viewpoints | More controlled clinical acquisition | Improved robustness to visual variability |
| Audio / narration | Surgeon commentary, transcribed and refined where available | No narration required | Linguistic grounding + visual verification |
| Taxonomy | Open-vocabulary surgical steps and instruments | Closed 13-phase ontology | Open-ended reasoning + standardized classification |
| Supervision | Step identification, visual observation, instrument identification, narration | Phase identification and grounded visual observation | Complementary SFT and GRPO supervision |

## 3.2 Dual-Resolution Temporal Representation

Surgical procedures exhibit a hierarchical temporal structure. Fine-grained surgical actions may occur over several seconds, whereas clinically meaningful transitions and procedural context may span several minutes. To capture both temporal scales, the dataset represents procedures at two resolutions.

### 3.2.1 Micro-Resolution: Step Clips

At the micro-resolution, individual surgical steps are extracted as short video clips using frame-accurate temporal segmentation. The clips have a mean duration of **13.2 ± 5.6 s** across the combined dataset and are normalized to a common temporal sampling rate of **10 FPS** for downstream processing. Each clip is associated with structured metadata and textual supervision for surgical-step understanding, visual observation, and reinforcement-learning objectives.

The clip-level data support four primary question types:

1. **Step identification**, which identifies the operative maneuver being performed.
2. **Instrument identification**, which identifies the actively used surgical instrument.
3. **Visual observation**, which describes visible anatomical or dynamic surgical changes.
4. **Phase identification**, which maps the clip to one of the 13 standardized surgical phases.

### 3.2.2 Macro-Resolution: Full Procedures

At the macro-resolution, complete surgical procedures are retained as full-length videos. These videos are paired with chronologically ordered, timestamped procedural narrations covering the major surgical steps and their temporal boundaries. This representation provides supervision for long-context procedural understanding and enables the model to associate localized surgical actions with the broader workflow of cataract surgery.

The resulting hierarchy therefore spans localized actions occurring over seconds and complete procedures unfolding over many minutes, providing complementary supervision for fine-grained surgical perception and long-range procedural reasoning.

## 3.3 Data Curation and Preprocessing

The data curation pipeline converts raw surgical recordings into temporally aligned, quality-controlled multimodal training samples. The processing stages include video acquisition and filtering, speech transcription, medical terminology refinement, multimodal surgical annotation, temporal segmentation, sample formatting, and quality control.

### 3.3.1 Speech Transcription and Medical Lexical Refinement

For recordings containing surgeon commentary, audio is extracted as single-channel, 16-bit PCM WAV sampled at 16 kHz. The **Whisper large** model is used to generate timestamped speech segments. Because automatic speech recognition may produce errors in technical ophthalmic terminology, an OpenAI-compatible large language model is subsequently used for medical lexical normalization. The refinement stage corrects terminology while preserving the original temporal boundaries of the ASR output. Examples include correcting phonetic errors such as *capsule rexis* to *capsulorhexis*, *faco* to *phacoemulsification*, and *keratom* to *keratome*.

### 3.3.2 Multimodal Surgical Annotation

The in-the-wild recordings undergo multimodal analysis to identify major surgical steps, temporal boundaries, visual descriptions, instruments, and anatomical structures. A lightweight textual screening stage first filters recordings for surgical relevance and removes unsuitable content. A multimodal VLM then analyzes the surgical video and produces structured event descriptions. These annotations are subsequently used to construct temporally localized training samples.

In addition, the generated procedural descriptions are reviewed for clinically relevant intraoperative events. The complication screening stage identifies events such as posterior capsule rupture, vitreous loss, iris prolapse, zonular dialysis, intraoperative floppy iris syndrome (IFIS), nucleus drop, and IOL drop when they are supported by the available visual evidence.

### 3.3.3 Teacher-Critic Quality Control for Phase Data

The Clinical Phase Dataset undergoes an independent quality-control procedure before SFT and GRPO sample generation. A **Teacher-Critic framework** is used to reduce unsupported visual claims and to verify the correspondence between the phase label and the observed surgical content.

The Teacher model generates a grounded visual observation conditioned on the visible surgical scene. An independent Critic model then reviews the video and evaluates whether the anatomical, instrumental, and procedural claims are supported by the visual evidence. Samples that contain unsupported claims are rejected or excluded from subsequent training-data generation.

Special attention is given to the **P13 (Idle)** phase. Passive support devices, such as eyelid specula or lid retractors, are distinguished from active intraocular manipulation. Clips labeled as Idle that contain evidence of active surgical action are rejected to reduce ambiguity at the boundary between inactive and active phases.

## 3.4 Data Partitioning and Leakage Prevention

To prevent temporal or procedural leakage between training and evaluation, data are partitioned at the **parent-procedure level** rather than at the individual clip level. Consequently, all clips derived from a given procedure inherit the same split assignment.

The final dataset contains **213 training procedures, 35 validation procedures, and 38 test procedures**, corresponding to **74.5%, 12.2%, and 13.3%** of the 286 parent procedures, respectively. A fixed pseudo-random seed (`seed = 42`) is used to establish the split, and the resulting assignments are stored in a dedicated split file. This procedure-level partitioning prevents clips originating from the same surgical procedure from appearing in multiple splits.

## 3.5 SFT Dataset Construction

The SFT dataset is designed to provide direct multimodal supervision for surgical visual understanding. Each clip-level SFT sample contains a video input, a task-specific textual instruction, and a target response. The target response provides a clinically grounded description or answer that links the observed visual evidence to the requested surgical concept.

The SFT data comprise **2,624 clip-level samples** and **136 full-procedure narration samples**, resulting in **2,760 SFT samples** in total. Clip-level supervision focuses on localized surgical understanding, whereas full-procedure samples provide timestamped descriptions across the complete workflow.

For clip-level tasks, prompts are constructed around surgical-step identification and grounded visual observation. The full-procedure task asks the model to generate a chronological narration with explicit temporal boundaries, allowing the model to learn the relationship between surgical events and their position within the overall procedure.

## 3.6 GRPO Dataset Construction and Verifiable Rewards

GRPO samples are designed to provide verifiable objectives for reinforcement-learning-based alignment. Each sample contains a video clip, a question, four candidate answers, and a known target answer or reference description. The question requires the model to use the visual content to identify a surgical step, instrument, visual observation, or surgical phase.

The GRPO dataset contains **5,505 records** distributed across four task types:

| Question Type | Source | Records | Primary Target | Reward |
| :--- | :--- | ---: | :--- | :--- |
| `step_identification` | YouTube | 1,441 | Operative surgical step | Deterministic |
| `instrument_identification` | YouTube | 1,440 | Active surgical instrument | Deterministic |
| `visual_observation` | YouTube | 1,440 | Anatomical / dynamic change | LLM judge |
| `phase_identification` | Clinical Phase Dataset | 1,184 | 13-class surgical phase | Deterministic |
| **Total** | **All sources** | **5,505** | — | — |

For classification-oriented tasks, the primary reward is deterministic exact-match verification. Given a predicted answer \(\hat{y}\) and a gold label \(y^*\), the deterministic reward is defined as

$$
R_{\mathrm{det}}(\hat{y}, y^*) = \mathbf{1}\left(\mathrm{ExtractAnswer}(\hat{y}) = y^*\right).
$$

For visual-observation tasks, correctness cannot always be reduced to an exact label match. These samples therefore use an LLM-based judge to evaluate the consistency of the generated response with a reference clinical observation:

$$
R_{\mathrm{judge}}(\hat{y}, y^*_{\mathrm{reason}}) \in [0,1].
$$

The combined reward design provides a mixture of strictly verifiable objectives and graded semantic evaluation. Overall, **78.5%** of the GRPO records use deterministic rewards, while **21.5%** use the LLM-judge reward.

### 3.6.1 Distractor Construction and Refinement

Multiple-choice distractors are designed to remain plausible while avoiding ambiguous alternatives. An automated judge is used to identify functionally overlapping or visually confusable choices. When necessary, ambiguous distractors are replaced with alternatives from sufficiently distinct functional or visual categories. This procedure reduces the likelihood that a question can be answered through superficial elimination rather than visual understanding.

## 3.7 Clinical Phase Ontology

The Clinical Phase Dataset uses a 13-class ontology representing the major phases of cataract surgery. The distribution of the 1,184 phase clips is shown below.

| Phase ID | Canonical Phase Name | Clinical Description | Sample Count |
| :--- | :--- | :--- | ---: |
| **P01** | Incision | Creation of clear corneal incision / paracentesis | 112 |
| **P02** | Viscoelastic | Injection of ophthalmic viscosurgical device (OVD) | 128 |
| **P03** | Capsulorhexis | Continuous curvilinear anterior capsulotomy | 142 |
| **P04** | Hydrodissection | Fluid wave dissection of cortical-capsular adhesions | 104 |
| **P05** | Phacoemulsification | Ultrasonic sculpting, chopping, and nuclear quadrant removal | 168 |
| **P06** | Irrigation-Aspiration | Cortical cleanup and subcapsular epinucleus removal | 134 |
| **P07** | Capsule Polishing | Posterior capsule scraping and polishing | 64 |
| **P08** | Lens Implantation | Injector delivery of foldable intraocular lens (IOL) | 118 |
| **P09** | Lens Positioning | Dialing and centering the IOL inside the capsular bag | 76 |
| **P10** | Viscoelastic Suction | Evacuation of residual viscoelastic material | 88 |
| **P11** | AC Flushing | Hydro-reformation and anterior chamber flushing | 52 |
| **P12** | Tonifying-Antibiotics | Intracameral antibiotic injection and stromal hydration | 46 |
| **P13** | Idle | Inactive operational interval | 52 |
| **Total** | **13 phases** | **Cataract surgical workflow** | **1,184** |

## 3.8 Dataset Statistics

### 3.8.1 Global Dataset Composition

| Dataset Partition | Parent Procedures | Video Clips | Full-Length Videos | Clip SFT | Full-Video SFT | GRPO |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Train** | 213 (74.5%) | 2,066 (78.7%) | 108 (79.4%) | 2,066 | 108 | 4,328 (78.6%) |
| **Validation** | 35 (12.2%) | 265 (10.1%) | 13 (9.6%) | 265 | 13 | 557 (10.1%) |
| **Test** | 38 (13.3%) | 293 (11.2%) | 15 (11.0%) | 293 | 15 | 620 (11.3%) |
| **Total** | **286** | **2,624** | **136** | **2,624** | **136** | **5,505** |

### 3.8.2 Source Breakdown

| Metric | In-the-Wild Surgical Corpus | Clinical Phase Dataset | Combined |
| :--- | ---: | ---: | ---: |
| Parent procedures | 136 | 150 | **286** |
| Video clips | 1,440 | 1,184 | **2,624** |
| Full-length videos | 136 | 0 | **136** |
| Full-procedure SFT narrations | 136 | 0 | **136** |
| Clip-level SFT samples | 1,440 | 1,184 | **2,624** |
| GRPO records | 4,321 | 1,184 | **5,505** |
| Mean clip duration | 12.4 ± 4.2 s | 14.1 ± 6.8 s | **13.2 ± 5.6 s** |
| Total video duration | ≈24.6 h | ≈4.6 h | **≈29.2 h** |

## 3.9 Data Record Structure

The dataset uses three principal record types.

### 3.9.1 Clip-Level SFT Records

Each clip-level SFT record contains a video input, a textual instruction, and a target response. The instruction specifies the desired surgical task, while the target response provides the corresponding clinically grounded answer or description.

A representative structure is:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "procedure_id/clip_01.mp4"
        },
        {
          "type": "text",
          "text": "What surgical step is being performed in this clip?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "The surgeon performs a continuous circular opening of the anterior capsule, consistent with capsulorhexis."
    }
  ]
}
```

### 3.9.2 GRPO Records

Each GRPO record contains the multimodal prompt, the multiple-choice options, the correct answer, the task type, and the information required for reward computation. Depending on the task, the reward is obtained by deterministic answer matching or semantic comparison against a reference observation.

```json
{
  "prompt": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "procedure_id/clip_01.mp4"
        },
        {
          "type": "text",
          "text": "Which instrument is actively manipulating ocular tissue in this clip?"
        }
      ]
    }
  ],
  "correct_answer": "B",
  "question_type": "instrument_identification",
  "reference_reasoning": "The instrument enters through the corneal incision and is used to manipulate the ocular tissue.",
  "reward_type": "deterministic"
}
```

### 3.9.3 Full-Procedure SFT Records

Full-procedure records contain the complete surgical video together with a timestamped narration describing the major procedural events. Each narration entry contains a temporal interval and a corresponding surgical description.

```text
[00:07 - 00:15] viscoelastic injection: A cannula is introduced into the anterior chamber to inject viscoelastic and stabilize the chamber.
[00:16 - 00:45] capsulorhexis: Capsulorhexis forceps create a circular opening in the anterior lens capsule.
[00:46 - 01:12] hydrodissection: Fluid is injected beneath the capsule to mobilize the lens nucleus.
[01:13 - 03:40] phacoemulsification: The phaco probe emulsifies and aspirates the cataractous nucleus.
```

## 3.10 Summary of the Dataset Design

The resulting dataset provides four complementary properties that support surgical VLM development:

1. **Hierarchical temporal supervision:** the dataset spans localized surgical actions and complete procedure-level narratives.
2. **Complementary data sources:** in-the-wild recordings provide visual and procedural diversity, while the annotated phase corpus provides structured clinical labels.
3. **Verifiable reinforcement-learning objectives:** GRPO samples combine deterministic classification rewards with graded semantic evaluation for visual-observation tasks.
4. **Procedure-level data integrity:** split assignment is performed at the parent-procedure level, while teacher-critic quality control and distractor refinement are used to improve annotation reliability.

Together, these design choices establish a unified dataset for supervised multimodal learning and verifiable reinforcement-learning alignment in cataract surgery.

## 3.11 Training and Evaluation

The dataset construction described above serves as the foundation for the subsequent model-training and evaluation procedures. The training methodology is presented separately and describes the base VLM, parameter-efficient fine-tuning strategy, supervised fine-tuning stage, GRPO optimization, and training configuration. The evaluation protocol is then presented independently, including the held-out test sets, comparison baselines, evaluation tasks, and quantitative metrics.

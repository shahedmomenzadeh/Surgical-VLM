# `phase_sft_dataset.py` - Grounded Phase SFT Generation

Generates critic-approved visual-observation SFT records from the selected
phase-recognition clips. This stage is separate from the YouTube pipeline and
must run before phase GRPO generation.

The stage uses two OpenAI-compatible chat-completion calls per accepted clip:

1. A teacher VLM creates a phase-relevant visual observation.
2. A critic VLM verifies the observation against the video and corrects or
   rejects unsupported content.

Candidate `P13` (`Idle`) clips receive an additional boundary-quality check. If
visible surgical activity occurs in an Idle clip, the clip is rejected and no
SFT record is written.

## Usage

Run from the project parent directory with the project virtual environment:

```bash
cd /mnt/d/programming/MSc_project
video_pipeline/.venv/bin/python -m video_pipeline.tools.phase_sft_dataset \
  --config video_pipeline/config.yaml
```

Process a small subset before starting a full run:

```bash
video_pipeline/.venv/bin/python -m video_pipeline.tools.phase_sft_dataset \
  --config video_pipeline/config.yaml \
  --split Validation \
  --limit 10
```

Available options:

| Option | Description |
|---|---|
| `--config PATH` | Configuration file to use |
| `--split SPLIT` | Process only `Train`, `Validation`, or `Test` |
| `--limit N` | Process at most `N` manifest records after split filtering |
| `--retry-rejected` | Reprocess previously rejected Idle clips |
| `--dry-run` | Check source paths and existing outputs without API calls |

## Configuration

The configuration is under `phase_sft` in `config.yaml`:

```yaml
phase_sft:
  source_dir: "./phase_dataset"
  manifest_file: "./phase_dataset/manifest.csv"
  output_dir: "./phase_vlm_dataset"
  log_file: "phase_sft_log.csv"
  materialization: "hardlink"
  hardlink_fallback_copy: true
  video_part_type: "video_url"
  max_attempts: 2
  max_retries: 3
  retry_delay_seconds: 3
  request_delay_seconds: 1
  max_video_bytes: 0
  fallback_max_frames: 32
  frame_quality: 90
  teacher:
    base_url: "http://localhost:8000/v1"
    model: "qwen3.8-max"
    api_key: "none"
    api_key_env: "PHASE_SFT_TEACHER_API_KEY"
  critic:
    base_url: "http://localhost:8000/v1"
    model: "qwen3.8-max"
    api_key: "none"
    api_key_env: "PHASE_SFT_CRITIC_API_KEY"
```

Important fields:

| Field | Meaning |
|---|---|
| `source_dir` | Root of the selected phase dataset |
| `manifest_file` | Manifest used as the source of truth for clip metadata |
| `output_dir` | Root of the generated `phase_vlm_dataset` |
| `log_file` | Progress log path; relative values resolve inside `output_dir` |
| `materialization` | `hardlink`, `copy`, or `symlink` for accepted MP4s |
| `hardlink_fallback_copy` | Copy the video if hardlink creation fails |
| `max_attempts` | Teacher/critic review rounds for one clip |
| `max_retries` | API retries for one media representation |
| `retry_delay_seconds` | Exponential retry delay base |
| `request_delay_seconds` | Delay after teacher and critic requests |
| `max_video_bytes` | Optional pre-request video-size limit; `0` disables it |
| `fallback_max_frames` | Maximum frames sent after full-video failure |
| `frame_quality` | JPEG quality used for frame fallback |

The phase generator supports a literal `api_key` in the phase configuration.
If it is absent, it resolves `api_key_env` through the normal `.env` and
Google Colab secret mechanism. The local test endpoint accepts `none` as the
API key.

## Input Manifest

The manifest must contain at least:

```text
split
sample_id
source_procedure
phase_id
phase_name
```

The existing manifest also provides `site`, `subclip_order`,
`phase_occurrence`, `duration_sec`, `previous_phase`, `next_phase`,
`context_bucket`, `source_path`, and `dataset_path`.

The source video is resolved from `dataset_path` when that file exists. If it
does not, the generator falls back to:

```text
{source_dir}/{split}/{source_procedure}/{sample_id}.mp4
```

Records are sorted by split, procedure, and source `subclip_order`. Neutral
clip numbers are assigned from that stable order.

## Media Request Strategy

The complete MP4 is always attempted first. It is read locally and encoded as
a base64 data URI:

```text
data:video/mp4;base64,...
```

The video is sent through an OpenAI-compatible `video_url` content part. The
training JSONL output is separate and continues to use the existing dataset's
`type: "video"` field.

### Frame fallback

The generator switches to frame input when the full-video request returns a
qualifying upstream or payload error, including HTTP `502`, `503`, `504`, or
`413`, and messages containing `upstream error`, `service unavailable`,
`gateway timeout`, or `payload too large`.

Before extracting frames, `ffprobe` checks the actual decoded frame count. The
number sent is:

```text
min(actual_frame_count, fallback_max_frames)
```

Frames are sampled evenly across the clip and encoded as base64 JPEG image
data URIs. A clip with fewer than 32 frames sends every available frame. The
frame representation uses OpenAI-compatible `image_url` content parts.

Authentication errors, malformed requests, and JSON validation errors are not
silently treated as media fallback conditions. They are retried using the
current representation and logged when retries are exhausted.

## Teacher and Critic

### Teacher response

The teacher receives the video and the verified phase ID/name as conditioning
context. The phase label focuses the observation but must not replace visual
evidence. The teacher must not mention the phase label, filename, or metadata
in the description.

The expected response contains:

```text
decision: ACCEPT, SKIP, or REJECT
visual_description: grounded visual description
instruments: visibly supported instruments
anatomy: visibly supported anatomy
reason: short explanation or rejection reason
```

For candidate `P13` clips, `SKIP` means that active surgery was visible and
the Idle boundary is invalid.

### Critic response

The critic receives the same video, phase context, and teacher proposal. It
checks visual grounding, phase consistency, instruments, anatomy, actions,
tissue changes, and unsupported complications.

The critic uses the same response fields and may return `ACCEPT`, `SKIP`, or
`REJECT`. Its approved or corrected observation becomes the final metadata and
SFT content.

For both active and Idle clips, a usable teacher `SKIP` or `REJECT` response is
still sent to the critic so the critic can independently inspect the video and
correct an overly cautious teacher decision. If the teacher request fails or
its response cannot be validated, the critic is marked `NOT_CALLED` for that
attempt.

## Output Layout

Accepted records are written under `phase_vlm_dataset`:

```text
phase_vlm_dataset/
├── Train/
├── Validation/
├── Test/
├── phase_sft_log.csv
└── .work/
    └── reviews/
```

Each accepted clip has:

```text
{split}/{parent_video_id}/
├── clip_XX.mp4
├── clip_XX.jsonl
└── clip_XX_sft.jsonl
```

The neutral clip number is assigned from manifest order within the procedure.
Rejected clips can leave numbering gaps; this keeps paths stable across
resumed runs. This stage does not create `full_video.*` or GRPO files.

## Metadata JSONL

The core `clip_XX.jsonl` record contains phase-specific metadata, including:

```text
clip_id
phase_clip_id
parent_video_id
clip_filename
video_path
phase_id
phase_name
step_number
step_title
clip_duration_seconds
duration_sec
previous_phase
next_phase
context_bucket
site
subclip_order
phase_occurrence
visual_description
instruments
anatomy
source_filename
source_path
dataset_path
materialization
quality_status
teacher_model
critic_model
```

`parent_video_url` is `null` because phase procedures do not have YouTube
URLs. The procedure identifier is stored in `parent_video_id`, and the neutral
training path is stored in `video_path`.

## SFT JSONL

Each accepted clip receives one SFT record. Its training-facing field names
match the existing dataset:

```text
messages
messages[].role
messages[].content
messages[].content[].type
messages[].content[].video
messages[].content[].text
```

The video reference is relative to the phase dataset root:

```text
PH_0004_2934_S2/clip_01.mp4
```

The student-facing SFT prompt does not contain the gold phase label. The
assistant content is the final critic-approved visual observation.

## Resume and Statuses

The progress log is append-only and uses the latest status for each
`record_id`. Terminal statuses are:

| Status | Meaning |
|---|---|
| `SUCCESS` | Teacher and critic approved the clip and output files exist |
| `REJECTED_IDLE_BOUNDARY` | Active surgery was visible in a candidate Idle clip |

Other statuses indicate a retryable or input/output problem:

| Status | Meaning |
|---|---|
| `MISSING_SOURCE` | The manifest record has no readable source MP4 |
| `ERROR` | API, media, validation, or output failure |

Review checkpoints are stored under:

```text
phase_vlm_dataset/.work/reviews/{split}/{parent_video_id}/{sample_id}.json
```

Every review attempt is also saved in the same directory:

```text
teacher_attempt_01.json
critic_attempt_01.json
teacher_attempt_02.json
critic_attempt_02.json
```

Each response file contains the stage name, prompt set, attempt number, input
mode, frame count, raw model response, parsed response, and any error. If the
critic is not called because the teacher request fails or the teacher response
cannot be validated, a `critic_attempt_XX.json` file is still written with
status `NOT_CALLED` and the reason.

The final checkpoint is written for successful, rejected, and retryable
outcomes. Retryable `ERROR` checkpoints are informational only and do not cause
the record to be skipped on the next run.

A successful or Idle-rejected checkpoint can be reused after an interrupted
run. `--retry-rejected` ignores rejected checkpoints and allows candidate Idle
clips to be reviewed again.

JSONL writes and video materialization use temporary files followed by an
atomic rename. A partially written final record should not be treated as
complete.

## Dry Run and Verification

Check the full manifest without making model calls:

```bash
video_pipeline/.venv/bin/python -m video_pipeline.tools.phase_sft_dataset \
  --config video_pipeline/config.yaml \
  --dry-run
```

The dry run reports selected records, missing source files, and complete
existing outputs. After a small real run, verify that accepted folders contain
an MP4, metadata JSONL, and SFT JSONL; rejected Idle clips contain no training
files; SFT video paths are neutral; and rerunning skips successful records.

## Troubleshooting

Test the local endpoint directly:

```bash
curl --fail-with-body --silent --show-error \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer none' \
  -d '{"model":"qwen3.8-max","messages":[{"role":"user","content":"What is 2 + 2? Answer briefly."}]}'
```

For full-video upstream or payload errors, frame fallback is automatic. For
other errors, inspect the `error` column in `phase_sft_log.csv`. The input mode
used by teacher and critic is also logged.

Frame extraction requires both executables:

```bash
ffmpeg -version
ffprobe -version
```

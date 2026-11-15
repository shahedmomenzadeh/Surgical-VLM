"""Generate grounded SFT data from the selected phase-recognition clips.

The stage deliberately runs before phase GRPO generation. A teacher and a critic
inspect every local video through an OpenAI-compatible chat-completions API. The
video is sent as a base64 data URL, and the final SFT file is written only after
critic approval. Candidate Idle clips are rejected when active surgery is visible.

Rows are reviewed concurrently with a thread pool (default 3 videos at a time)
since the teacher/critic pass is dominated by network-bound API calls, not CPU
work. Only the main thread ever touches the progress log, the run counters, and
the progress bar; each worker thread only reads the shared API clients/settings
and writes to paths that are unique to its own row, so no locking is needed.

Usage:
    python -m video_pipeline.tools.phase_sft_dataset [--config PATH]
    python -m video_pipeline.tools.phase_sft_dataset --dry-run
    python -m video_pipeline.tools.phase_sft_dataset --max-workers 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from ..common.io import ProgressLog, now_str
from ..common.llm import get_openai_client, resolve_and_get_client
from ..config import PhaseSftSettings, build_settings, resolve_path
from .phase_media import (
    encode_video_data_uri,
    extract_frame_data_uris,
    is_frame_fallback_error,
)


LOG_HEADERS = [
    "record_id", "sample_id", "split", "parent_video_id", "phase_id",
    "status", "attempts", "teacher_input_mode", "critic_input_mode",
    "teacher_frames", "critic_frames", "source_path", "output_dir", "error",
    "timestamp",
]
TERMINAL_STATUSES = {"SUCCESS", "REJECTED_IDLE_BOUNDARY"}
PHASE_IDLE = "P13"
VALID_SPLITS = ("Train", "Validation", "Test")
DEFAULT_MAX_WORKERS = 3


def _get_phase_client(settings: Any) -> Any:
    """Build a client from a literal config key or the configured env key."""
    literal_key = settings.extra.get("api_key")
    if literal_key:
        return get_openai_client(literal_key, settings.base_url)
    return resolve_and_get_client(settings.api_key_env, settings.base_url)


ACTIVE_TEACHER_SYSTEM_PROMPT = """You are a cataract-surgery VLM data annotator.

Inspect the attached surgical video and produce a strictly visual, phase-relevant
observation for the verified active surgical phase. Never rely on the phase label
alone and never describe a textbook definition instead of what is visible.

Describe only evidence visible in the video:
- instruments that can actually be seen;
- anatomical structures that can actually be seen;
- surgical actions and observable tissue, fluid, or lens changes.

Do not invent instruments, anatomy, actions, complications, or off-screen events.
Do not mention the metadata, filename, phase label, or this instruction.

Return only one JSON object with this schema:
{
  "decision": "ACCEPT" or "SKIP",
  "visual_description": "A grounded description, required when ACCEPT",
  "instruments": ["Only visibly supported instruments"],
  "anatomy": ["Only visibly supported anatomy"],
  "reason": "Short quality-control explanation"
}"""

ACTIVE_CRITIC_SYSTEM_PROMPT = """You are a strict critic for cataract-surgery VLM data.

Inspect the attached video and the proposed observation for an active surgical
phase. Independently verify every claim using only visible evidence. Correct the
wording when it is mostly valid but overclaims a detail. The final description
must be phase-relevant while remaining grounded in the video, not a generic
definition of the phase.

Reject or remove unsupported claims about instruments, anatomy, actions, tissue
changes, timing, or complications. Do not use the candidate phase label as proof
that an action occurred. A teacher SKIP or REJECT decision is not authoritative;
accept it only if the video supports that decision.

Return ONLY one JSON object with this schema:
{
  "decision": "ACCEPT" or "SKIP",
  "visual_description": "The final grounded description, required when ACCEPT",
  "instruments": ["Only visibly supported instruments"],
  "anatomy": ["Only visibly supported anatomy"],
  "reason": "Short critique or quality-control explanation"
}"""

IDLE_TEACHER_SYSTEM_PROMPT = """You are a cataract-surgery VLM quality reviewer for an Idle phase clip.

Answer one visual question: do you see an active surgeon-controlled instrument
interacting with the eye or ocular structures?

An eyelid speculum, eyelid holder, lid retractor, or the two static holders used
to keep the eye open are support devices. Their presence, contact with the lids,
or continued static positioning does NOT count as active surgery and must NOT by
itself cause rejection. A stationary support device may be mentioned as a
support device, but it is not an active operative instrument.

Reject the clip only when a surgeon is visibly using an instrument to manipulate
the cornea, anterior chamber, iris, capsule, lens, IOL, or other ocular tissue,
or is performing a recognizable surgical maneuver such as incision, injection,
irrigation, aspiration, phacoemulsification, lens positioning, or medication
administration. Mere instrument presence is not sufficient.

If no active surgeon-controlled interaction is visible, accept the clip and
describe the inactivity and any static exposure support without inventing an
operation.

Return ONLY one JSON object with this schema:
{
  "decision": "ACCEPT" or "SKIP",
  "visual_description": "A grounded Idle observation, required when ACCEPT",
  "instruments": ["Visible support devices or active instruments only"],
  "anatomy": ["Only visibly supported anatomy"],
  "reason": "Short explanation of the active-interaction decision"
}"""

IDLE_CRITIC_SYSTEM_PROMPT = """You are an independent cataract-surgery Idle-phase critic.

Inspect the video yourself and decide whether an active surgeon-controlled
instrument is interacting with the eye or ocular structures. Do not treat the
teacher decision or the phase label as proof.

The two eyelid holders, eyelid speculum, or lid retractors used to keep the eye
open are static exposure-support devices. They may touch or hold the eyelids and
must not be classified as active surgery. Their presence alone is not a reason
to reject an Idle clip.

Reject only if the surgeon visibly performs an operative maneuver or actively
manipulates the cornea, anterior chamber, iris, capsule, lens, IOL, or other
ocular tissue. If only static holders and an inactive surgical field are visible,
accept the clip and correct the teacher if it mistakenly rejected the clip.

Return ONLY one JSON object with this schema:
{
  "decision": "ACCEPT" or "SKIP",
  "visual_description": "The final grounded Idle observation, required when ACCEPT",
  "instruments": ["Visible support devices or active instruments only"],
  "anatomy": ["Only visibly supported anatomy"],
  "reason": "Short independent critique"
}"""

ACTIVE_TEACHER_USER_TEMPLATE = """Candidate verified active phase context:
{phase_id} - {phase_name}

Use this label only to focus the observation on visible phase-relevant evidence.
Describe what is visibly happening and do not repeat the label in the final
description.

Return only one JSON object with this schema:
{{
  "decision": "ACCEPT" or "SKIP",
  "visual_description": "A grounded description, required when ACCEPT",
  "instruments": ["Only visibly supported instruments"],
  "anatomy": ["Only visibly supported anatomy"],
  "reason": "Short quality-control explanation"
}}"""

ACTIVE_CRITIC_USER_TEMPLATE = """Candidate verified active phase context:
{phase_id} - {phase_name}

Proposed teacher output:
{draft_json}

Check the attached video independently against this proposal. The teacher may
have returned SKIP or REJECT; correct that decision when the video supports a
grounded observation. Return an accepted, corrected observation or SKIP it when
the evidence is insufficient.

Return ONLY one JSON object with this schema:
{{
  "decision": "ACCEPT" or "SKIP",
  "visual_description": "The final grounded description, required when ACCEPT",
  "instruments": ["Only visibly supported instruments"],
  "anatomy": ["Only visibly supported anatomy"],
  "reason": "Short critique or quality-control explanation"
}}"""

IDLE_TEACHER_USER_TEMPLATE = """Verified phase context: P13 - Idle

Inspect the video specifically for active surgeon-controlled interaction with
the eye. The two eyelid holders or other static exposure-support devices used to
keep the eye open are not active surgical instruments. Accept if those are the
only visible devices and no ocular structure is being actively manipulated;
return SKIP only when active interaction is visible.

Return ONLY one JSON object with this schema:
{{
  "decision": "ACCEPT" or "SKIP",
  "visual_description": "A grounded Idle observation, required when ACCEPT",
  "instruments": ["Visible support devices or active instruments only"],
  "anatomy": ["Only visibly supported anatomy"],
  "reason": "Short explanation of the active-interaction decision"
}}"""

IDLE_CRITIC_USER_TEMPLATE = """Verified phase context: P13 - Idle

Proposed teacher output:
{draft_json}

Independently answer whether an active surgeon-controlled instrument is
interacting with the eye. Do not reject the clip because the two static eyelid
holders/speculum are visible. Accept a genuinely inactive field with static
exposure support; return SKIP only for visible active surgical manipulation.

Return ONLY one JSON object with this schema:
{{
  "decision": "ACCEPT" or "SKIP",
  "visual_description": "The final grounded Idle observation, required when ACCEPT",
  "instruments": ["Visible support devices or active instruments only"],
  "anatomy": ["Only visibly supported anatomy"],
  "reason": "Short independent critique"
}}"""


def _review_prompts(phase_id: str) -> tuple[str, str, str, str, str]:
    """Return a completely separate prompt set for active versus Idle clips."""
    if phase_id == PHASE_IDLE:
        return (
            "idle",
            IDLE_TEACHER_SYSTEM_PROMPT,
            IDLE_TEACHER_USER_TEMPLATE,
            IDLE_CRITIC_SYSTEM_PROMPT,
            IDLE_CRITIC_USER_TEMPLATE,
        )
    return (
        "active",
        ACTIVE_TEACHER_SYSTEM_PROMPT,
        ACTIVE_TEACHER_USER_TEMPLATE,
        ACTIVE_CRITIC_SYSTEM_PROMPT,
        ACTIVE_CRITIC_USER_TEMPLATE,
    )

SFT_USER_TEXT = (
    "Describe the visually observable surgical activity in this cataract surgery "
    "video clip. Focus on visible instruments, anatomy, actions, and observable "
    "changes. Do not infer details that cannot be seen."
)


class ObservationResponse(BaseModel):
    decision: str = "ACCEPT"
    visual_description: str = ""
    instruments: list[str] = Field(default_factory=list)
    anatomy: list[str] = Field(default_factory=list)
    reason: str = ""


@dataclass
class ReviewOutcome:
    status: str
    observation: ObservationResponse | None = None
    reason: str = ""
    attempts: int = 0
    teacher_input_mode: str = ""
    critic_input_mode: str = ""
    teacher_frames: int = 0
    critic_frames: int = 0
    prompt_set: str = ""


@dataclass
class RowResult:
    """Everything the main thread needs to record the outcome of one row.

    Workers build and return this instead of touching the progress log, the
    run counters, or the progress bar directly, since those are only safe to
    mutate from a single thread.
    """
    row: dict[str, str]
    status: str
    counts_key: str
    output_dir: Path
    attempts: int = 0
    error: str = ""
    outcome: ReviewOutcome | None = None


def _model_dump(model: BaseModel) -> dict[str, Any]:
    """Support both Pydantic v1 and v2 environments."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _normalise_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalise_response(payload: Any) -> ObservationResponse:
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object")

    # Accept a few harmless aliases without changing the final dataset schema.
    data = dict(payload)
    if "decision" not in data:
        data["decision"] = data.get("status", "ACCEPT")
    if "visual_description" not in data:
        data["visual_description"] = data.get("description", "")
    data["instruments"] = _normalise_list(data.get("instruments"))
    data["anatomy"] = _normalise_list(data.get("anatomy"))
    data["decision"] = str(data.get("decision", "ACCEPT")).strip().upper()
    data["reason"] = str(data.get("reason", "")).strip()
    data["visual_description"] = str(data.get("visual_description", "")).strip()

    response = ObservationResponse(**data)
    if response.decision not in {"ACCEPT", "SKIP", "REJECT"}:
        raise ValueError(f"Unsupported decision: {response.decision}")
    if response.decision == "ACCEPT" and not response.visual_description:
        raise ValueError("Accepted response has no visual_description")
    return response


JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _find_balanced_json(text: str, start_offset: int = 0) -> str | None:
    """Locate the outermost brace-balanced JSON object at or after
    ``start_offset``, ignoring braces inside strings."""
    start = text.find("{", start_offset)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a dict JSON object from a model response.

    Direct parsing is tried first. The backup extraction (regex and balanced
    brace scan) is used only when the full response is not valid JSON, which
    covers models that prepend an explanation before the JSON object.
    """
    raw = (text or "").strip()

    candidates = [raw]
    fenced = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    if fenced != raw:
        candidates.append(fenced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    # Backup 1: greedy regex over the fence-stripped response.
    match = JSON_OBJECT_PATTERN.search(fenced)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Backup 2: brace-balanced scan. Successive candidate objects are tried
    # so braces inside the explanation itself do not shadow the real JSON.
    offset = 0
    while True:
        balanced = _find_balanced_json(fenced, offset)
        if balanced is None:
            break
        try:
            parsed = json.loads(balanced)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        next_start = fenced.find("{", offset)
        if next_start < 0:
            break
        offset = next_start + 1

    raise ValueError("No JSON object found in model response")


def _message_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif getattr(item, "text", None):
                parts.append(str(item.text))
        return "".join(parts)
    return str(content or "")


def _call_json(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_retries: int,
    retry_delay_seconds: float,
) -> tuple[dict[str, Any] | None, str, bool, str]:
    """Call an OpenAI-compatible endpoint and robustly parse its JSON result."""
    last_error = "unknown model error"
    last_raw_response = ""
    use_response_format = True

    for attempt in range(1, max_retries + 1):
        request = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        if use_response_format:
            request["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**request)
            last_raw_response = _message_text(response)
            payload = _extract_json_object(last_raw_response)
            return payload, "", False, last_raw_response
        except Exception as exc:
            last_error = str(exc)
            lower = last_error.lower()
            if is_frame_fallback_error(last_error):
                # Let the caller retry the request with image frames instead
                # of spending all full-video retries on the same upstream
                # failure.
                return None, last_error, True, last_raw_response
            if use_response_format and (
                "response_format" in lower
                or "json_object" in lower
                or "unsupported" in lower
            ):
                # Some OpenAI-compatible gateways reject response_format even
                # though they support normal chat completions.
                use_response_format = False
            if attempt < max_retries:
                delay = min(retry_delay_seconds * (2 ** (attempt - 1)), 60.0)
                time.sleep(delay)

    return None, last_error, False, last_raw_response


def _video_messages(
    system_prompt: str,
    user_prompt: str,
    data_uri: str,
    video_part_type: str,
) -> list[dict[str, Any]]:
    if video_part_type != "video_url":
        raise ValueError(
            "Only video_part_type='video_url' is supported by the chat-completions "
            "base64 adapter")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": data_uri}},
            {"type": "text", "text": user_prompt},
        ]},
    ]


def _frame_messages(
    system_prompt: str,
    user_prompt: str,
    frame_uris: list[str],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            *(
                {"type": "image_url", "image_url": {"url": uri}}
                for uri in frame_uris
            ),
            {"type": "text", "text": user_prompt},
        ]},
    ]


def _call_json_with_media(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    source_path: Path,
    settings: PhaseSftSettings,
) -> tuple[dict[str, Any] | None, str, str, int, str]:
    """Call with the complete video, falling back to checked video frames."""
    data_uri = encode_video_data_uri(source_path, settings.max_video_bytes)
    payload, error, fallback, raw_response = _call_json(
        client,
        model,
        _video_messages(system_prompt, user_prompt, data_uri, settings.video_part_type),
        settings.max_retries,
        settings.retry_delay_seconds,
    )
    if payload is not None:
        return payload, "full_video", "", 0, raw_response
    if not fallback:
        return None, "full_video", error, 0, raw_response

    frame_uris, actual_frame_count = extract_frame_data_uris(
        source_path,
        max_frames=settings.fallback_max_frames,
        quality=settings.frame_quality,
    )
    payload, error, _, raw_response = _call_json(
        client,
        model,
        _frame_messages(system_prompt, user_prompt, frame_uris),
        settings.max_retries,
        settings.retry_delay_seconds,
    )
    if payload is not None:
        return payload, "frames", "", actual_frame_count, raw_response
    return None, "frames", error, actual_frame_count, raw_response


def _save_review_response(
    review_dir: Path,
    stage: str,
    attempt: int,
    input_mode: str,
    frame_count: int,
    raw_response: str = "",
    parsed_response: dict[str, Any] | None = None,
    error: str = "",
    status: str = "RESPONSE",
    prompt_set: str = "",
) -> None:
    """Persist an inspectable teacher/critic response for one attempt."""
    _atomic_write_jsonl(review_dir / f"{stage}_attempt_{attempt:02d}.json", {
        "stage": stage,
        "prompt_set": prompt_set,
        "attempt": attempt,
        "status": status,
        "input_mode": input_mode,
        "frame_count": frame_count,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "error": error,
    })


def _review_clip(
    row: dict[str, str],
    source_path: Path,
    teacher_client: Any,
    critic_client: Any,
    settings: PhaseSftSettings,
    review_dir: Path,
) -> ReviewOutcome:
    phase_id = row["phase_id"]
    phase_name = row["phase_name"]
    (
        prompt_set,
        teacher_system_prompt,
        teacher_user_template,
        critic_system_prompt,
        critic_user_template,
    ) = _review_prompts(phase_id)
    attempts = 0
    last_reason = ""
    teacher_mode = critic_mode = ""
    teacher_frames = critic_frames = 0

    for attempts in range(1, settings.max_attempts + 1):
        teacher_prompt = teacher_user_template.format(
            phase_id=phase_id, phase_name=phase_name)
        try:
            (
                teacher_payload, teacher_mode, error, teacher_frames,
                teacher_raw_response,
            ) = _call_json_with_media(
                teacher_client,
                settings.teacher.model,
                teacher_system_prompt,
                teacher_prompt,
                source_path,
                settings,
            )
        except Exception as exc:
            teacher_payload = None
            teacher_mode = ""
            teacher_frames = 0
            teacher_raw_response = ""
            error = str(exc)
        _save_review_response(
            review_dir, "teacher", attempts, teacher_mode, teacher_frames,
            raw_response=teacher_raw_response, parsed_response=teacher_payload,
            error=error if teacher_payload is None else "",
            status="RESPONSE" if teacher_payload is not None else "ERROR",
            prompt_set=prompt_set,
        )
        if settings.request_delay_seconds > 0:
            time.sleep(settings.request_delay_seconds)
        if teacher_payload is None:
            _save_review_response(
                review_dir, "critic", attempts, "", 0,
                error="Critic not called because the teacher request did not produce a usable response.",
                status="NOT_CALLED",
                prompt_set=prompt_set,
            )
            last_reason = f"teacher: {error}"
            continue

        try:
            teacher_response = _normalise_response(teacher_payload)
        except (ValidationError, ValueError, TypeError) as exc:
            _save_review_response(
                review_dir, "critic", attempts, "", 0,
                error="Critic not called because the teacher response failed validation.",
                status="NOT_CALLED",
                prompt_set=prompt_set,
            )
            last_reason = f"teacher validation: {exc}"
            continue

        critic_prompt = critic_user_template.format(
            phase_id=phase_id,
            phase_name=phase_name,
            draft_json=json.dumps(_model_dump(teacher_response), ensure_ascii=False),
        )
        try:
            (
                critic_payload, critic_mode, error, critic_frames,
                critic_raw_response,
            ) = _call_json_with_media(
                critic_client,
                settings.critic.model,
                critic_system_prompt,
                critic_prompt,
                source_path,
                settings,
            )
        except Exception as exc:
            critic_payload = None
            critic_mode = ""
            critic_frames = 0
            critic_raw_response = ""
            error = str(exc)
        _save_review_response(
            review_dir, "critic", attempts, critic_mode, critic_frames,
            raw_response=critic_raw_response, parsed_response=critic_payload,
            error=error if critic_payload is None else "",
            status="RESPONSE" if critic_payload is not None else "ERROR",
            prompt_set=prompt_set,
        )
        if settings.request_delay_seconds > 0:
            time.sleep(settings.request_delay_seconds)
        if critic_payload is None:
            last_reason = f"critic: {error}"
            continue

        try:
            critic_response = _normalise_response(critic_payload)
        except (ValidationError, ValueError, TypeError) as exc:
            last_reason = f"critic validation: {exc}"
            continue

        if critic_response.decision != "ACCEPT":
            if phase_id == PHASE_IDLE:
                return ReviewOutcome(
                    "REJECTED_IDLE_BOUNDARY", reason=critic_response.reason,
                    attempts=attempts, teacher_input_mode=teacher_mode,
                    critic_input_mode=critic_mode, teacher_frames=teacher_frames,
                    critic_frames=critic_frames, prompt_set=prompt_set)
            last_reason = f"critic rejected: {critic_response.reason}"
            continue

        return ReviewOutcome(
            "SUCCESS",
            observation=critic_response,
            reason=critic_response.reason,
            attempts=attempts,
            teacher_input_mode=teacher_mode,
            critic_input_mode=critic_mode,
            teacher_frames=teacher_frames,
            critic_frames=critic_frames,
            prompt_set=prompt_set,
        )

    return ReviewOutcome(
        "ERROR", reason=last_reason, attempts=attempts,
        teacher_input_mode=teacher_mode, critic_input_mode=critic_mode,
        teacher_frames=teacher_frames, critic_frames=critic_frames,
        prompt_set=prompt_set,
    )


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {"split", "sample_id", "source_procedure", "phase_id", "phase_name"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    seen = set()
    for row in rows:
        key = f"{row['split']}/{row['source_procedure']}/{row['sample_id']}"
        if key in seen:
            raise ValueError(f"Duplicate manifest sample: {key}")
        seen.add(key)
        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"Unsupported split in manifest: {row['split']}")
    return rows


def _sort_number(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except (TypeError, ValueError):
        return 10**12, value or ""


def _assign_clip_names(rows: list[dict[str, str]]) -> None:
    """Assign stable neutral clip names within each source procedure."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["source_procedure"])].append(row)

    for group_rows in groups.values():
        group_rows.sort(key=lambda item: (_sort_number(item.get("subclip_order", "")), item["sample_id"]))
        for index, row in enumerate(group_rows, 1):
            row["_clip_number"] = str(index)
            row["_clip_filename"] = f"clip_{index:02d}.mp4"


def _source_path(row: dict[str, str], settings: PhaseSftSettings) -> Path:
    dataset_path = row.get("dataset_path", "").strip()
    if dataset_path:
        candidate = Path(resolve_path(dataset_path))
        if candidate.is_file():
            return candidate

    return (
        Path(settings.source_dir)
        / row["split"]
        / row["source_procedure"]
        / f"{row['sample_id']}.mp4"
    )


def _record_id(row: dict[str, str]) -> str:
    return f"{row['split']}/{row['source_procedure']}/{row['sample_id']}"


def _output_dir(row: dict[str, str], settings: PhaseSftSettings) -> Path:
    return Path(settings.output_dir) / row["split"] / row["source_procedure"]


def _is_complete(output_dir: Path, clip_filename: str) -> bool:
    paths = [
        output_dir / clip_filename,
        output_dir / f"{Path(clip_filename).stem}.jsonl",
        output_dir / f"{Path(clip_filename).stem}_sft.jsonl",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _atomic_write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
            prefix=f".{path.name}.", suffix=".tmp") as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def _materialize_video(source: Path, destination: Path, settings: PhaseSftSettings) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    temporary_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary_path.exists() or temporary_path.is_symlink():
        temporary_path.unlink()

    try:
        mode = settings.materialization.lower()
        if mode == "hardlink":
            try:
                os.link(source, temporary_path)
            except OSError:
                if not settings.hardlink_fallback_copy:
                    raise
                shutil.copy2(source, temporary_path)
        elif mode == "copy":
            shutil.copy2(source, temporary_path)
        elif mode == "symlink":
            os.symlink(source.resolve(), temporary_path)
        else:
            raise ValueError(f"Unsupported materialization mode: {settings.materialization}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()


def _build_metadata(
    row: dict[str, str],
    observation: ObservationResponse,
    settings: PhaseSftSettings,
    critic_status: str = "critic_accepted",
) -> dict[str, Any]:
    clip_number = int(row["_clip_number"])
    clip_filename = row["_clip_filename"]
    parent_id = row["source_procedure"]
    video_path = f"{parent_id}/{clip_filename}"

    def optional_float(value: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "clip_id": f"{parent_id}_clip_{clip_number:02d}",
        "phase_clip_id": row["sample_id"],
        "parent_video_id": parent_id,
        "clip_filename": clip_filename,
        "video_path": video_path,
        "parent_video_title": f"Phase Recognition procedure {parent_id}",
        "parent_video_url": None,
        "step_number": clip_number,
        "clip_duration_seconds": optional_float(row.get("duration_sec", "")),
        "timestamp_start_in_parent": None,
        "timestamp_end_in_parent": None,
        "step_title": row["phase_name"],
        "visual_description": observation.visual_description,
        "transcript_context": "",
        "instruments": observation.instruments,
        "anatomy": observation.anatomy,
        "phase_id": row["phase_id"],
        "phase_name": row["phase_name"],
        "site": row.get("site", ""),
        "subclip_order": row.get("subclip_order", ""),
        "phase_occurrence": row.get("phase_occurrence", ""),
        "duration_sec": optional_float(row.get("duration_sec", "")),
        "previous_phase": row.get("previous_phase", ""),
        "next_phase": row.get("next_phase", ""),
        "context_bucket": row.get("context_bucket", ""),
        "source_filename": f"{row['sample_id']}.mp4",
        "source_path": row.get("source_path", ""),
        "dataset_path": row.get("dataset_path", ""),
        "materialization": settings.materialization,
        "quality_status": critic_status,
        "teacher_model": settings.teacher.model,
        "critic_model": settings.critic.model,
    }


def _build_sft_record(video_path: str, visual_description: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": SFT_USER_TEXT},
            ]},
            {"role": "assistant", "content": visual_description},
        ]
    }


def _append_log(log: ProgressLog, row: dict[str, str], status: str, attempts: int,
                output_dir: Path, error: str = "", outcome: ReviewOutcome | None = None) -> None:
    outcome = outcome or ReviewOutcome(status=status, attempts=attempts)
    log.append(
        record_id=_record_id(row),
        sample_id=row["sample_id"],
        split=row["split"],
        parent_video_id=row["source_procedure"],
        phase_id=row["phase_id"],
        status=status,
        attempts=attempts,
        teacher_input_mode=outcome.teacher_input_mode,
        critic_input_mode=outcome.critic_input_mode,
        teacher_frames=outcome.teacher_frames,
        critic_frames=outcome.critic_frames,
        source_path=row.get("dataset_path", ""),
        output_dir=str(output_dir),
        error=error[:4000],
        timestamp=now_str(),
    )


def _ensure_log_schema(path: Path) -> None:
    """Upgrade an older phase log without discarding resumability data."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == LOG_HEADERS:
            return
        rows = list(reader)

    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in LOG_HEADERS})
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _checkpoint_path(row: dict[str, str], output_root: Path) -> Path:
    return (
        output_root / ".work" / "reviews" / row["split"]
        / row["source_procedure"] / f"{row['sample_id']}.json"
    )


def _write_review_checkpoint(path: Path, outcome: ReviewOutcome) -> None:
    observation = _model_dump(outcome.observation) if outcome.observation else None
    _atomic_write_jsonl(path, {
        "status": outcome.status,
        "observation": observation,
        "reason": outcome.reason,
        "attempts": outcome.attempts,
        "teacher_input_mode": outcome.teacher_input_mode,
        "critic_input_mode": outcome.critic_input_mode,
        "teacher_frames": outcome.teacher_frames,
        "critic_frames": outcome.critic_frames,
        "prompt_set": outcome.prompt_set,
    })


def _load_review_checkpoint(path: Path) -> ReviewOutcome | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.loads(handle.readline())
        observation = data.get("observation")
        return ReviewOutcome(
            status=data.get("status", "ERROR"),
            observation=ObservationResponse(**observation) if observation else None,
            reason=data.get("reason", ""),
            attempts=int(data.get("attempts", 0)),
            teacher_input_mode=data.get("teacher_input_mode", ""),
            critic_input_mode=data.get("critic_input_mode", ""),
            teacher_frames=int(data.get("teacher_frames", 0)),
            critic_frames=int(data.get("critic_frames", 0)),
            prompt_set=data.get("prompt_set", ""),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return None


def _process_row(
    row: dict[str, str],
    output_root: Path,
    teacher_client: Any,
    critic_client: Any,
    settings: PhaseSftSettings,
    retry_rejected: bool,
) -> RowResult:
    """Run the full review + write pipeline for a single manifest row.

    This is the unit of work submitted to the thread pool, so it must stay
    self-contained: it only reads the shared clients/settings and writes to
    paths derived from this row's own sample_id (checkpoint file, materialized
    clip, metadata jsonl, sft jsonl), none of which any other row touches. The
    caller is responsible for logging the returned RowResult and updating the
    run counters on the main thread.
    """
    output_dir = _output_dir(row, settings)
    clip_path = output_dir / row["_clip_filename"]

    if _is_complete(output_dir, row["_clip_filename"]):
        return RowResult(row, "SUCCESS", "skipped", output_dir)

    source_path = _source_path(row, settings)
    if not source_path.is_file():
        return RowResult(row, "MISSING_SOURCE", "errors", output_dir, error=str(source_path))

    checkpoint = _checkpoint_path(row, output_root)
    outcome = None
    if not retry_rejected:
        saved_outcome = _load_review_checkpoint(checkpoint)
        # A rejected checkpoint is not reused here. The progress log controls
        # terminal rejection skipping, so converting a rejection to ERROR (or
        # using --retry-rejected) must force a fresh teacher/critic review.
        if saved_outcome and saved_outcome.status == "SUCCESS":
            outcome = saved_outcome

    if outcome is None:
        try:
            outcome = _review_clip(
                row, source_path, teacher_client, critic_client, settings,
                checkpoint.parent)
        except Exception as exc:
            outcome = ReviewOutcome("ERROR", reason=str(exc), attempts=0)
        # Keep the final outcome for inspection even when it is retryable.
        # Retry logic only skips statuses in TERMINAL_STATUSES.
        _write_review_checkpoint(checkpoint, outcome)

    if outcome.status == "REJECTED_IDLE_BOUNDARY":
        return RowResult(row, outcome.status, "rejected_idle", output_dir,
                          attempts=outcome.attempts, error=outcome.reason, outcome=outcome)
    if outcome.status != "SUCCESS" or not outcome.observation:
        return RowResult(row, "ERROR", "errors", output_dir,
                          attempts=outcome.attempts, error=outcome.reason, outcome=outcome)

    try:
        _materialize_video(source_path, clip_path, settings)
        metadata = _build_metadata(row, outcome.observation, settings)
        sft_record = _build_sft_record(
            metadata["video_path"], outcome.observation.visual_description)
        stem = Path(row["_clip_filename"]).stem
        _atomic_write_jsonl(output_dir / f"{stem}.jsonl", metadata)
        _atomic_write_jsonl(output_dir / f"{stem}_sft.jsonl", sft_record)
        return RowResult(row, "SUCCESS", "success", output_dir,
                          attempts=outcome.attempts, outcome=outcome)
    except Exception as exc:
        return RowResult(row, "ERROR", "errors", output_dir,
                          attempts=outcome.attempts, error=f"write: {exc}", outcome=outcome)


def run_phase_sft(
    settings: PhaseSftSettings,
    split: str | None = None,
    limit: int | None = None,
    retry_rejected: bool = False,
    dry_run: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, int]:
    rows = _read_manifest(Path(settings.manifest_file))
    _assign_clip_names(rows)
    rows.sort(key=lambda item: (
        VALID_SPLITS.index(item["split"]),
        item["source_procedure"],
        int(item["_clip_number"]),
    ))

    if split:
        rows = [row for row in rows if row["split"] == split]
    if limit is not None:
        rows = rows[:limit]

    output_root = Path(settings.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_log_schema(Path(settings.log_file))
    progress_log = ProgressLog(str(settings.log_file), LOG_HEADERS)
    completed = progress_log.read_processed(
        TERMINAL_STATUSES if not retry_rejected else {"SUCCESS"})

    counts = {"success": 0, "rejected_idle": 0, "errors": 0, "skipped": 0}
    print(f"Found {len(rows)} phase samples to inspect.")
    print(f"Output: {output_root}")

    if dry_run:
        missing = sum(not _source_path(row, settings).is_file() for row in rows)
        complete = sum(
            _is_complete(_output_dir(row, settings), row["_clip_filename"])
            for row in rows
        )
        print(f"Source files missing: {missing}")
        print(f"Complete output samples: {complete}")
        return counts

    teacher_client = _get_phase_client(settings.teacher)
    critic_client = _get_phase_client(settings.critic)
    if not teacher_client or not critic_client:
        return {"success": 0, "rejected_idle": 0, "errors": len(rows), "skipped": 0}

    # Rows the log already marks terminal are dropped up front, so the thread
    # pool only ever sees rows that genuinely need a teacher/critic call.
    to_process = []
    for row in rows:
        if _record_id(row) in completed:
            counts["skipped"] += 1
            continue
        to_process.append(row)

    max_workers = max(1, max_workers)
    print(f"Skipping {counts['skipped']} already-completed samples.")
    print(f"Reviewing {len(to_process)} samples with {max_workers} concurrent workers.")

    worker = partial(
        _process_row,
        output_root=output_root,
        teacher_client=teacher_client,
        critic_client=critic_client,
        settings=settings,
        retry_rejected=retry_rejected,
    )

    # Every _process_row call runs in its own worker thread, but the progress
    # log, the counts dict, and the progress bar are only ever touched here on
    # the main thread as each future completes, so none of them need a lock.
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="phase-sft") as executor:
        futures = {executor.submit(worker, row): row for row in to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating phase SFT"):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = RowResult(row, "ERROR", "errors", _output_dir(row, settings),
                                    error=f"worker crashed: {exc}")

            _append_log(progress_log, result.row, result.status, result.attempts,
                        result.output_dir, result.error, result.outcome)
            counts[result.counts_key] += 1

    print(
        "Completed: "
        f"success={counts['success']}, "
        f"idle_rejected={counts['rejected_idle']}, "
        f"errors={counts['errors']}, "
        f"skipped={counts['skipped']}"
    )
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate critic-approved phase SFT data from local video clips.")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: bundled config or VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--split", choices=VALID_SPLITS, default=None,
                        help="Process only one dataset split")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N manifest rows after filtering")
    parser.add_argument("--retry-rejected", action="store_true",
                        help="Retry previously rejected Idle clips")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate paths and report existing outputs without API calls")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Number of videos to review concurrently (default: {DEFAULT_MAX_WORKERS})")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config).phase_sft
    run_phase_sft(
        settings,
        split=args.split,
        limit=args.limit,
        retry_rejected=args.retry_rejected,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()

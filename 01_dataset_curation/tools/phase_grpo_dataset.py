"""Generate deterministic phase GRPO records from accepted phase SFT clips.

This stage makes no LLM/VLM/API calls. It reads critic-approved phase SFT
folders, reuses the validated visual observation as ``reference_reasoning``,
and creates one 13-class surgical-ontology question per accepted clip.

Usage:
    python -m video_pipeline.tools.phase_grpo_dataset [--config PATH]
    python -m video_pipeline.tools.phase_grpo_dataset --dry-run
    python -m video_pipeline.tools.phase_grpo_dataset --split Validation
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..common.io import ProgressLog, now_str
from ..config import PhaseGrpoSettings, build_settings


VALID_SPLITS = ("Train", "Validation", "Test")
LOG_HEADERS = [
    "record_id", "split", "parent_video_id", "clip_filename", "phase_id",
    "status", "error", "timestamp",
]
TERMINAL_STATUSES = {"SUCCESS"}

PHASE_ONTOLOGY = (
    ("P01", "Incision", "Initial corneal incision"),
    ("P02", "Viscoelastic", "Injection of viscoelastic material"),
    ("P03", "Capsulorhexis", "Continuous circular anterior capsulotomy"),
    ("P04", "Hydrodissection", "Fluid separation and lens mobilization"),
    ("P05", "Phacoemulsification", "Ultrasonic lens-nucleus fragmentation and removal"),
    ("P06", "Irrigation-Aspiration", "Removal of cortex or residual lens material"),
    ("P07", "Capsule Polishing", "Cleaning or polishing the lens capsule"),
    ("P08", "Lens Implantation", "Insertion of the intraocular lens"),
    ("P09", "Lens Positioning", "Final positioning of the intraocular lens"),
    ("P10", "Viscoelastic Suction", "Removal of viscoelastic material"),
    ("P11", "Anterior Chamber Flushing", "Flushing or irrigating the anterior chamber"),
    ("P12", "Tonifying-Antibiotics", "Medication or antibiotic administration"),
    ("P13", "Idle", "No active surgical action or inactive time"),
)
PHASE_IDS = {phase_id for phase_id, _, _ in PHASE_ONTOLOGY}
ONTOLOGY_TEXT = "\n".join(
    f"{phase_id} - {phase_name}: {description}"
    for phase_id, phase_name, description in PHASE_ONTOLOGY
)

PHASE_GRPO_USER_TEXT = f"""Given this video, identify the surgical phase from the surgical ontology.

Surgical ontology:
{ONTOLOGY_TEXT}

Use only visible evidence from the video. Provide your reasoning first, then
state the final ontology ID on its own line using exactly this format:
Final answer: PXX"""


@dataclass
class ClipCandidate:
    split: str
    parent_video_id: str
    clip_filename: str
    metadata_path: Path
    video_path: Path
    sft_path: Path
    metadata: dict[str, Any]

    @property
    def record_id(self) -> str:
        return f"{self.split}/{self.parent_video_id}/{self.clip_filename}"

    @property
    def grpo_path(self) -> Path:
        return self.metadata_path.with_name(
            f"{self.metadata_path.stem}_grpo.jsonl")


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_core_metadata(path: Path) -> dict[str, Any]:
    records = _read_jsonl_records(path)
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("core metadata must contain exactly one JSON object")
    return records[0]


def _collect_candidates(input_dir: Path, split: str | None = None) -> tuple[list[ClipCandidate], list[str]]:
    """Collect complete accepted-SFT clip folders and report skipped paths."""
    candidates = []
    skipped = []
    splits = (split,) if split else VALID_SPLITS

    for split_name in splits:
        split_dir = input_dir / split_name
        if not split_dir.is_dir():
            skipped.append(f"missing split directory: {split_dir}")
            continue
        for procedure_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for metadata_path in sorted(procedure_dir.glob("clip_*.jsonl")):
                if metadata_path.name.endswith("_sft.jsonl") or metadata_path.name.endswith("_grpo.jsonl"):
                    continue
                stem = metadata_path.stem
                video_path = procedure_dir / f"{stem}.mp4"
                sft_path = procedure_dir / f"{stem}_sft.jsonl"
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    skipped.append(f"{split_name}/{procedure_dir.name}/{stem}: missing video")
                    continue
                if not sft_path.is_file() or sft_path.stat().st_size == 0:
                    skipped.append(f"{split_name}/{procedure_dir.name}/{stem}: missing SFT")
                    continue
                try:
                    metadata = _read_core_metadata(metadata_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    skipped.append(f"{metadata_path}: invalid metadata ({exc})")
                    continue

                phase_id = str(metadata.get("phase_id", "")).strip()
                if phase_id not in PHASE_IDS:
                    skipped.append(f"{metadata_path}: invalid phase_id={phase_id!r}")
                    continue
                candidates.append(ClipCandidate(
                    split=split_name,
                    parent_video_id=procedure_dir.name,
                    clip_filename=video_path.name,
                    metadata_path=metadata_path,
                    video_path=video_path,
                    sft_path=sft_path,
                    metadata=metadata,
                ))

    candidates.sort(key=lambda item: (
        VALID_SPLITS.index(item.split), item.parent_video_id, item.clip_filename))
    return candidates, skipped


def _build_grpo_record(candidate: ClipCandidate) -> dict[str, Any]:
    metadata = candidate.metadata
    phase_id = str(metadata["phase_id"]).strip()
    video_path = metadata.get("video_path") or (
        f"{candidate.parent_video_id}/{candidate.clip_filename}"
    )
    visual_description = str(metadata.get("visual_description", "")).strip()
    if not visual_description:
        raise ValueError("metadata has no critic-approved visual_description")

    return {
        "prompt": [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": PHASE_GRPO_USER_TEXT},
            ],
        }],
        "correct_answer": phase_id,
        "question_type": "phase_identification",
        "reference_reasoning": visual_description,
        "reward_type": "deterministic",
    }


def _is_valid_grpo_file(path: Path) -> bool:
    try:
        records = _read_jsonl_records(path)
        if len(records) != 1:
            return False
        record = records[0]
        required = {"prompt", "correct_answer", "question_type",
                    "reference_reasoning", "reward_type"}
        return (
            required.issubset(record)
            and record["question_type"] == "phase_identification"
            and record["reward_type"] == "deterministic"
            and record["correct_answer"] in PHASE_IDS
            and isinstance(record["reference_reasoning"], str)
            and isinstance(record["prompt"], list)
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return False


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


def _ensure_log_schema(path: Path) -> None:
    """Upgrade an older GRPO log while retaining its history."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    import csv
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


def _append_log(log: ProgressLog, candidate: ClipCandidate, status: str, error: str = "") -> None:
    log.append(
        record_id=candidate.record_id,
        split=candidate.split,
        parent_video_id=candidate.parent_video_id,
        clip_filename=candidate.clip_filename,
        phase_id=candidate.metadata.get("phase_id", ""),
        status=status,
        error=error[:4000],
        timestamp=now_str(),
    )


def run_phase_grpo(
    settings: PhaseGrpoSettings,
    split: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    input_dir = Path(settings.input_dir)
    candidates, skipped_sources = _collect_candidates(input_dir, split)
    if limit is not None:
        candidates = candidates[:limit]

    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_log_schema(log_path)
    progress_log = ProgressLog(str(log_path), LOG_HEADERS)
    completed = progress_log.read_processed(TERMINAL_STATUSES)

    counts = {
        "generated": 0,
        "skipped": 0,
        "errors": 0,
        "incomplete_sources": len(skipped_sources),
    }
    print(f"Found {len(candidates)} complete accepted SFT clips.")
    print(f"Skipped source records: {len(skipped_sources)}")
    print(f"Input: {input_dir}")

    if dry_run:
        existing = sum(_is_valid_grpo_file(candidate.grpo_path) for candidate in candidates)
        print(f"Valid existing GRPO files: {existing}")
        print(f"Would generate: {len(candidates) - existing}")
        return counts

    for candidate in tqdm(candidates, desc="Generating phase GRPO"):
        if candidate.record_id in completed:
            counts["skipped"] += 1
            continue
        if _is_valid_grpo_file(candidate.grpo_path):
            _append_log(progress_log, candidate, "SUCCESS")
            counts["skipped"] += 1
            continue
        try:
            record = _build_grpo_record(candidate)
            _atomic_write_jsonl(candidate.grpo_path, record)
            _append_log(progress_log, candidate, "SUCCESS")
            counts["generated"] += 1
        except (OSError, ValueError, TypeError, KeyError) as exc:
            _append_log(progress_log, candidate, "ERROR", str(exc))
            counts["errors"] += 1

    print(
        "Completed: "
        f"generated={counts['generated']}, "
        f"skipped={counts['skipped']}, "
        f"errors={counts['errors']}, "
        f"incomplete_sources={counts['incomplete_sources']}"
    )
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic phase GRPO data from accepted phase SFT clips.")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: bundled config or VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--split", choices=VALID_SPLITS, default=None,
                        help="Process only one dataset split")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N eligible clips after split filtering")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect inputs and existing outputs without writing files")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config).phase_grpo
    run_phase_grpo(
        settings,
        split=args.split,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

"""Merge grpo_dataset into the ./dataset layout (YouTube-track GRPO schema).

Split policy:
    The split stored in grpo_dataset (annotations['split'] and splits.json)
    is IGNORED. Every sample is assigned to the split of its parent PH_*
    procedure as it exists in ./dataset ({Train,Validation,Test}/PH_*).

Schema conversion:
    Each record is converted to the legacy dataset GRPO schema, which has
    exactly these keys:

        prompt / correct_answer / question_type / reference_reasoning / reward_type

    - task          -> question_type  (they are identical in every record)
    - reward_type    kept ("deterministic")
    - reference_reasoning kept (null -> optionally synthesized from target)
    - all provenance (sample_id, video, source, target, window, frames, split)
      is preserved under `metadata` (extended mode) or dropped (strict mode).

Layout produced (mirrors the clip_XX.mp4 / clip_XX_grpo.jsonl convention):

    <out>/
    ├── procedure_split_map.json          # {procedure_id: split} derived from ./dataset
    ├── Train/…/Test/
    │   └── PH_XXXX_….S* /
    │       ├── <sample_id>.mp4           # e.g. grpo_0568_temporal_localization.mp4
    │       └── <sample_id>_grpo.jsonl    # one converted record per sample

Usage:
    python3 tools/merge_grpo_into_dataset.py --dry-run
    python3 tools/merge_grpo_into_dataset.py --jsonl-only
    python3 tools/merge_grpo_into_dataset.py --link-videos
    python3 tools/merge_grpo_into_dataset.py --copy-videos
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SPLITS = ("Train", "Validation", "Test")
CORE_KEYS = ("prompt", "correct_answer", "question_type",
             "reference_reasoning", "reward_type")

REASONING_TEMPLATES = {
    "timestamp_to_phase": "At {t:.1f} s into the clip the visible activity "
                          "corresponds to {label}; the queried instant falls "
                          "inside occurrence {occ} of {name} "
                          "({vs:.1f}-{ve:.1f} s clip-local).",
    "contextual_phase_recognition": "The portion after the inactive interval "
                                    "shows {name} ({phase}), visible at "
                                    "clip-local {vs:.1f}-{ve:.1f} s.",
    "temporal_localization": "{name} ({phase}) occupies clip-local "
                             "{vs:.1f}-{ve:.1f} s in this clip.",
    "boundary_detection": "{name} ({phase}) {verb} at {ans:.1f} s clip-local, "
                          "inside an interval visible at "
                          "{vs:.1f}-{ve:.1f} s.",
}


def load_dataset_split_map(dataset_dir: Path) -> dict[str, str]:
    """Map each PH_* procedure folder to its split in ./dataset."""
    mapping = {}
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue
        for entry in sorted(split_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("PH_"):
                mapping[entry.name] = split
    return mapping


def synthesize_reasoning(record: dict) -> str | None:
    task = record.get("task") or record.get("question_type")
    target = record.get("target") or {}
    template = REASONING_TEMPLATES.get(task)
    if template is None or not target:
        return None
    name = target.get("phase_name") or ""
    phase = target.get("phase") or ""
    occ = target.get("occurrence")
    vs = target.get("visible_start_clip")
    ve = target.get("visible_end_clip")
    kwargs = dict(name=name, phase=phase, occ=occ, vs=vs, ve=ve)
    if task == "timestamp_to_phase":
        kwargs["t"] = target.get("queried_timestamp_clip")
        kwargs["label"] = target.get("phase_at_queried_label") or f"{phase} {name}".strip()
    elif task == "boundary_detection":
        question = record["prompt"][0]["content"][1]["text"].lower()
        kwargs["verb"] = "ends" if "end" in question else "begins"
        ca = record.get("correct_answer")
        if isinstance(ca, dict):
            kwargs["ans"] = ca.get("timestamp")
        else:
            kwargs["ans"] = ca
    elif task == "temporal_localization":
        ca = record.get("correct_answer") or {}
        if isinstance(ca, dict):
            kwargs["vs"], kwargs["ve"] = ca.get("start"), ca.get("end")
    if any(v is None for v in kwargs.values()):
        return None
    return template.format(**kwargs)


def convert_record(record: dict, mode: str = "extended",
                   synthesize: bool = False) -> dict:
    """Transform one grpo_dataset record into the legacy GRPO schema."""
    prompt = json.loads(json.dumps(record["prompt"]))  # deep copy
    video_name = record["video"].rsplit("/", 1)[-1]
    proc = record["source"]["procedure_id"]
    for content in prompt[0]["content"]:
        if content.get("type") == "video":
            # dataset convention: {parent_folder}/{file}, split-relative
            content["video"] = f"{proc}/{video_name}"

    reference = record.get("reference_reasoning")
    if synthesize and reference is None:
        reference = synthesize_reasoning(record)

    converted = {
        "prompt": prompt,
        "correct_answer": record["correct_answer"],
        "question_type": record["task"],
        "reference_reasoning": reference,
        "reward_type": record.get("reward_type", "deterministic"),
    }
    if mode == "extended":
        converted["metadata"] = {
            "sample_id": record["sample_id"],
            "video": record["video"],
            "source": record["source"],
            "target": record["target"],
            "window": record.get("window"),
            "frames": record.get("frames"),
            "grpo_split_ignored": record.get("split"),
        }
    return converted


def verify_staged(staged: Path, expected: dict[str, list[dict]]) -> dict:
    """Check staged JSONLs against the annotation-derived expectation."""
    issues = []
    ok = 0
    for split, records in expected.items():
        for rec in records:
            sample_id = rec["sample_id"]
            proc = rec["source"]["procedure_id"]
            jsonl = staged / split / proc / f"{sample_id}_grpo.jsonl"
            if not jsonl.is_file():
                issues.append(f"missing {jsonl.relative_to(staged)}")
                continue
            data = json.loads(jsonl.read_text(encoding="utf-8"))
            if tuple(k for k in CORE_KEYS if k in data) != CORE_KEYS:
                issues.append(f"bad keys {jsonl.relative_to(staged)}")
                continue
            ref = data["prompt"][0]["content"][0]["video"]
            if ref != f"{proc}/{sample_id}.mp4":
                issues.append(f"bad video ref {jsonl.relative_to(staged)}: {ref}")
                continue
            ok += 1
    return {"verified": ok, "issues": issues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge phase_grpo_dataset into dataset_grpo layout, honoring "
                    "parent-procedure splits and the legacy GRPO schema.")
    parser.add_argument("--dataset-dir", default="dataset_grpo",
                        help="dataset_grpo (or dataset_sft) root used to derive PH_* splits (default: dataset_grpo)")
    parser.add_argument("--grpo-dir", default="phase_grpo_dataset",
                        help="phase_grpo_dataset root (annotations.jsonl) (default: phase_grpo_dataset)")
    parser.add_argument("--out", default="dataset_grpo",
                        help="target output dataset_grpo root (default: dataset_grpo)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--jsonl-only", action="store_true",
                      help="write converted JSONLs + split map, skip videos")
    mode.add_argument("--link-videos", action="store_true", default=True,
                      help="copy videos via hardlink (fallback: copy2) (default)")
    mode.add_argument("--copy-videos", action="store_true",
                      help="copy videos with shutil.copy2")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="emit only the 5 standard keys (prompt, correct_answer, question_type, reference_reasoning, reward_type) (default: True)")
    parser.add_argument("--extended", action="store_true",
                        help="emit extended metadata")
    parser.add_argument("--synthesize-reasoning", action="store_true", default=True,
                        help="fill null reference_reasoning from target fields (default: True)")
    parser.add_argument("--no-synthesize-reasoning", dest="synthesize_reasoning", action="store_false",
                        help="leave reference_reasoning as null")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without writing")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    grpo_dir = Path(args.grpo_dir).resolve()
    out = Path(args.out).resolve() if args.out else (grpo_dir / "merged_dataset")

    split_map = load_dataset_split_map(dataset_dir)
    if not split_map:
        print(f"❌ No PH_* procedure folders found under {dataset_dir}")
        return 1

    ann_path = grpo_dir / "annotations.jsonl"
    records = [json.loads(line) for line in
               ann_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(records)} records from {ann_path}")
    print(f"PH_* procedure split map from {dataset_dir}: {len(split_map)} procedures")

    # ---- assign splits by parent procedure (ignore grpo 'split') ----
    assigned = {}
    missing = set()
    per_split = {s: [] for s in SPLITS}
    for rec in records:
        proc = rec["source"]["procedure_id"]
        split = split_map.get(proc)
        if split is None:
            missing.add(proc)
            continue
        per_split[split].append(rec)
        assigned[rec["sample_id"]] = split

    old_dist = {}
    for rec in records:
        old_dist[rec["split"]] = old_dist.get(rec["split"], 0) + 1
    new_dist = {s: len(v) for s, v in per_split.items()}

    print("\nSplit distribution:")
    print(f"  grpo_dataset original  : " +
          ", ".join(f"{k}={v}" for k, v in sorted(old_dist.items())))
    print(f"  parent-procedure based : " +
          ", ".join(f"{k}={v}" for k, v in new_dist.items()))
    moved = sum(
        1 for rec in records
        if assigned.get(rec["sample_id"])
        and {"train": "Train", "val": "Validation", "test": "Test"}[rec["split"]]
        != assigned[rec["sample_id"]]
    )
    print(f"  samples moved across splits: {moved}")
    if missing:
        print(f"⚠️ {len(missing)} sample(s) skipped (no PH_ folder in ./dataset): "
              f"{sorted(missing)[:5]}")
        return 1

    if args.dry_run:
        print("\nDRY-RUN: nothing written.")
        print(f"Would stage into {out} "
              f"({'videos' if (args.copy_videos or args.link_videos) else 'jsonl only'})")
        return 0

    # ---- write split map ----
    out.mkdir(parents=True, exist_ok=True)
    (out / "procedure_split_map.json").write_text(
        json.dumps(split_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- convert + write ----
    converted_total = 0
    for split, recs in per_split.items():
        for rec in recs:
            proc = rec["source"]["procedure_id"]
            sample_id = rec["sample_id"]
            folder = out / split / proc
            folder.mkdir(parents=True, exist_ok=True)

            converted = convert_record(
                rec, mode="extended" if args.extended else "strict",
                synthesize=args.synthesize_reasoning)

            (folder / f"{sample_id}_grpo.jsonl").write_text(
                json.dumps(converted, ensure_ascii=False) + "\n", encoding="utf-8")
            converted_total += 1

            if args.copy_videos or args.link_videos:
                src = grpo_dir / rec["video"]
                dst = folder / f"{sample_id}.mp4"
                if dst.exists():
                    continue
                if args.link_videos:
                    try:
                        os.link(src, dst)
                        continue
                    except OSError:
                        pass  # fall back to copy2
                shutil.copy2(src, dst)

    print(f"\n✅ Converted {converted_total} records -> {out}")

    videos_copied = sum(1 for _ in out.rglob("*.mp4"))
    print(f"Videos staged: {videos_copied} / {len(records)}")
    check = verify_staged(out, per_split)
    print(f"Verification: {check['verified']} JSONLs OK, "
          f"{len(check['issues'])} issues")
    for issue in check["issues"][:10]:
        print(" ⚠️", issue)
    return 0 if not check["issues"] else 2


if __name__ == "__main__":
    sys.exit(main())
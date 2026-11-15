"""
Temporary migration: mirror ./dataset into ./dataset_sft and ./dataset_grpo.

- Videos (clip_*.mp4 + full_video.mp4) are hardlinked (fallback copy) into BOTH datasets.
- clip_*.jsonl (metadata, not sft/grpo) → BOTH
- *_sft.jsonl + full_video_sft.jsonl → only dataset_sft
- *_grpo.jsonl (YT) → only dataset_grpo, with fix:
    * reward_type llm_judge → deterministic
    * prompt suffix "Provide your reasoning first..." → JSON instruction
- archived/, grpo_archived/, venv/, __pycache__ etc. are NOT copied
- grpo_procedure_split_map.json → BOTH roots
- Does NOT modify ./dataset

Usage:
    python -m video_pipeline.tools.build_separated_datasets
    python -m video_pipeline.tools.build_separated_datasets --src dataset --dst-sft dataset_sft --dst-grpo dataset_grpo --materialization hardlink
"""
import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# New JSON instruction to replace the old suffix
OLD_SUFFIX = "Provide your reasoning first, then state your answer."
NEW_SUFFIX = (
    'Respond ONLY with a JSON object with two keys: "explanation" and "answer". '
    '"explanation" is 1-3 sentences of reasoning describing what is visible (natural language, no field names). '
    '"answer" is the single letter A, B, C or D (nothing more). '
    'Example: {"explanation": "In the video the surgeon uses a keratome to create a corneal incision.", "answer": "B"}'
)

ARCHIVED_DIRS = {"archived", "grpo_archived", "venv", "__pycache__", ".venv"}
SPLITS = ["Train", "Validation", "Test"]


def _materialize(src: Path, dst: Path, mode: str = "hardlink", fallback_copy: bool = True):
    """Hardlink or copy src -> dst atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # if dst already exists with same size, skip; otherwise overwrite
        try:
            if dst.stat().st_size == src.stat().st_size:
                return
        except Exception:
            pass
        dst.unlink()
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    try:
        if mode == "hardlink":
            try:
                os.link(src, tmp)
            except OSError as e:
                if fallback_copy:
                    shutil.copy2(src, tmp)
                else:
                    raise
        elif mode == "copy":
            shutil.copy2(src, tmp)
        elif mode == "symlink":
            tmp.symlink_to(src)
        else:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _transform_grpo_line(line: str):
    """Fix reward_type + prompt suffix for a single GRPO JSONL line."""
    rec = json.loads(line)
    changed = False
    # 1. reward_type llm_judge -> deterministic
    if rec.get("reward_type") == "llm_judge":
        rec["reward_type"] = "deterministic"
        changed = True
    # 2. prompt text suffix replacement
    try:
        prompt = rec.get("prompt", [])
        if prompt and isinstance(prompt[0], dict):
            content = prompt[0].get("content", [])
            # content is list[ {type:video}, {type:text} ]
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if OLD_SUFFIX in text:
                        item["text"] = text.replace(OLD_SUFFIX, NEW_SUFFIX)
                        changed = True
                    elif NEW_SUFFIX not in text and "Respond ONLY with a JSON" not in text:
                        # fallback: if suffix missing but still old phrasing variant, append new
                        # Only if it looks like a MCQ (contains A) B) etc.)
                        if "A)" in text and "B)" in text:
                            # try to replace any Provide... variant
                            if "Provide your reasoning" in text:
                                import re
                                new_text = re.sub(r"Provide your reasoning.*", NEW_SUFFIX, text, flags=re.DOTALL)
                                if new_text != text:
                                    item["text"] = new_text
                                    changed = True
                    break
    except Exception as e:
        print(f"⚠️ prompt transform failed: {e}")
    return rec, changed


def build_separated(
    src_root: Path,
    dst_sft: Path,
    dst_grpo: Path,
    materialization: str = "hardlink",
    fallback_copy: bool = True,
    dry_run: bool = False,
):
    src_root = src_root.resolve()
    dst_sft = dst_sft.resolve()
    dst_grpo = dst_grpo.resolve()
    print(f"Source: {src_root}")
    print(f"Dest SFT: {dst_sft}")
    print(f"Dest GRPO: {dst_grpo}")
    print(f"Materialization: {materialization} (fallback_copy={fallback_copy})")
    if dry_run:
        print("DRY RUN — no files will be written")

    # collect all video_id folders
    sft_copied = grpo_copied = 0
    videos_mirrored = 0
    metadata_mirrored = 0
    grpo_fixed = 0
    grpo_files = 0
    sft_files = 0

    for split in SPLITS:
        split_src = src_root / split
        if not split_src.is_dir():
            print(f"⚠️ Split missing: {split_src}, skipping")
            continue
        for vid_id in sorted(os.listdir(split_src)):
            if vid_id in ARCHIVED_DIRS:
                continue
            vid_src = split_src / vid_id
            if not vid_src.is_dir():
                continue
            # prepare dst dirs
            sft_vid = dst_sft / split / vid_id
            grpo_vid = dst_grpo / split / vid_id
            if not dry_run:
                sft_vid.mkdir(parents=True, exist_ok=True)
                grpo_vid.mkdir(parents=True, exist_ok=True)

            for fname in sorted(os.listdir(vid_src)):
                fpath = vid_src / fname
                if not fpath.is_file():
                    continue

                # categorize
                is_video = fname.endswith(".mp4")
                is_metadata = fname.startswith("clip_") and fname.endswith(".jsonl") and not fname.endswith("_sft.jsonl") and not fname.endswith("_grpo.jsonl")
                is_sft = fname.endswith("_sft.jsonl")
                is_grpo = fname.endswith("_grpo.jsonl")
                is_full_video = fname == "full_video.mp4"
                is_full_sft = fname == "full_video_sft.jsonl"

                # videos → both
                if is_video or is_full_video:
                    if not dry_run:
                        _materialize(fpath, sft_vid / fname, materialization, fallback_copy)
                        _materialize(fpath, grpo_vid / fname, materialization, fallback_copy)
                    videos_mirrored += 1
                elif is_metadata:
                    if not dry_run:
                        shutil.copy2(fpath, sft_vid / fname)
                        shutil.copy2(fpath, grpo_vid / fname)
                    metadata_mirrored += 1
                elif is_sft or is_full_sft:
                    # SFT only to sft dataset
                    if not dry_run:
                        shutil.copy2(fpath, sft_vid / fname)
                    sft_files += 1
                    sft_copied += 1
                elif is_grpo:
                    # GRPO only to grpo dataset, with transform
                    grpo_files += 1
                    if dry_run:
                        continue
                    # read source, transform, write to dst
                    out_path = grpo_vid / fname
                    # ensure parent exists (already)
                    tmp_out = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
                    changed_lines = 0
                    total_lines = 0
                    try:
                        with open(fpath, "r", encoding="utf-8") as fin, open(tmp_out, "w", encoding="utf-8") as fout:
                            for line in fin:
                                if not line.strip():
                                    continue
                                total_lines += 1
                                rec, changed = _transform_grpo_line(line)
                                if changed:
                                    changed_lines += 1
                                    grpo_fixed += 1
                                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        os.replace(tmp_out, out_path)
                    finally:
                        if tmp_out.exists():
                            try:
                                tmp_out.unlink()
                            except Exception:
                                pass
                    grpo_copied += 1
                else:
                    # unknown file → copy to both for safety (e.g., README inside vid?)
                    if not dry_run:
                        shutil.copy2(fpath, sft_vid / fname)
                        shutil.copy2(fpath, grpo_vid / fname)

    # copy top-level grpo_procedure_split_map.json to both
    for top_file in ["grpo_procedure_split_map.json"]:
        src_top = src_root / top_file
        if src_top.is_file():
            for dst_root in [dst_sft, dst_grpo]:
                if not dry_run:
                    dst_root.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_top, dst_root / top_file)

    print("\n=== Summary ===")
    print(f"Videos mirrored (counted per vid, x2 destinations): {videos_mirrored} source files → {videos_mirrored*2} dest files")
    print(f"Metadata clip_*.jsonl mirrored: {metadata_mirrored}")
    print(f"SFT files copied to dataset_sft: {sft_files}")
    print(f"GRPO files copied+transformed to dataset_grpo: {grpo_files} (lines fixed: {grpo_fixed})")
    # verify no cross-contamination
    if not dry_run:
        # quick verification
        sft_has_grpo = list(dst_sft.rglob("clip_*_grpo.jsonl"))
        grpo_has_sft = list(dst_grpo.rglob("clip_*_sft.jsonl")) + list(dst_grpo.rglob("full_video_sft.jsonl"))
        sft_mp4 = sum(1 for _ in dst_sft.rglob("*.mp4"))
        grpo_mp4 = sum(1 for _ in dst_grpo.rglob("*.mp4"))
        print(f"Verification: dataset_sft has GRPO files? {len(sft_has_grpo)} (should 0)")
        print(f"Verification: dataset_grpo has SFT files? {len(grpo_has_sft)} (should 0)")
        print(f"Verification: MP4 counts — SFT: {sft_mp4}, GRPO: {grpo_mp4} (expected ~2760 each: 2624 clips +136 full)")
        # check llm_judge remaining
        import subprocess
        try:
            import glob
            remaining = 0
            for jf in dst_grpo.rglob("clip_*_grpo.jsonl"):
                for line in open(jf, encoding="utf-8"):
                    if '"llm_judge"' in line:
                        remaining += 1
            print(f"Verification: remaining 'llm_judge' lines in dataset_grpo: {remaining} (should 0)")
        except Exception as e:
            print(f"verify failed: {e}")

        # check new prompt
        try:
            import random
            samples = list(dst_grpo.rglob("clip_*_grpo.jsonl"))
            if samples:
                samp = random.choice(samples)
                rec = json.loads(open(samp, encoding="utf-8").readline())
                txt = rec["prompt"][0]["content"][1]["text"]
                has_new = "Respond ONLY with a JSON" in txt and '"explanation"' in txt and '"answer"' in txt
                print(f"Sample GRPO prompt contains new JSON instruction? {has_new} — file: {samp}")
                if not has_new:
                    print(f"  sample text preview: {txt[:400]}")
        except Exception as e:
            print(f"sample check failed: {e}")

    return {
        "videos_mirrored": videos_mirrored,
        "metadata": metadata_mirrored,
        "sft_files": sft_files,
        "grpo_files": grpo_files,
        "grpo_fixed": grpo_fixed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mirror dataset into SFT/GRPO separated datasets.")
    parser.add_argument("--src", default="dataset", help="Source dataset root (default: dataset, relative to CWD or config)")
    parser.add_argument("--dst-sft", default="dataset_sft", help="Destination for SFT dataset")
    parser.add_argument("--dst-grpo", default="dataset_grpo", help="Destination for GRPO dataset")
    parser.add_argument("--materialization", default="hardlink", choices=["hardlink", "copy", "symlink"], help="How to mirror videos")
    parser.add_argument("--fallback-copy", action="store_true", default=True, help="Fallback to copy if hardlink fails")
    parser.add_argument("--no-fallback", dest="fallback_copy", action="store_false", help="Do not fallback")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files, just summarize")
    args = parser.parse_args(argv)

    src = Path(args.src)
    # if src is relative, resolve against CWD; if not exists try config workspace root?
    if not src.exists():
        # try resolve via config workspace root
        try:
            from ..config import get_workspace_root

            ws = Path(get_workspace_root())
            alt = ws / args.src
            if alt.exists():
                src = alt
        except Exception:
            pass
    dst_sft = Path(args.dst_sft)
    dst_grpo = Path(args.dst_grpo)
    # if dsts are relative and src was absolute, make them relative to src parent?
    if not dst_sft.is_absolute():
        # place next to src if src is absolute and dst is simple name
        if src.is_absolute():
            dst_sft = src.parent / args.dst_sft
        else:
            dst_sft = Path.cwd() / args.dst_sft
    if not dst_grpo.is_absolute():
        if src.is_absolute():
            dst_grpo = src.parent / args.dst_grpo
        else:
            dst_grpo = Path.cwd() / args.dst_grpo

    build_separated(src, dst_sft, dst_grpo, args.materialization, args.fallback_copy, args.dry_run)


if __name__ == "__main__":
    main()

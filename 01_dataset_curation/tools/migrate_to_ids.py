"""One-time migration: title-based names -> YouTube IDs.

Renames videos/audio/transcripts/refined transcripts/VLM and clip artifacts
from old title-based filenames to ``{youtube_id}`` naming and rewrites all
referencing CSVs. Matching is normalization-based (handles filesystem
sanitization like ``™`` → ``__``).

Usage:
    python -m video_pipeline.tools.migrate_to_ids [--dry-run|--apply]
                                                  [--config <path>]
                                                  [--project-root DIR]
"""

import os
import re
import json
import csv
import shutil
import argparse
from pathlib import Path

from ..common.utils import extract_youtube_id, normalize_id


def list_files(directory, suffixes):
    if not os.path.isdir(directory):
        return {}
    result = {}
    for f in os.listdir(directory):
        if f.lower().endswith(suffixes) and not f.startswith('.'):
            result[normalize_id(Path(f).stem)] = f
    return result


def rename_file(directory, old_name, new_name, dry_run, moved, warnings):
    old_path = os.path.join(directory, old_name)
    new_path = os.path.join(directory, new_name)
    if not os.path.exists(old_path):
        warnings.append(f"  ⚠️  Missing source (skipped): {old_path}")
        return
    if os.path.exists(new_path) and new_path != old_path:
        warnings.append(f"  ⚠️  Target already exists (skipped): {new_path}")
        return
    print(f"  🔄 {old_name}  ->  {new_name}")
    if not dry_run:
        os.rename(old_path, new_path)
    moved.append(old_name)


def read_csv_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def build_filename_to_id_map(metadata_path, fallback_path=None):
    """{normalized_old_filename: {"youtube_id": str, "filename": old}}"""
    rows = read_csv_rows(metadata_path)
    source_label = metadata_path
    if rows is None and fallback_path:
        rows = read_csv_rows(fallback_path)
        source_label = fallback_path
    if rows is None:
        print(f"❌ Metadata not found: {metadata_path}")
        return {}
    print(f"ℹ️  Using mapping source: {source_label}")

    mapping = {}
    for row in rows:
        filename = (row.get("filename") or row.get("video_name") or "").strip()
        url = (row.get("url") or "").strip()
        if not filename:
            continue
        yt_id = (row.get("video_id") or "").strip() or extract_youtube_id(url)
        if not yt_id:
            print(f"  ⚠️  No YouTube ID derivable for '{filename}' — left as-is.")
            continue
        mapping[normalize_id(Path(filename).stem)] = {"youtube_id": yt_id, "filename": filename}
    return mapping


def migrate_media(root, mapping, dry_run, moved, warnings):
    dirs = {
        "videos": (["videos"], ('.mp4', '.mkv', '.webm', '.mov', '.avi')),
        "audio": (["audio"], ('.wav',)),
        "transcripts": (["transcripts"], ('.json',)),
        "refined_transcripts": (["refined_transcripts"], ('.txt',)),
    }
    for label, (parts, suffixes) in dirs.items():
        directory = Path(root).joinpath(*parts)
        if not directory.is_dir():
            continue
        print(f"\n--- {label}/ ---")
        disk = list_files(str(directory), suffixes)
        for norm, entry in mapping.items():
            old_name = disk.get(norm)
            if not old_name:
                continue
            rename_file(str(directory), old_name, f"{entry['youtube_id']}{Path(old_name).suffix}",
                        dry_run, moved, warnings)

    full_dir = Path(root) / "refined_transcripts" / "full_responses"
    if full_dir.is_dir():
        print("\n--- refined_transcripts/full_responses/ ---")
        disk = list_files(str(full_dir), ('.txt',))
        for norm, entry in mapping.items():
            prefix = norm + "fullresponse"
            for disk_norm, old_name in disk.items():
                if disk_norm.startswith(prefix):
                    rename_file(str(full_dir), old_name, f"{entry['youtube_id']}_full_response.txt",
                                dry_run, moved, warnings)


def migrate_csvs(root, mapping, dry_run, warnings):
    meta_path = Path(root) / "videos" / "video_metadata.csv"
    rows = read_csv_rows(meta_path)
    if rows is not None:
        print("\n--- videos/video_metadata.csv ---")
        changed = 0
        for row in rows:
            filename = (row.get("filename") or "").strip()
            entry = mapping.get(normalize_id(Path(filename).stem))
            if not entry:
                continue
            row["video_id"] = entry["youtube_id"]
            if filename:
                row["filename"] = f"{entry['youtube_id']}{Path(filename).suffix}"
            audio = (row.get("audio_filename") or "").strip()
            if audio and audio not in ("N/A", "FAILED", "SKIPPED_DURATION"):
                row["audio_filename"] = f"{entry['youtube_id']}{Path(audio).suffix}"
            changed += 1
        print(f"  ℹ️  {changed} rows updated.")
        if not dry_run:
            write_csv_rows(str(meta_path), rows, list(rows[0].keys()) if rows else [])

    log_path = Path(root) / "refined_transcripts" / "refinement_log.csv"
    rows = read_csv_rows(log_path)
    if rows is not None:
        print("\n--- refined_transcripts/refinement_log.csv ---")
        changed = 0
        for row in rows:
            vname = (row.get("video_name") or "").strip()
            entry = mapping.get(normalize_id(Path(vname).stem))
            if not entry or vname in ("NOT_FOUND", "FAILED"):
                continue
            row["video_name"] = f"{entry['youtube_id']}{Path(vname).suffix}"
            tname = (row.get("transcript_name") or "").strip()
            if tname:
                row["transcript_name"] = f"{entry['youtube_id']}{Path(tname).suffix}"
            changed += 1
        print(f"  ℹ️  {changed} rows updated.")
        if not dry_run:
            write_csv_rows(str(log_path), rows, list(rows[0].keys()) if rows else [])


def migrate_vlm_dataset(root, mapping, dry_run, moved, warnings):
    vlm_dir = Path(root) / "vlm_dataset"
    if not vlm_dir.is_dir():
        return
    print("\n--- vlm_dataset/ ---")
    for f in sorted(vlm_dir.iterdir()):
        if f.suffix != ".jsonl":
            continue
        entry = mapping.get(normalize_id(f.stem))
        if not entry:
            continue
        new_path = vlm_dir / f"{entry['youtube_id']}.jsonl"
        if not dry_run:
            with open(f, 'r', encoding='utf-8') as fh:
                lines = [json.loads(l) for l in fh if l.strip()]
            for rec in lines:
                if "video_id" in rec:
                    rec["video_id"] = entry["youtube_id"]
                if "original_filename" in rec and rec.get("original_filename"):
                    rec["original_filename"] = f"{entry['youtube_id']}.mp4"
            with open(new_path, 'w', encoding='utf-8') as fh:
                for rec in lines:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.unlink()
        print(f"  🔄 {f.name}  ->  {new_path.name}")


def migrate_clip_dataset(root, mapping, dry_run, moved, warnings):
    clip_dir = Path(root) / "clip_dataset"
    if not clip_dir.is_dir():
        return
    print("\n--- clip_dataset/ ---")
    for folder in sorted(clip_dir.iterdir()):
        if not folder.is_dir():
            continue
        entry = mapping.get(normalize_id(folder.name))
        if not entry:
            continue
        new_folder = clip_dir / entry["youtube_id"]
        if not dry_run and new_folder != folder:
            if new_folder.exists():
                warnings.append(f"  ⚠️  Target folder exists, merging may collide: {new_folder}")
                continue
            for clip_file in folder.iterdir():
                if clip_file.suffix == ".jsonl":
                    with open(clip_file, 'r', encoding='utf-8') as fh:
                        rec = json.loads(fh.readline())
                    rec["parent_video_id"] = entry["youtube_id"]
                    clip_base = Path(rec.get("clip_filename", "")).stem
                    if "clip_id" in rec:
                        rec["clip_id"] = f"{entry['youtube_id']}_{clip_base}"
                    with open(clip_file, 'w', encoding='utf-8') as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.rename(folder, new_folder)
        print(f"  🔄 {folder.name}/  ->  {new_folder.name}/")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the migration (default behavior when --apply is absent)")
    parser.add_argument("--apply", action="store_true",
                        help="Perform the migration (default: dry-run preview)")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--project-root", default=None,
                        help="Root of the OLD project being migrated "
                             "(default: the active config's workspace root)")
    args = parser.parse_args(argv)
    dry_run = not args.apply

    from ..config import build_settings, get_workspace_root
    settings = build_settings(config_path=args.config)
    root = args.project_root or get_workspace_root(args.config)

    print("=" * 60)
    print("  MIGRATION: title-based names -> YouTube IDs")
    print(f"  Mode: {'DRY RUN (no changes)' if dry_run else 'APPLY'}")
    print("=" * 60)

    meta_path = Path(root) / "videos" / "video_metadata.csv"
    fallback_path = Path(root) / "dataset_info.csv"
    if not meta_path.exists() and not fallback_path.exists():
        print(f"❌ No metadata source found: {meta_path} or {fallback_path}")
        return

    mapping = build_filename_to_id_map(str(meta_path), str(fallback_path))
    print(f"ℹ️  {len(mapping)} entries with a resolvable YouTube ID.")

    moved, warnings = [], []

    migrate_media(root, mapping, dry_run, moved, warnings)
    migrate_csvs(root, mapping, dry_run, warnings)
    migrate_vlm_dataset(root, mapping, dry_run, moved, warnings)
    migrate_clip_dataset(root, mapping, dry_run, moved, warnings)

    print("\n" + "=" * 60)
    print(f"  Files/dirs renamed : {len(moved)}")
    print(f"  Warnings           : {len(warnings)}")
    for w in warnings:
        print(w)
    print("=" * 60)
    print("  Re-run with --apply to perform the migration." if dry_run else "  Done.")
    print(f"  dataset/ already uses YouTube IDs and was not touched.")


if __name__ == "__main__":
    main()

"""Rename dataset folders to YouTube IDs and verify sync.

Renames ``dataset/{Train,Validation,Test}/{old_name}/`` folders to
``{youtube_id}/`` and rewrites all internal references (parent_video_id,
clip_id, SFT/GRPO video paths), then deeply verifies the output.

Usage:
    python -m video_pipeline.tools.rename_to_youtubeid [--in DIR] [--out DIR]
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm

from ..config import build_settings

SPLITS = ["Train", "Validation", "Test"]


def extract_youtube_id(url):
    if not url:
        return ""
    if "youtu.be" in url:
        return url.rstrip('/').split('/')[-1].split('?')[0]
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    return ""


def _is_meta_file(name):
    return not name.endswith("_sft.jsonl") and not name.endswith("_grpo.jsonl")


def _rewrite_video_refs(items, yt_id):
    """Rewrite every {'type': 'video', 'video': 'old/name.mp4'} ref to 'id/name.mp4'."""
    for item in items:
        content_data = item.get("content", [])
        if isinstance(content_data, list):
            for entry in content_data:
                if isinstance(entry, dict) and entry.get("type") == "video" and "video" in entry:
                    filename = entry["video"].split("/")[-1]
                    entry["video"] = f"{yt_id}/{filename}"


def process_and_rename_dataset(input_dir, output_dir):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"❌ Input directory not found: {in_dir}")

    print("=== Phase 1: Processing and Renaming ===")

    folder_mapping = {}

    for split in SPLITS:
        split_path = in_dir / split
        if not split_path.exists():
            continue

        parent_folders = [f for f in split_path.iterdir() if f.is_dir()]

        for parent_folder in tqdm(parent_folders, desc=f"Processing {split}"):
            old_parent_name = parent_folder.name

            meta_files = list(parent_folder.glob("clip_*.jsonl"))
            core_meta_files = [f for f in meta_files if _is_meta_file(f.name)]

            if not core_meta_files:
                print(f"\n⚠️ Skipping '{old_parent_name}': No core clip_xx.jsonl found to extract YouTube ID.")
                continue

            yt_id = None
            with open(core_meta_files[0], 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line.strip():
                    data = json.loads(first_line)
                    yt_id = extract_youtube_id(data.get("parent_video_url", ""))

            if not yt_id:
                print(f"\n⚠️ Skipping '{old_parent_name}': Could not extract YouTube ID from URL.")
                continue

            folder_mapping[old_parent_name] = yt_id

            target_dir = out_dir / split / yt_id
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in parent_folder.iterdir():
                if not file_path.is_file():
                    continue

                target_path = target_dir / file_path.name

                if file_path.suffix == ".mp4":
                    shutil.copy2(file_path, target_path)

                elif file_path.suffix == ".jsonl":
                    with open(file_path, 'r', encoding='utf-8') as src_f, \
                         open(target_path, 'w', encoding='utf-8') as dst_f:
                        for line in src_f:
                            if not line.strip():
                                continue

                            data = json.loads(line)

                            if _is_meta_file(file_path.name):
                                clip_filename = data.get("clip_filename", "")
                                clip_base = Path(clip_filename).stem if clip_filename else file_path.stem
                                data["parent_video_id"] = yt_id
                                data["clip_id"] = f"{yt_id}_{clip_base}"

                            elif file_path.name.endswith("_sft.jsonl"):
                                _rewrite_video_refs(data.get("messages", []), yt_id)

                            elif file_path.name.endswith("_grpo.jsonl"):
                                _rewrite_video_refs(data.get("prompt", []), yt_id)

                            dst_f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ Phase 1 Complete. Data written to {output_dir}\n")
    return SPLITS


def verify_dataset_sync(output_dir, splits):
    out_dir = Path(output_dir)
    print("=== Phase 2: Solid Two-Way Sync Verification ===")

    global_errors = 0
    verified_files = 0
    verified_folders = 0

    for split in splits:
        split_path = out_dir / split
        if not split_path.exists():
            continue

        parent_folders = [f for f in split_path.iterdir() if f.is_dir()]

        for parent_folder in parent_folders:
            yt_id = parent_folder.name
            verified_folders += 1

            for file_path in parent_folder.iterdir():
                if file_path.suffix != ".jsonl":
                    continue

                verified_files += 1

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue

                        try:
                            data = json.loads(line)

                            if _is_meta_file(file_path.name):
                                if data.get("parent_video_id") != yt_id:
                                    print(f"❌ Sync Error in {file_path.name}: parent_video_id mismatch")
                                    global_errors += 1
                                clip_base = Path(data.get("clip_filename", "")).stem
                                if data.get("clip_id") != f"{yt_id}_{clip_base}":
                                    print(f"❌ Sync Error in {file_path.name}: clip_id mismatch")
                                    global_errors += 1

                            elif file_path.name.endswith("_sft.jsonl"):
                                for message in data.get("messages", []):
                                    content_data = message.get("content", [])
                                    if isinstance(content_data, list):
                                        for item in content_data:
                                            if isinstance(item, dict) and item.get("type") == "video":
                                                if not item.get("video", "").startswith(f"{yt_id}/"):
                                                    print(f"❌ Sync Error in {file_path.name}: video path does not start with '{yt_id}/'")
                                                    global_errors += 1

                            elif file_path.name.endswith("_grpo.jsonl"):
                                for prompt_item in data.get("prompt", []):
                                    content_data = prompt_item.get("content", [])
                                    if isinstance(content_data, list):
                                        for item in content_data:
                                            if isinstance(item, dict) and item.get("type") == "video":
                                                if not item.get("video", "").startswith(f"{yt_id}/"):
                                                    print(f"❌ Sync Error in {file_path.name}: video path does not start with '{yt_id}/'")
                                                    global_errors += 1

                        except json.JSONDecodeError:
                            print(f"❌ JSON Decode Error in {file_path.name} at line {line_num}")
                            global_errors += 1

    print("\n=== Final Sync Report ===")
    print(f"Folders verified: {verified_folders}")
    print(f"JSONL Files deeply checked: {verified_files}")

    if global_errors == 0:
        print("🎉 SUCCESS: Solid Two-Way Verification Passed!")
    else:
        print(f"⚠️ WARNING: Verification failed with {global_errors} errors. Check the logs above.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rename dataset folders to YouTube IDs.")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--in", dest="input_dir", default=None, help="Input dataset directory")
    parser.add_argument("--out", dest="output_dir", default=None, help="Output directory")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    input_dir = args.input_dir or os.path.join(os.path.dirname(settings.clips_dir), "dataset")
    output_dir = args.output_dir or os.path.join(os.path.dirname(settings.clips_dir), "dataset_new")

    active_splits = process_and_rename_dataset(input_dir, output_dir)
    verify_dataset_sync(output_dir, active_splits)


if __name__ == "__main__":
    main()

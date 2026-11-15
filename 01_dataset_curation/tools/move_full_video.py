"""Copy full videos into the final dataset folders.

Adds ``full_video.mp4`` to every parent folder of the dataset (used by the
video-level dataset tooling) and verifies the copies.

Usage:
    python -m video_pipeline.tools.move_full_video [--videos-dir DIR] [--dataset-dir DIR]
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

from ..common.utils import normalize_id
from ..config import build_settings

SPLITS = ["Train", "Validation", "Test"]


def get_available_videos(videos_dir):
    catalog = []
    if not videos_dir.exists():
        print(f"❌ Videos directory not found at: {videos_dir}")
        return catalog

    for f in videos_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov']:
            catalog.append({
                'path': f,
                'name': f.stem,
                'norm_name': normalize_id(f.stem),
            })
    return catalog


def find_matching_video(yt_id, title, video_catalog):
    # Strategy 1: YouTube ID contained in the filename
    if yt_id:
        for v in video_catalog:
            if yt_id in v['name']:
                return v['path']

    # Strategy 2: normalized title containment
    norm_title = normalize_id(title)
    if norm_title:
        for v in video_catalog:
            if norm_title in v['norm_name'] or v['norm_name'] in norm_title:
                return v['path']

    return None


def copy_main_videos_to_dataset(videos_dir_path, dataset_dir_path):
    videos_dir = Path(videos_dir_path)
    dataset_dir = Path(dataset_dir_path)

    print("=== Phase 1: Robust Video Matching & Copying ===")

    video_catalog = get_available_videos(videos_dir)
    print(f"🔍 Found {len(video_catalog)} videos in source directory.\n")

    if not video_catalog:
        return

    not_found_list = []
    success_count = 0

    for split in SPLITS:
        split_path = dataset_dir / split
        if not split_path.exists():
            continue

        parent_folders = [f for f in split_path.iterdir() if f.is_dir()]

        for parent_folder in tqdm(parent_folders, desc=f"Processing {split} split"):
            yt_id = parent_folder.name

            meta_files = list(parent_folder.glob("clip_*.jsonl"))
            core_meta_files = [f for f in meta_files
                               if not f.name.endswith("_sft.jsonl") and not f.name.endswith("_grpo.jsonl")]

            if not core_meta_files:
                continue

            parent_video_title = ""
            with open(core_meta_files[0], 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line.strip():
                    try:
                        data = json.loads(first_line)
                        parent_video_title = data.get("parent_video_title", "")
                    except json.JSONDecodeError:
                        pass

            matched_video_path = find_matching_video(yt_id, parent_video_title, video_catalog)

            target_path = parent_folder / "full_video.mp4"

            if matched_video_path:
                if not target_path.exists() or target_path.stat().st_size == 0:
                    shutil.copy2(matched_video_path, target_path)
                success_count += 1
            else:
                not_found_list.append({'yt_id': yt_id, 'title': parent_video_title})

    print(f"\n✅ Phase 1 Complete. Successfully matched/copied: {success_count}")
    if not_found_list:
        print(f"⚠️ Could not find source videos for {len(not_found_list)} folders:")
        for item in not_found_list[:5]:
            print(f"   - ID: {item['yt_id']} | Title: {item['title']}")
        if len(not_found_list) > 5:
            print(f"   ... and {len(not_found_list) - 5} more.")


def verify_dataset_videos(dataset_dir_path):
    print("\n=== Phase 2: Robust Verification ===")
    dataset_dir = Path(dataset_dir_path)

    global_errors = 0
    verified_folders = 0
    missing_videos = []
    empty_videos = []

    for split in SPLITS:
        split_path = dataset_dir / split
        if not split_path.exists():
            continue

        parent_folders = [f for f in split_path.iterdir() if f.is_dir()]

        for parent_folder in parent_folders:
            yt_id = parent_folder.name
            verified_folders += 1

            if not list(parent_folder.glob("*.jsonl")):
                continue

            full_video_path = parent_folder / "full_video.mp4"

            if not full_video_path.exists():
                missing_videos.append(yt_id)
                global_errors += 1
                continue

            file_size_mb = full_video_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 1.0:
                empty_videos.append(yt_id)
                global_errors += 1

    print(f"Total Folders Verified: {verified_folders}")

    if global_errors == 0:
        print("🎉 SUCCESS: 100% Verification Passed! All folders have a valid 'full_video.mp4'.")
    else:
        print(f"❌ Verification Failed with {global_errors} total issues.")
        if missing_videos:
            print(f"   -> Missing 'full_video.mp4' in {len(missing_videos)} folders: {', '.join(missing_videos[:10])}...")
        if empty_videos:
            print(f"   -> Corrupted/Empty (<1MB) 'full_video.mp4' in {len(empty_videos)} folders: {', '.join(empty_videos[:10])}...")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Copy full videos into the dataset(s) and verify (mirrored to SFT+GRPO).")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--videos-dir", default=None, help="Source videos directory")
    parser.add_argument("--dataset-dir", default=None, help="Final dataset directory (legacy single)")
    parser.add_argument("--dataset-sft-dir", default=None, help="SFT dataset directory (separated mode)")
    parser.add_argument("--dataset-grpo-dir", default=None, help="GRPO dataset directory (separated mode)")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    videos_dir = args.videos_dir or settings.splitter_videos_dir
    # Separated mode: if either sft/grpo explicitly passed or config has them, mirror to both
    if args.dataset_sft_dir or args.dataset_grpo_dir or (hasattr(settings, 'dataset_sft_dir') and hasattr(settings, 'dataset_grpo_dir')):
        sft_dir = args.dataset_sft_dir or getattr(settings, 'dataset_sft_dir', None) or args.dataset_dir or settings.dataset_dir
        grpo_dir = args.dataset_grpo_dir or getattr(settings, 'dataset_grpo_dir', None) or args.dataset_dir or settings.dataset_dir
        # Avoid duplicate if same path
        targets = []
        for d in [sft_dir, grpo_dir]:
            if d and d not in targets:
                targets.append(d)
        for t in targets:
            copy_main_videos_to_dataset(videos_dir, t)
            verify_dataset_videos(t)
    else:
        dataset_dir = args.dataset_dir or settings.dataset_dir
        copy_main_videos_to_dataset(videos_dir, dataset_dir)
        verify_dataset_videos(dataset_dir)


if __name__ == "__main__":
    main()

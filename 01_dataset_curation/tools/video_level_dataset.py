"""Generate the full-video narration dataset (video-level track).

Complementary to the clip-level dataset: for each whole surgery video it
creates a timestamped narration SFT record built deterministically from the
core clip metadata (no LLM required).

Usage:
    python -m video_pipeline.tools.video_level_dataset [--dataset-dir DIR]
"""

import os
import json
import argparse
from pathlib import Path

from ..common.utils import time_str_to_seconds
from ..config import build_settings

SPLITS = ["Train", "Validation", "Test"]


def get_ordered_clips(parent_folder):
    """Read all core clip JSONLs and return a chronologically sorted list."""
    clips = []
    for file_path in parent_folder.glob("clip_*.jsonl"):
        if file_path.name.endswith("_sft.jsonl") or file_path.name.endswith("_grpo.jsonl"):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.loads(f.readline())
                clips.append(data)
            except json.JSONDecodeError:
                continue
    clips.sort(key=lambda x: time_str_to_seconds(x.get("timestamp_start_in_parent", "00:00")) or 0)
    return clips


def process_dataset(dataset_dir):
    print("=== Generating Full-Video Narration Dataset ===")

    processed_count = skipped_count = 0

    for split in SPLITS:
        split_path = Path(dataset_dir) / split
        if not split_path.exists():
            continue

        parent_folders = [f for f in split_path.iterdir() if f.is_dir()]

        for parent_folder in parent_folders:
            yt_id = parent_folder.name

            sft_filepath = parent_folder / "full_video_sft.jsonl"

            if sft_filepath.exists() and sft_filepath.stat().st_size > 0:
                print(f"⏩ Skipped {yt_id}: Already processed.")
                skipped_count += 1
                continue

            video_ref_path = f"{yt_id}/full_video.mp4"
            clips = get_ordered_clips(parent_folder)

            if not clips:
                continue

            print(f"⏳ Processing {yt_id} ({len(clips)} clips)...", end="", flush=True)

            narration_lines = []

            for clip in clips:
                start = clip.get("timestamp_start_in_parent", "00:00")
                end = clip.get("timestamp_end_in_parent", "00:00")
                step = clip.get("step_title", "Unknown Step")
                desc = clip.get("visual_description", "")

                narration_lines.append(f"[{start} - {end}] {step}: {desc}")

            full_narration_text = "\n".join(narration_lines)

            sft_narration = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "video", "video": video_ref_path},
                        {"type": "text", "text": "Provide a detailed timestamped narration of this surgical procedure."},
                    ]},
                    {"role": "assistant", "content": full_narration_text},
                ]
            }

            with open(sft_filepath, 'w', encoding='utf-8') as f:
                f.write(json.dumps(sft_narration, ensure_ascii=False) + "\n")

            print(" ✅ Success.")
            processed_count += 1

    print("\n=== Generation Summary ===")
    print(f"✅ Newly Processed: {processed_count}")
    print(f"⏩ Skipped (Already Done): {skipped_count}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate full-video narration SFT dataset (SFT dataset only).")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--dataset-dir", default=None, help="Final dataset directory (legacy) or SFT dataset dir")
    parser.add_argument("--dataset-sft-dir", default=None, help="SFT dataset directory (separated mode)")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    # full_video_sft is SFT-only, so prefer SFT dataset in separated mode
    if args.dataset_sft_dir:
        dataset_dir = args.dataset_sft_dir
    elif args.dataset_dir:
        dataset_dir = args.dataset_dir
    elif hasattr(settings, 'dataset_sft_dir'):
        dataset_dir = settings.dataset_sft_dir
    else:
        dataset_dir = settings.dataset_dir

    process_dataset(dataset_dir)


if __name__ == "__main__":
    main()

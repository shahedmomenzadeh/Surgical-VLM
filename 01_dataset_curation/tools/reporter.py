"""Console statistics report for the VLM dataset.

Prints per-video clip counts and clip-length stats (min/max/average) to the
console.

Usage:
    python -m video_pipeline.tools.reporter [--vlm-dir DIR]
"""

import os
import json
import argparse

from ..common.utils import time_str_to_seconds
from ..config import build_settings


def generate_expanded_vlm_report(vlm_dir):
    total_clips = 0
    total_duration = 0
    video_files_count = 0

    max_clips_per_video = 0
    max_clips_video_name = ""
    min_clips_per_video = float('inf')
    min_clips_video_name = ""

    max_clip_duration = 0
    max_clip_duration_video_name = ""
    min_clip_duration = float('inf')
    min_clip_duration_video_name = ""

    if not os.path.exists(vlm_dir):
        print(f"Directory {vlm_dir} not found.")
        return

    for filename in os.listdir(vlm_dir):
        if filename.endswith(".jsonl") and "all.jsonl" not in filename:
            video_files_count += 1
            file_path = os.path.join(vlm_dir, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        video_name = data.get("video_title") or data.get("video_id") or "Unknown"
                        annotations = data.get("vlm_annotations", [])
                        num_clips = len(annotations)
                        total_clips += num_clips

                        if num_clips > max_clips_per_video:
                            max_clips_per_video = num_clips
                            max_clips_video_name = video_name
                        if num_clips < min_clips_per_video:
                            min_clips_per_video = num_clips
                            min_clips_video_name = video_name

                        for clip in annotations:
                            start = time_str_to_seconds(clip.get("timestamp_start", "00:00")) or 0
                            end = time_str_to_seconds(clip.get("timestamp_end", "00:00")) or 0
                            duration = end - start
                            total_duration += duration

                            if duration > max_clip_duration:
                                max_clip_duration = duration
                                max_clip_duration_video_name = video_name
                            if duration < min_clip_duration:
                                min_clip_duration = duration
                                min_clip_duration_video_name = video_name

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

    avg_length = total_duration / total_clips if total_clips > 0 else 0
    avg_clips_per_video = total_clips / video_files_count if video_files_count > 0 else 0

    if total_clips == 0:
        min_clips_per_video = 0
        min_clip_duration = 0

    print("=== Expanded VLM Dataset Report ===")
    print(f"Total video files:             {video_files_count}")
    print(f"Total sub-video clips:         {total_clips}")
    print("-" * 50)
    print(f"Average clips per video:       {avg_clips_per_video:.2f}")
    print(f"Max clips in a video:          {max_clips_per_video} ({max_clips_video_name})")
    print(f"Min clips in a video:          {min_clips_per_video} ({min_clips_video_name})")
    print("-" * 50)
    print(f"Average clip length:           {avg_length:.2f} seconds")
    print(f"Max clip length:               {max_clip_duration}s ({max_clip_duration_video_name})")
    print(f"Min clip length:               {min_clip_duration}s ({min_clip_duration_video_name})")


def main(argv=None):
    parser = argparse.ArgumentParser(description="VLM dataset console report.")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--vlm-dir", default=None, help="VLM dataset directory")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    generate_expanded_vlm_report(args.vlm_dir or settings.vlm_dir)


if __name__ == "__main__":
    main()

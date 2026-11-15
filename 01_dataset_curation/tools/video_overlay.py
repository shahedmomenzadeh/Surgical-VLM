"""Burn VLM visual descriptions as overlays on full videos (moviepy).

Unlike ``video_annotation`` (which cuts clips), this overlays the captions
onto the whole video and writes ``labeled_{filename}`` files.

Usage:
    python -m video_pipeline.tools.video_overlay --jsonl <file> [--out DIR]
"""

import json
import os
import argparse

from ..common.utils import time_str_to_seconds
from ..config import build_settings


def _moviepy():
    """Lazy moviepy import (optional dependency)."""
    try:
        from moviepy import VideoFileClip, TextClip, CompositeVideoClip
        return VideoFileClip, TextClip, CompositeVideoClip
    except ImportError as e:
        print("❌ Error: 'moviepy' package not installed. Run: pip install moviepy")
        raise SystemExit(1) from e


def process_videos(jsonl_path, videos_dir, output_dir="output_videos"):
    VideoFileClip, TextClip, CompositeVideoClip = _moviepy()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            filename = entry.get('original_filename')
            video_path = os.path.join(videos_dir, filename)

            if not os.path.exists(video_path):
                print(f"Warning: Video file not found at {video_path}. Skipping.")
                continue

            print(f"Processing: {filename}...")

            try:
                video = VideoFileClip(video_path)
                clips = [video]

                annotations = entry.get('vlm_annotations', [])
                for ann in annotations:
                    text_content = ann.get('visual_description', '')
                    start_time = time_str_to_seconds(ann.get('timestamp_start', '00:00')) or 0
                    end_time = time_str_to_seconds(ann.get('timestamp_end', '00:00')) or 0
                    duration = end_time - start_time

                    if duration <= 0:
                        continue

                    txt_clip = (TextClip(
                        text=text_content,
                        font_size=40,
                        color='white',
                        stroke_color='black',
                        stroke_width=2,
                        size=(int(video.w * 0.9), None),
                        method='caption',
                        text_align='center',
                    )
                        .with_position(('center', 70))
                        .with_start(start_time)
                        .with_duration(duration))

                    clips.append(txt_clip)

                final_video = CompositeVideoClip(clips)

                output_path = os.path.join(output_dir, f"labeled_{filename}")
                final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

                print(f"Saved to: {output_path}")

                video.close()
                final_video.close()

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Overlay descriptions on full videos.")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--jsonl", required=True, help="Per-video VLM JSONL file")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    if not os.path.exists(args.jsonl):
        print(f"Error: JSONL file not found at {args.jsonl}")
        return
    process_videos(args.jsonl, settings.videos_dir, args.out or "output_videos")


if __name__ == "__main__":
    main()

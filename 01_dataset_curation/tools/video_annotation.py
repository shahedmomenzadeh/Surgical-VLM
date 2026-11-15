"""Burn VLM visual descriptions into step clips (moviepy).

Cuts each annotated step from the source video and overlays the
``visual_description`` text, writing annotated clips to
``<workspace>/annotated_videos/{video_id}/``.

Usage:
    python -m video_pipeline.tools.video_annotation [--video-ids id1 id2 ... | --all]
"""

import json
import os
import argparse

from ..common.utils import time_str_to_seconds
from ..config import build_settings

OUTPUT_BASE_DIR = "annotated_videos"


def _moviepy():
    """Lazy moviepy import (optional dependency)."""
    try:
        from moviepy import VideoFileClip, TextClip, CompositeVideoClip
        return VideoFileClip, TextClip, CompositeVideoClip
    except ImportError as e:
        print("❌ Error: 'moviepy' package not installed. Run: pip install moviepy")
        raise SystemExit(1) from e


def parse_timestamp(time_str):
    """MM:SS / HH:MM:SS → seconds (0 on failure)."""
    return time_str_to_seconds(time_str) or 0


def annotate_clips(vlm_dir, videos_source_dir, video_ids=None, output_base=None):
    VideoFileClip, TextClip, CompositeVideoClip = _moviepy()
    output_base = output_base or os.path.join(os.path.dirname(vlm_dir), OUTPUT_BASE_DIR)
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    for filename in os.listdir(vlm_dir):
        if not filename.endswith(".jsonl") or "all.jsonl" in filename:
            continue

        video_id = os.path.splitext(filename)[0]
        if video_ids and video_id not in video_ids:
            continue

        jsonl_path = os.path.join(vlm_dir, filename)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            entry = json.loads(f.readline())

        orig_filename = entry.get('original_filename')
        video_path = os.path.join(videos_source_dir, orig_filename)

        if not os.path.exists(video_path):
            print(f"⚠️ Video source not found: {video_path}")
            continue

        video_output_folder = os.path.join(output_base, video_id)
        os.makedirs(video_output_folder, exist_ok=True)

        print(f"🎬 Processing {len(entry.get('vlm_annotations', []))} clips for: {video_id}")

        try:
            full_video = VideoFileClip(video_path)
            annotations = entry.get('vlm_annotations', [])

            for ann in annotations:
                step_num = ann.get('step_number', 0)
                text_content = ann.get('visual_description', '')
                start_t = parse_timestamp(ann.get('timestamp_start', '00:00'))
                end_t = parse_timestamp(ann.get('timestamp_end', '00:00'))

                duration = end_t - start_t
                if duration <= 0:
                    continue

                sub_clip = full_video.subclipped(start_t, end_t)

                txt_clip = (TextClip(
                    text=text_content,
                    font_size=40,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    size=(int(sub_clip.w * 0.9), None),
                    method='caption',
                    text_align='center',
                )
                    .with_position(('center', 70))
                    .with_duration(duration))

                final_clip = CompositeVideoClip([sub_clip, txt_clip])
                clip_filename = f"clip_{step_num}.mp4"
                output_path = os.path.join(video_output_folder, clip_filename)

                final_clip.write_videofile(output_path, codec='libx264',
                                           audio_codec='aac', logger=None)
                print(f"   ✅ Saved: {clip_filename}")

                final_clip.close()

            full_video.close()

        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Annotate step clips with visual descriptions.")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--video-ids", nargs="*", default=None,
                        help="Video IDs (filename stems) to annotate; default: all")
    parser.add_argument("--all", action="store_true",
                        help="Annotate every video in the VLM dataset")
    parser.add_argument("--out", dest="output_base", default=None,
                        help="Output base directory")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    video_ids = None if args.all else args.video_ids
    annotate_clips(settings.vlm_dir, settings.videos_dir, video_ids, args.output_base)


if __name__ == "__main__":
    main()

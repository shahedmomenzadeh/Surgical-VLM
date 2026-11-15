"""Downsample videos to a target frame rate.

Re-samples every video in the videos directory to a target FPS, writing to
a separate output directory (default: ``<workspace>/videos_down``).

Usage:
    python -m video_pipeline.tools.downsample_videos [--fps 10] [--out DIR]
"""

import os
import shutil
import subprocess
import argparse

from ..common.utils import is_ffmpeg_installed
from ..config import build_settings

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'}


def downsample_video(input_path, output_path, fps):
    """Downsample a video with live FFmpeg console progress."""
    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'fps={fps}',
        '-c:a', 'copy',
        '-stats', '-loglevel', 'info',
        output_path,
    ]
    print(f"\n[CMD] Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed for {os.path.basename(input_path)} with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def downsample_directory(input_dir, output_dir, fps):
    """Downsample every video in ``input_dir`` to ``fps``, writing to ``output_dir``."""
    if not is_ffmpeg_installed():
        print("❌ Error: FFmpeg is not installed or not in the system PATH.")
        return 0, 0

    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Input:  {os.path.abspath(input_dir)}")
    print(f"📂 Output: {os.path.abspath(output_dir)}\n")

    video_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]

    if not video_files:
        print(f"⚠️ No video files found in {input_dir}.")
        return 0, 0

    print(f"✅ Found {len(video_files)} videos. Starting sequential processing...\n")

    success_count = error_count = 0
    for i, filename in enumerate(video_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print("=" * 50)
        print(f"Processing Video {i}/{len(video_files)}: {filename}")
        print("=" * 50)
        if downsample_video(input_path, output_path, fps):
            print(f"\n✅ Successfully processed: {filename}")
            success_count += 1
        else:
            print(f"\n❌ Failed to process: {filename}")
            error_count += 1

    print("\n" + "=" * 30)
    print("   PROCESSING COMPLETED")
    print("=" * 30)
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors:  {error_count}")
    print(f"📂 Output Location: {os.path.abspath(output_dir)}")
    return success_count, error_count


def main(argv=None):
    parser = argparse.ArgumentParser(description="Video downsampling utility (FPS).")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--in", dest="input_dir", default=None, help="Input directory")
    parser.add_argument("--out", dest="output_dir", default=None, help="Output directory")
    parser.add_argument("--fps", type=int, default=None, help="Target frame rate (default: config)")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    input_dir = args.input_dir or settings.downsample.input_dir
    output_dir = args.output_dir or settings.downsample.output_dir
    fps = args.fps or settings.downsample.target_fps

    print(f"--- VIDEO DOWNSAMPLING UTILITY (VERBOSE MODE) ---")
    downsample_directory(input_dir, output_dir, fps)


if __name__ == "__main__":
    main()

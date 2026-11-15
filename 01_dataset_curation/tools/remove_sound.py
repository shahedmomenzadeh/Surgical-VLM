"""Remove the audio stream from a directory tree of videos.

Strips audio with ``ffmpeg -c:v copy -an`` (video copied, no re-encode),
preserving the folder structure. Default: ``clip_dataset`` →
``<workspace>/clip_dataset_muted``.

Usage:
    python -m video_pipeline.tools.remove_sound [--in DIR] [--out DIR]
"""

import subprocess
from pathlib import Path
import argparse

from ..config import build_settings

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}


def remove_audio_from_dataset(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    for video_file in input_path.rglob('*'):
        if video_file.suffix.lower() in VIDEO_EXTENSIONS:
            relative_path = video_file.relative_to(input_path)
            out_file = output_path / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

            command = [
                'ffmpeg',
                '-i', str(video_file),
                '-c:v', 'copy',
                '-an',
                '-y',
                str(out_file),
            ]
            print(f"Muting: {relative_path}")
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {video_file.name}. It might be corrupted "
                      f"or in an unsupported format: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Strip audio from videos (recursive).")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--in", dest="input_directory", default=None, help="Input folder")
    parser.add_argument("--out", dest="output_directory", default=None,
                        help="Output folder (default: <workspace>/clip_dataset_muted)")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    input_directory = args.input_directory or settings.clips_dir
    output_directory = args.output_directory or os.path.join(
        os.path.dirname(settings.clips_dir), "clip_dataset_muted")

    print(f"Starting to process videos from '{input_directory}'...")
    remove_audio_from_dataset(input_directory, output_directory)
    print("Finished processing all videos!")


if __name__ == "__main__":
    import os
    main()

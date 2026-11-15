"""Splitter stage: cuts full-length videos into step-level clips with FFmpeg.

Input: per-video VLM JSONL files (`vlm_dataset/{video_id}.jsonl`).
Output: `clip_dataset/{video_id}/clip_XX.mp4` + `clip_XX.jsonl` metadata.
"""

import os
import json
import subprocess
import time
from tqdm import tqdm

from ..common.io import ProgressLog
from ..common.utils import is_ffmpeg_installed, time_str_to_seconds

SPLIT_LOG_HEADERS = ["video_id", "clips_generated", "errors", "timestamp"]


def find_video(videos_dir, original_filename, video_id):
    """Locate the raw video: exact original filename, then `{video_id}.ext`."""
    if original_filename:
        exact_path = os.path.join(videos_dir, original_filename)
        if os.path.exists(exact_path):
            return exact_path

    for ext in ['.mp4', '.mkv', '.webm', '.mov']:
        path = os.path.join(videos_dir, f"{video_id}{ext}")
        if os.path.exists(path):
            return path

    return None


def extract_clip(input_video_path, output_video_path, start_sec, duration_sec):
    """FFmpeg: extract a clip with exact frame cuts (re-encodes video)."""
    command = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-i', input_video_path,
        '-t', str(duration_sec),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
        '-c:a', 'copy',
        output_video_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"❌ Unexpected error during FFmpeg execution: {e}")
        return False


def run_splitter_pipeline(vlm_input_dir, videos_dir, output_dir, log_filename):
    """Main entry point for the splitter stage."""
    print("\n=== VIDEO SPLITTER PIPELINE ===\n")

    if not is_ffmpeg_installed():
        print("❌ Error: FFmpeg is not installed or not in your system PATH. Required for splitting videos.")
        return

    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_filename)
    progress_log = ProgressLog(log_file_path, SPLIT_LOG_HEADERS)

    if not os.path.exists(vlm_input_dir):
        print(f"❌ Input directory not found: {vlm_input_dir}")
        return

    input_files = [f for f in os.listdir(vlm_input_dir)
                   if f.endswith('.jsonl') and "all.jsonl" not in f]

    if not input_files:
        print("⚠️ No input VLM JSONL files found.")
        return

    print(f"Found {len(input_files)} VLM files to process for clipping.\n")

    total_clips_generated = 0
    total_videos_processed = 0

    for filename in tqdm(input_files, desc="Splitting Videos"):
        video_id = os.path.splitext(filename)[0]
        vlm_file_path = os.path.join(vlm_input_dir, filename)

        try:
            with open(vlm_file_path, 'r', encoding='utf-8') as f:
                line = f.readline()
                if not line:
                    continue
                vlm_data = json.loads(line)
        except Exception as e:
            tqdm.write(f"❌ Error reading {filename}: {e}")
            continue

        annotations = vlm_data.get("vlm_annotations", [])
        if not annotations:
            continue

        original_filename = vlm_data.get("original_filename")
        raw_video_path = find_video(videos_dir, original_filename, video_id)

        if not raw_video_path:
            tqdm.write(f"⚠️ Raw video not found for {video_id}. Searched in {videos_dir}.")
            progress_log.append(video_id=video_id, clips_generated=0, errors="Video missing",
                                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            continue

        video_clip_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_clip_dir, exist_ok=True)

        clips_created = 0
        clip_errors = 0

        for step in annotations:
            step_number = step.get("step_number")
            start_str = step.get("timestamp_start")
            end_str = step.get("timestamp_end")

            start_sec = time_str_to_seconds(start_str)
            end_sec = time_str_to_seconds(end_str)

            if start_sec is None or end_sec is None or start_sec >= end_sec:
                clip_errors += 1
                continue

            duration_sec = end_sec - start_sec
            clip_id = f"{video_id}_clip_{step_number:02d}"

            clip_video_filename = f"clip_{step_number:02d}.mp4"
            clip_jsonl_filename = f"clip_{step_number:02d}.jsonl"

            clip_video_path = os.path.join(video_clip_dir, clip_video_filename)
            clip_jsonl_path = os.path.join(video_clip_dir, clip_jsonl_filename)

            # Skip if already processed
            if os.path.exists(clip_video_path) and os.path.exists(clip_jsonl_path):
                clips_created += 1
                continue

            success = extract_clip(raw_video_path, clip_video_path, start_sec, duration_sec)

            if not success:
                clip_errors += 1
                continue

            clip_data = {
                "clip_id": clip_id,
                "parent_video_id": video_id,
                "clip_filename": clip_video_filename,
                "parent_video_title": vlm_data.get("video_title"),
                "parent_video_url": vlm_data.get("video_url"),
                "step_number": step_number,
                "clip_duration_seconds": duration_sec,
                "timestamp_start_in_parent": start_str,
                "timestamp_end_in_parent": end_str,
                "step_title": step.get("step_title"),
                "visual_description": step.get("visual_description"),
                "transcript_context": step.get("transcript_context"),
                "instruments": step.get("instruments", []),
                "anatomy": step.get("anatomy", []),
            }

            with open(clip_jsonl_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(clip_data) + "\n")

            clips_created += 1

        total_clips_generated += clips_created
        total_videos_processed += 1

        error_msg = f"{clip_errors} step(s) failed" if clip_errors > 0 else "None"
        progress_log.append(video_id=video_id, clips_generated=clips_created,
                            errors=error_msg,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

    print("\n" + "=" * 30)
    print("   VIDEO SPLITTING COMPLETED")
    print("=" * 30)
    print(f"🎬 Videos Processed: {total_videos_processed}")
    print(f"✂️ Total Clips Made: {total_clips_generated}")
    print(f"📂 Output Folder:    {os.path.abspath(output_dir)}")

"""Adverse event detector stage: LLM-based safety analysis of VLM annotations.

Uses an OpenAI-compatible endpoint (text-only analysis of the step timeline).
"""

import os
import json
import time
from tqdm import tqdm

from ..common.io import ProgressLog
from ..common.llm import resolve_and_get_client, chat_json_with_retry
from ..prompts import SAFETY_ANALYST_PROMPT

ADVERSE_EVENT_LOG_HEADERS = ["video_id", "status", "event_count", "timestamp"]
RATE_LIMIT_SLEEP_SECONDS = 30


def extract_visual_steps(vlm_data):
    """Format the VLM annotations into a chronological text timeline."""
    annotations = vlm_data.get("vlm_annotations", [])
    if not annotations:
        return ""

    text_context = "Surgical Steps Timeline:\n"
    for step in annotations:
        start = step.get("timestamp_start", "??:??")
        end = step.get("timestamp_end", "??:??")
        desc = step.get("visual_description", "")
        text_context += f"[{start} - {end}]: {desc}\n"

    return text_context


def detect_adverse_events(client, model_name, visual_context_text):
    """Send the visual step timeline to the LLM to detect complications."""
    try:
        result = chat_json_with_retry(
            client, model_name,
            [{"role": "user", "content": SAFETY_ANALYST_PROMPT +
              f"\nAnalyze this surgery:\n{visual_context_text}"}],
            response_format={"type": "json_object"},
            max_retries=3, retry_delay=2.0,
        )
        return result
    except Exception as e:
        print(f"  ⚠️ Adverse Event Detection failed: {e}")
        return None


def run_adverse_event_pipeline(vlm_input_dir, output_dir, log_filename,
                               aggregate_filename, model_name, base_url, api_key_env):
    """Main entry point for the adverse event detector stage."""
    print("\n=== ADVERSE EVENT DETECTION PIPELINE ===\n")

    client = resolve_and_get_client(api_key_env, base_url)
    if not client:
        return

    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_filename)
    aggregate_file_path = os.path.join(output_dir, aggregate_filename)
    progress_log = ProgressLog(log_file_path, ADVERSE_EVENT_LOG_HEADERS)

    processed_ids = progress_log.read_processed(['DETECTED', 'NO_EVENT'])
    print(f"ℹ️ Resuming... {len(processed_ids)} videos already checked.")

    if not os.path.exists(vlm_input_dir):
        print(f"❌ Input directory not found: {vlm_input_dir}")
        return

    input_files = [f for f in os.listdir(vlm_input_dir)
                   if f.endswith('.jsonl') and "all.jsonl" not in f]

    if not input_files:
        print("⚠️ No input VLM JSONL files found.")
        return

    print(f"Processing {len(input_files)} VLM files for safety analysis...\n")

    detected_count = 0
    clean_count = 0
    error_count = 0
    skipped_count = 0

    for filename in tqdm(input_files, desc="Scanning for Adverse Events"):
        video_id = os.path.splitext(filename)[0]

        if video_id in processed_ids:
            skipped_count += 1
            continue

        file_path = os.path.join(vlm_input_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line = f.readline()
                if not line:
                    continue
                vlm_data = json.loads(line)
        except Exception as e:
            tqdm.write(f"❌ Error reading {filename}: {e}")
            error_count += 1
            continue

        visual_context = extract_visual_steps(vlm_data)
        if not visual_context:
            tqdm.write(f"⚠️ No visual steps found for {video_id}, skipping.")
            continue

        result = detect_adverse_events(client, model_name, visual_context)
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        if result is None:
            progress_log.append(video_id=video_id, status="ERROR", event_count=0,
                                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            error_count += 1
            continue

        events = result.get("adverse_events", [])

        if events:
            detected_count += 1

            output_entry = {
                "video_id": vlm_data.get("video_id"),
                "original_filename": vlm_data.get("original_filename"),
                "status": "DETECTED",
                "video_url": vlm_data.get("video_url"),
                "video_title": vlm_data.get("video_title"),
                "download_date": vlm_data.get("download_date"),
                "adverse_events": events,
            }

            individual_output_path = os.path.join(output_dir, f"{video_id}.jsonl")
            with open(individual_output_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(output_entry) + "\n")

            with open(aggregate_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_entry) + "\n")

            progress_log.append(video_id=video_id, status="DETECTED", event_count=len(events),
                                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            tqdm.write(f"  🚨 Adverse Event Detected in {video_id}: {len(events)} event(s)")
        else:
            clean_count += 1
            progress_log.append(video_id=video_id, status="NO_EVENT", event_count=0,
                                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

    print("\n" + "=" * 30)
    print("   SAFETY ANALYSIS COMPLETED")
    print("=" * 30)
    print(f"🚨 Events Detected:  {detected_count}")
    print(f"✅ Clean Videos:     {clean_count}")
    print(f"⏭️ Skipped (Done):   {skipped_count}")
    print(f"❌ Errors:           {error_count}")
    print(f"📂 Output Folder:    {os.path.abspath(output_dir)}")

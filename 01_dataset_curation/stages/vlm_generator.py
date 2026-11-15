"""VLM generator stage: two-stage annotation of surgery videos.

Stage 1 (Gatekeeper): a fast OpenAI-compatible text model decides whether the
transcript is worth processing.
Stage 2 (Analyst): Google Gemini (google-genai SDK) watches the video via its
YouTube URL and generates step-by-step surgical annotations.
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from google.genai import types

from ..common.io import ProgressLog
from ..common.llm import resolve_and_get_client, chat_json_with_retry
from ..common.gemini import get_gemini_client
from ..prompts import GATEKEEPER_PROMPT, ANALYST_PROMPT

PROCESS_LOG_HEADERS = ["video_id", "original_filename", "status", "decision",
                       "confidence", "reasoning", "video_title", "url",
                       "download_date", "timestamp"]

GATEKEEPER_TRANSCRIPT_LIMIT = 25000
RATE_LIMIT_SLEEP_SECONDS = 60


def get_stable_id(filename_or_name):
    """Return the stable ID (filename without extension). With YouTube-ID
    naming this is simply the YouTube video ID."""
    if not filename_or_name:
        return None
    return os.path.splitext(str(filename_or_name))[0]


def check_transcript_quality(client, model_name, transcript_text):
    """Gatekeeper: decide if the transcript is worth processing (JSON out)."""
    try:
        prompt = GATEKEEPER_PROMPT.format(transcript_text=transcript_text[:GATEKEEPER_TRANSCRIPT_LIMIT])
        result = chat_json_with_retry(
            client, model_name,
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_retries=3, retry_delay=2.0,
        )
        if result is None:
            raise RuntimeError("Gatekeeper returned no valid JSON after retries")
        return result
    except Exception as e:
        return {"decision": "ERROR", "reasoning": str(e)}


def generate_vlm_entry(gemini_client, model_name, video_url, transcript_text):
    """Analyst (Google GenAI): watch the YouTube video URL and annotate it."""
    try:
        formatted_prompt = ANALYST_PROMPT.format(transcript_text=transcript_text)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(file_data=types.FileData(file_uri=video_url)),
                    types.Part.from_text(text=formatted_prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )

        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"  ⚠️ VLM Generation failed for {video_url}: {e}")
        return None


def _locate_transcript(refined_dir, transcript_name, video_id):
    """Locate the refined transcript file using several naming conventions."""
    t_path_1 = os.path.join(refined_dir, transcript_name)
    t_path_2 = os.path.join(refined_dir, f"{os.path.splitext(transcript_name)[0]}.txt")
    t_path_3 = os.path.join(refined_dir, f"{video_id}.txt")

    for candidate in (t_path_1, t_path_2, t_path_3):
        if os.path.exists(candidate):
            return candidate
    return None


def run_vlm_generation_pipeline(dataset_summary_path, refined_dir, output_dir,
                                aggregate_filename, log_filename,
                                gatekeeper_settings, analyst_model, analyst_api_key_env):
    """Main entry point for the VLM generation stage."""
    print("\n=== VLM DATASET GENERATION PIPELINE ===\n")

    gemini_client = get_gemini_client(analyst_api_key_env)
    if not gemini_client:
        return

    gatekeeper_client = resolve_and_get_client(
        gatekeeper_settings.api_key_env, gatekeeper_settings.base_url)
    if not gatekeeper_client:
        return

    os.makedirs(output_dir, exist_ok=True)

    aggregate_file_path = os.path.join(output_dir, aggregate_filename)
    log_file_path = os.path.join(output_dir, log_filename)
    progress_log = ProgressLog(log_file_path, PROCESS_LOG_HEADERS)

    if not os.path.exists(dataset_summary_path):
        print(f"❌ Summary file not found at: {dataset_summary_path}")
        return

    df = pd.read_csv(dataset_summary_path)
    if df.empty:
        print("⚠️ Summary file is empty.")
        return

    # Resume: files present on disk OR logged as ACCEPTED/REJECTED
    existing_files = set(os.listdir(output_dir))
    processed_log_ids = progress_log.read_processed(['ACCEPTED', 'REJECTED'])
    print(f"ℹ️ Resuming... {len(processed_log_ids)} completed videos found in log.")

    success_count = 0
    skip_count = 0
    error_count = 0
    already_done_count = 0

    print(f"Processing {len(df)} videos from summary...\n")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating VLM Data"):
        raw_video_name = str(row.get('video_name', ''))
        video_url = row.get('url', '')
        video_title = row.get('title', '')
        download_date = row.get('download_date', '')

        if not raw_video_name or not video_url:
            continue

        video_id = get_stable_id(raw_video_name)

        expected_filename = f"{video_id}.jsonl"
        if expected_filename in existing_files:
            already_done_count += 1
            continue

        if video_id in processed_log_ids:
            already_done_count += 1
            continue

        transcript_name = str(row.get('transcript_name', ''))
        final_t_path = _locate_transcript(refined_dir, transcript_name, video_id)
        if not final_t_path:
            error_count += 1
            continue

        try:
            with open(final_t_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        except Exception as e:
            error_count += 1
            continue

        # --- STEP 1: GATEKEEPER ---
        quality_result = check_transcript_quality(gatekeeper_client,
                                                  gatekeeper_settings.model,
                                                  transcript_text)
        decision = quality_result.get('decision', 'NO').upper()
        reason = quality_result.get('reasoning', 'No reason provided')
        confidence = quality_result.get('confidence_score', 0.0)

        if decision != 'YES':
            skip_count += 1
            progress_log.append(
                video_id=video_id, original_filename=raw_video_name, status="REJECTED",
                decision=decision, confidence=confidence, reasoning=reason,
                video_title=video_title, url=video_url, download_date=download_date,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            continue

        # --- STEP 2: GENERATOR (Google GenAI, YouTube URL) ---
        tqdm.write(f"  🎥 Analyze: {video_title}")
        vlm_data = generate_vlm_entry(gemini_client, analyst_model, video_url, transcript_text)
        tqdm.write(f" ⏳ Waiting {RATE_LIMIT_SLEEP_SECONDS}s to avoid rate limits...")
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        if vlm_data:
            final_entry = {
                "video_id": video_id,
                "original_filename": raw_video_name,
                "status": "SUCCESS",
                "video_url": video_url,
                "video_title": video_title,
                "download_date": download_date,
                "transcript_quality_check": quality_result,
                "vlm_annotations": vlm_data,
            }

            individual_file_path = os.path.join(output_dir, expected_filename)
            with open(individual_file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(final_entry) + "\n")

            with open(aggregate_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(final_entry) + "\n")

            progress_log.append(
                video_id=video_id, original_filename=raw_video_name, status="ACCEPTED",
                decision=decision, confidence=confidence, reasoning=reason,
                video_title=video_title, url=video_url, download_date=download_date,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            success_count += 1
        else:
            error_count += 1
            tqdm.write(f"  ❌ Failed to generate VLM data for {video_id}")
            progress_log.append(
                video_id=video_id, original_filename=raw_video_name, status="ERROR_GENERATION",
                decision=decision, confidence=confidence,
                reasoning="Model failed to generate valid JSON",
                video_title=video_title, url=video_url, download_date=download_date,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

    print("\n" + "=" * 30)
    print("   VLM GENERATION COMPLETED")
    print("=" * 30)
    print(f"✅ Accepted & Generated: {success_count}")
    print(f"⏩ Rejected (Quality):   {skip_count}")
    print(f"⏭️ Skipped (Already Done): {already_done_count}")
    print(f"❌ Errors:               {error_count}")
    print(f"📂 Output Folder:        {os.path.abspath(output_dir)}")

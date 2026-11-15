"""Refiner stage: LLM-based medical editing of Whisper transcripts.

Uses an OpenAI-compatible endpoint (default: Cerebras) with the medical
editor prompt. Files are keyed by YouTube ID: `{video_id}.json` in,
`{video_id}.txt` + `{video_id}_full_response.txt` out.
"""

import os
import json
import time
from tqdm import tqdm

from ..common.io import ProgressLog
from ..common.llm import resolve_and_get_client
from ..prompts import MEDICAL_EDITOR_SYSTEM_PROMPT

REFINEMENT_LOG_HEADERS = ["video_name", "transcript_name", "total_characters",
                          "total_words", "video_found", "success"]


def parse_llm_json_output(raw_output):
    """Strip thinking tags and extract the JSON array from an LLM response."""
    THINK_END_TAG = "</think>"
    if THINK_END_TAG in raw_output:
        after_think = raw_output.split(THINK_END_TAG, 1)[1].strip()
    else:
        after_think = raw_output.strip()

    start = after_think.find("[")
    end = after_think.rfind("]")

    if start != -1 and end != -1 and end > start:
        return after_think[start:end + 1].strip()
    return after_think


def format_time_to_mm_ss(seconds):
    s = int(seconds)
    return f"{s//60:02d}:{s%60:02d}"


def format_segments_to_txt(segments):
    lines = []
    for seg in segments:
        try:
            start = format_time_to_mm_ss(seg["start"])
            end = format_time_to_mm_ss(seg["end"])
            text = seg["text"].strip()
            lines.append(f"[{start} - {end}]: {text}")
        except KeyError:
            continue
    return "\n".join(lines)


def find_matching_video(videos_dir, video_id):
    """With YouTube-ID naming the video is simply `{video_id}.mp4`."""
    if not os.path.isdir(videos_dir):
        return "NOT_FOUND", False
    for ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.mpeg', '.mpg'):
        candidate = os.path.join(videos_dir, f"{video_id}{ext}")
        if os.path.exists(candidate):
            return os.path.basename(candidate), True
    return "NOT_FOUND", False


def refine_with_llm(client, model_name, text):
    """Send the JSON segments to the LLM for medical editing. Returns raw
    response text, or None (retryable failure)."""
    try:
        messages = [
            {"role": "system", "content": MEDICAL_EDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        response = client.chat.completions.create(messages=messages, model=model_name)
        return response.choices[0].message.content
    except Exception as e:
        if any(x in str(e).lower() for x in ["context", "limit", "too large"]):
            tqdm.write(f"Context limit exceeded: {e}")
            return '{"error": "MODEL CONTEXT LIMIT EXCEEDED"}'
        tqdm.write(f"API error: {e} → retrying in 10s...")
        time.sleep(10)
        return None


def run_refiner_pipeline(input_dir, videos_dir, refined_output_dir, log_path,
                         model_name, base_url, api_key_env, delay, max_files):
    """Main entry point for the refiner stage."""
    print("\n=== TRANSCRIPT REFINEMENT PIPELINE ===\n")
    full_response_dir = os.path.join(refined_output_dir, 'full_responses')

    os.makedirs(refined_output_dir, exist_ok=True)
    os.makedirs(full_response_dir, exist_ok=True)
    progress_log = ProgressLog(log_path, REFINEMENT_LOG_HEADERS)

    client = resolve_and_get_client(api_key_env, base_url)
    if not client:
        return

    try:
        raw = {os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.json')}
        done = {os.path.splitext(f)[0] for f in os.listdir(refined_output_dir) if f.endswith('.txt')}
    except FileNotFoundError as e:
        print(f"Directory not found: {e}")
        return

    to_process = sorted(raw - done)[:max_files]
    if not to_process:
        print("✅ All transcripts already refined!")
        return

    print(f"Processing {len(to_process)} transcript(s)...\n")

    for basename in tqdm(to_process, desc="Refining"):
        video_name, video_found = find_matching_video(videos_dir, basename)

        json_path = os.path.join(input_dir, f"{basename}.json")
        txt_path = os.path.join(refined_output_dir, f"{basename}.txt")
        full_path = os.path.join(full_response_dir, f"{basename}_full_response.txt")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        except Exception as e:
            tqdm.write(f"Failed to load {basename}.json → {e}")
            continue

        if not segments:
            continue

        payload = json.dumps(segments, indent=4)

        raw_response = None
        while raw_response is None:
            raw_response = refine_with_llm(client, model_name, payload)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(raw_response)

        success = False
        total_chars = total_words = 0
        try:
            cleaned_json = parse_llm_json_output(raw_response)
            refined_segments = json.loads(cleaned_json)
            total_words = sum(len(seg.get("text", "").split()) for seg in refined_segments)
            formatted_text = format_segments_to_txt(refined_segments)
            total_chars = len(formatted_text)

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            success = True
        except Exception as e:
            tqdm.write(f"Failed to parse/save {basename} → {e}")

        progress_log.append(
            video_name=video_name,
            transcript_name=f"{basename}.txt",
            total_characters=total_chars,
            total_words=total_words,
            video_found=video_found,
            success=success,
        )

        if basename != to_process[-1]:
            time.sleep(delay)

    print("\nRefinement process completed.")

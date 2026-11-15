"""Regenerate missing per-clip SFT/GRPO files in the final dataset.

Scans all ``clip_*.mp4`` across Train/Validation/Test and regenerates
``clip_XX_sft.jsonl`` / ``clip_XX_grpo.jsonl`` for clips missing either.
Reuses the same QA generation as the dataset formatter stage (3 MCQs per
clip, Pydantic-validated).

Usage:
    python -m video_pipeline.tools.generate_missing_grpo_sft <dataset_dir>
        [--dry-run] [--base-url ...] [--model ...] [--api-key ...] [--retries N]
"""

import os
import json
import time
import glob
import argparse

from ..common.llm import resolve_and_get_client
from ..stages.dataset_formatter import (
    QAList,
    build_sft_description,
    build_sft_qa,
    build_grpo_qa,
)
from ..prompts import QA_SYSTEM_PROMPT, build_qa_user_prompt
from ..config import build_settings


def find_clip_mp4s(dataset_dir):
    """Find all clip_XX.mp4 files across Train/Validation/Test splits."""
    clips = []
    for split in ["Train", "Validation", "Test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for vid_id in sorted(os.listdir(split_dir)):
            vid_dir = os.path.join(split_dir, vid_id)
            if not os.path.isdir(vid_dir):
                continue
            for mp4 in sorted(glob.glob(os.path.join(vid_dir, "clip_*.mp4"))):
                base_name = os.path.basename(mp4).replace(".mp4", "")
                clips.append({
                    "split": split,
                    "vid_id": vid_id,
                    "base_name": base_name,
                    "mp4_path": mp4,
                    "vid_dir": vid_dir,
                })
    return clips


def read_clip_metadata(vid_dir, base_name):
    meta_path = os.path.join(vid_dir, f"{base_name}.jsonl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.loads(f.readline())


def generate_qa_pairs(client, model_name, clip_data, max_retries=3):
    """Call LLM to generate 3 validated QA pairs for a clip."""
    result = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": build_qa_user_prompt(clip_data)},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2048,
            )
            return QAList(**json.loads(response.choices[0].message.content)).qa_pairs
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [attempt {attempt + 1}] Validation error: {e}")
            time.sleep(2)
    return None


def write_sft_file(vid_dir, base_name, video_rel_path, clip_data, qa_pairs):
    path = os.path.join(vid_dir, f"{base_name}_sft.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(build_sft_description(clip_data, video_rel_path)) + "\n")
        for qa in qa_pairs:
            f.write(json.dumps(build_sft_qa(video_rel_path, qa)) + "\n")


def write_grpo_file(vid_dir, base_name, video_rel_path, qa_pairs):
    path = os.path.join(vid_dir, f"{base_name}_grpo.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(build_grpo_qa(video_rel_path, qa)) + "\n")


def generate_missing(dataset_dir, base_url, model, api_key_env, max_retries=3,
                     dry_run=False, target=None):
    """Generate missing per-clip SFT/GRPO files in the final dataset.

    In separated mode (dataset_sft / dataset_grpo), set target="sft" or "grpo"
    to only check/generate that type. Legacy single-dataset leaves target=None.
    """
    all_clips = find_clip_mp4s(dataset_dir)
    print(f"Found {len(all_clips)} total clip_*.mp4 files across all splits.\n")

    # Determine what to check based on target
    if target == "sft":
        missing_sft = [c for c in all_clips
                       if not os.path.exists(os.path.join(c["vid_dir"], f"{c['base_name']}_sft.jsonl"))]
        missing_grpo = []
    elif target == "grpo":
        missing_sft = []
        missing_grpo = [c for c in all_clips
                       if not os.path.exists(os.path.join(c["vid_dir"], f"{c['base_name']}_grpo.jsonl"))]
    else:
        missing_sft = [c for c in all_clips
                       if not os.path.exists(os.path.join(c["vid_dir"], f"{c['base_name']}_sft.jsonl"))]
        missing_grpo = [c for c in all_clips
                       if not os.path.exists(os.path.join(c["vid_dir"], f"{c['base_name']}_grpo.jsonl"))]

    print(f"Missing _sft.jsonl: {len(missing_sft)}")
    print(f"Missing _grpo.jsonl: {len(missing_grpo)}")
    print()

    if dry_run:
        for label, items in [("SFT", missing_sft), ("GRPO", missing_grpo)]:
            for c in items:
                print(f"  {c['split']}/{c['vid_id']}/{c['base_name']}_{label.lower()}.jsonl")
        return None

    clips_to_process = {}
    for c in missing_sft + missing_grpo:
        clips_to_process.setdefault((c["split"], c["vid_id"], c["base_name"]), c)

    if not clips_to_process:
        if target == "sft":
            print("All clips already have _sft.jsonl. Nothing to do.")
        elif target == "grpo":
            print("All clips already have _grpo.jsonl. Nothing to do.")
        else:
            print("All clips already have both _sft.jsonl and _grpo.jsonl. Nothing to do.")
        return None

    print(f"Clips to process: {len(clips_to_process)}\n")

    client = resolve_and_get_client(api_key_env, base_url)
    if not client:
        return None

    success_count = error_count = 0
    missing_sft_keys = {(c["split"], c["vid_id"], c["base_name"]) for c in missing_sft}
    missing_grpo_keys = {(c["split"], c["vid_id"], c["base_name"]) for c in missing_grpo}

    for (split, vid_id, base_name), clip in clips_to_process.items():
        clip_data = read_clip_metadata(clip["vid_dir"], base_name)
        if not clip_data:
            print(f"[SKIP] {split}/{vid_id}/{base_name} — no metadata .jsonl found")
            error_count += 1
            continue

        video_rel_path = f"{vid_id}/{base_name}.mp4"
        print(f"[{split}] {vid_id}/{base_name} ... ", end="", flush=True)

        qa_pairs = generate_qa_pairs(client, model, clip_data, max_retries=max_retries)
        if qa_pairs is None:
            print("FAILED")
            error_count += 1
            continue

        key = (split, vid_id, base_name)
        if key in missing_sft_keys:
            write_sft_file(clip["vid_dir"], base_name, video_rel_path, clip_data, qa_pairs)
        if key in missing_grpo_keys:
            write_grpo_file(clip["vid_dir"], base_name, video_rel_path, qa_pairs)

        print(f"OK (3 QA pairs)")
        success_count += 1

    print(f"\nDone. Success: {success_count}, Errors: {error_count}")
    return {"success": success_count, "errors": error_count}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate missing clip SFT/GRPO files using an LLM.")
    parser.add_argument("dataset_dir", help="Path to the dataset root (containing Train/Validation/Test)")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key (default: FORMATTER_API_KEY)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--retries", type=int, default=None, help="Max retries per clip")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what's missing, don't generate")
    parser.add_argument("--target", choices=["sft", "grpo"], default=None,
                        help="In separated mode, only process SFT or GRPO (default: both for legacy)")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    missing_settings = settings.generate_missing_grpo_sft
    base_url = args.base_url or missing_settings.base_url
    model_name = args.model or missing_settings.model
    retries = args.retries or missing_settings.extra.get("max_retries", 3)

    generate_missing(args.dataset_dir, base_url=base_url, model=model_name,
                     api_key_env=missing_settings.api_key_env,
                     max_retries=retries, dry_run=args.dry_run, target=args.target)


if __name__ == "__main__":
    main()

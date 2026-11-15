"""Dataset formatter stage: generates SFT & GRPO training files from clips.

For each clip, an OpenAI-compatible LLM generates 3 multiple-choice QA pairs
(step_identification, visual_observation, instrument_identification). Outputs
`{split}_{format}.jsonl` files with an 80/10/10 parent-video-level split.

Output schemas are preserved exactly (same as the original implementation):
- SFT: 4 records per clip (1 description + 3 QA), `messages` format
- GRPO: 3 records per clip, `prompt`/`correct_answer`/`question_type`/
  `reference_reasoning`/`reward_type` format
- `dataset_split.json`: {"train": [...], "val": [...], "test": [...]} (seed 42)
"""

import os
import json
import random
import time
from tqdm import tqdm
from pydantic import BaseModel, ValidationError

from ..common.io import ProgressLog
from ..common.llm import resolve_and_get_client, chat_json_with_retry
from ..prompts import QA_SYSTEM_PROMPT, build_qa_user_prompt, GRPO_JSON_INSTRUCTION

FORMATTER_LOG_HEADERS = ["clip_id", "status", "timestamp"]
SPLIT_SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1
# All question types now deterministic (visual_observation previously llm_judge, now fixed)
DETERMINISTIC_QUESTION_TYPES = ["step_identification", "visual_observation", "instrument_identification"]


class QAPair(BaseModel):
    question_type: str
    question: str
    options: dict[str, str]
    correct_answer: str
    reference_reasoning: str


class QAList(BaseModel):
    qa_pairs: list[QAPair]


def manage_train_val_test_split(clips_dir, output_dir, split_filename):
    """Create (or load) the parent-video-level split, preventing data leakage."""
    split_filepath = os.path.join(output_dir, split_filename)

    if os.path.exists(split_filepath):
        with open(split_filepath, 'r') as f:
            return json.load(f)

    parent_video_ids = [d for d in os.listdir(clips_dir)
                        if os.path.isdir(os.path.join(clips_dir, d))]
    random.seed(SPLIT_SEED)
    random.shuffle(parent_video_ids)

    total = len(parent_video_ids)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": parent_video_ids[:train_end],
        "val": parent_video_ids[train_end:val_end],
        "test": parent_video_ids[val_end:],
    }

    with open(split_filepath, 'w') as f:
        json.dump(splits, f, indent=4)

    print(f"Created new dataset splits: {len(splits['train'])} Train, "
          f"{len(splits['val'])} Val, {len(splits['test'])} Test parent videos.")
    return splits


def get_split_for_clip(parent_video_id, splits):
    for split_name, ids in splits.items():
        if parent_video_id in ids:
            return split_name
    return "train"  # Default fallback


def generate_qa_pairs(client, model_name, clip_data, max_retries):
    """Ask the LLM for 3 validated MCQ QA pairs about a clip."""
    result = chat_json_with_retry(
        client, model_name,
        [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": build_qa_user_prompt(clip_data)},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=2048,
        max_retries=max_retries,
        retry_delay=2.0,
        on_retry=lambda attempt, e: tqdm.write(
            f"⚠️ Validation failed on attempt {attempt} for {clip_data.get('clip_id')}: {e}"),
    )
    if result is None:
        return None
    try:
        return QAList(**result).qa_pairs
    except ValidationError as e:
        tqdm.write(f"⚠️ Schema validation failed for {clip_data.get('clip_id')}: {e}")
        return None


def build_sft_description(clip_data, video_path):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Describe what is happening in this cataract surgical video clip."},
            ]},
            {"role": "assistant", "content": clip_data.get("visual_description", "")},
        ]
    }


def build_sft_qa(video_path, qa):
    options_text = "\n".join([f"{k}) {v}" for k, v in qa.options.items()])
    question_text = f"{qa.question}\n{options_text}\n\nProvide your reasoning first, then state your answer."
    assistant_answer = f"{qa.reference_reasoning} Therefore the answer is {qa.correct_answer}) {qa.options[qa.correct_answer]}."
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question_text},
            ]},
            {"role": "assistant", "content": assistant_answer},
        ]
    }


def build_grpo_qa(video_path, qa):
    options_text = "\n".join([f"{k}) {v}" for k, v in qa.options.items()])
    question_text = f"{qa.question}\n{options_text}\n\n{GRPO_JSON_INSTRUCTION}"
    # All rewards now deterministic (previously visual_observation was llm_judge)
    reward_type = "deterministic"
    return {
        "prompt": [
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question_text},
            ]},
        ],
        "correct_answer": qa.correct_answer,
        "question_type": qa.question_type,
        "reference_reasoning": qa.reference_reasoning,
        "reward_type": reward_type,
    }


def _gather_clip_files(clips_dir):
    """Collect all per-clip JSONL metadata paths from the clips directory."""
    clips = []
    for parent_id in os.listdir(clips_dir):
        parent_dir = os.path.join(clips_dir, parent_id)
        if not os.path.isdir(parent_dir):
            continue
        for file in os.listdir(parent_dir):
            if file.endswith('.jsonl'):
                clips.append(os.path.join(parent_dir, file))
    return clips


def run_dataset_formatter(clips_dir, output_dir, model_name, base_url, api_key_env,
                          log_filename, split_filename, max_retries):
    """Main entry point for the dataset formatter stage."""
    print("\n=== DATASET FORMATTER (SFT & GRPO) ===\n")

    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_filename)
    progress_log = ProgressLog(log_file_path, FORMATTER_LOG_HEADERS)

    splits = manage_train_val_test_split(clips_dir, output_dir, split_filename)
    processed_clips = progress_log.read_processed(['SUCCESS'])
    print(f"ℹ️ Resuming... {len(processed_clips)} clips already formatted.")

    client = resolve_and_get_client(api_key_env, base_url)
    if not client:
        return

    all_clips = _gather_clip_files(clips_dir)

    if not all_clips:
        print("⚠️ No clips found in the clip directory.")
        return

    file_handles = {}
    for split_type in ["train", "val", "test"]:
        for format_type in ["sft", "grpo"]:
            file_path = os.path.join(output_dir, f"{split_type}_{format_type}.jsonl")
            file_handles[f"{split_type}_{format_type}"] = open(file_path, 'a', encoding='utf-8')

    try:
        for clip_path in tqdm(all_clips, desc="Generating VLM Formats"):
            with open(clip_path, 'r', encoding='utf-8') as f:
                try:
                    clip_data = json.loads(f.readline())
                except json.JSONDecodeError:
                    continue

            clip_id = clip_data.get("clip_id")
            parent_video_id = clip_data.get("parent_video_id")
            clip_filename = clip_data.get("clip_filename")

            if not clip_id or clip_id in processed_clips:
                continue

            split_group = get_split_for_clip(parent_video_id, splits)
            video_path = f"{parent_video_id}/{clip_filename}"

            generated_qa_pairs = generate_qa_pairs(client, model_name, clip_data, max_retries)
            if generated_qa_pairs is None:
                progress_log.append(clip_id=clip_id, status="ERROR_LLM",
                                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
                continue

            sft_handle = file_handles[f"{split_group}_sft"]
            grpo_handle = file_handles[f"{split_group}_grpo"]

            # Task 1: SFT description task (deterministic from clip metadata)
            sft_handle.write(json.dumps(build_sft_description(clip_data, video_path)) + "\n")

            # Task 2: LLM-generated QA pairs
            for qa in generated_qa_pairs:
                sft_handle.write(json.dumps(build_sft_qa(video_path, qa)) + "\n")
                grpo_handle.write(json.dumps(build_grpo_qa(video_path, qa)) + "\n")

            progress_log.append(clip_id=clip_id, status="SUCCESS",
                                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    finally:
        for handle in file_handles.values():
            handle.close()

    print("\n" + "=" * 30)
    print("   DATASET FORMATTING COMPLETED")
    print("=" * 30)
    print(f"📂 Output Folder: {os.path.abspath(output_dir)}")
    print(f"📊 Splits used: {len(splits['train'])} Train, {len(splits['val'])} Val, "
          f"{len(splits['test'])} Test parent videos.")

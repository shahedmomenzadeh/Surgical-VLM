"""Refine too-similar instrument distractors in existing SFT/GRPO files.

An LLM judge examines the 4 options of every instrument_identification
question in the dataset's SFT/GRPO files and replaces distractors that are
too visually/functionally similar to the correct answer. Progress is tracked
in ``<dataset_dir>/processed_clips.txt`` for resumption.

Usage:
    python -m video_pipeline.tools.refine_instrument_options <dataset_dir>
        [--dry-run] [--base-url ...] [--model ...] [--api-key ...] [--retries N]
"""

import os
import re
import json
import time
import argparse
from typing import Dict

from pydantic import BaseModel, ValidationError, Field, field_validator

from ..common.llm import resolve_and_get_client
from ..config import build_settings
from ..prompts import GRPO_JSON_INSTRUCTION

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas
# ══════════════════════════════════════════════════════════════════════════════

class InstrumentEvaluation(BaseModel):
    are_options_too_similar: bool
    similarity_explanation: str = Field(default="")
    new_options: Dict[str, str] = Field(default_factory=dict)

    @field_validator('new_options', mode='before')
    @classmethod
    def ensure_dict(cls, v):
        if isinstance(v, list) and len(v) == 0:
            return {}
        return v


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """You are an expert surgical instrument specialist evaluating multiple-choice questions for a cataract surgery VLM training dataset.

Your task: examine the 4 multiple-choice options for an instrument_identification question and determine if the distractors (wrong answers) are too similar to the correct answer, making the question artificially easy.

DEFINITION OF "TOO SIMILAR":
Two instruments are "too similar" when they could reasonably be confused even by someone watching the video. This includes:
- Instruments that look visually similar (e.g., Sinskey hook vs Chopper — both are thin metallic hooks)
- Instruments that serve overlapping functions (e.g., Phacoemulsification handpiece vs Irrigation/aspiration handpiece)
- Different names for essentially the same tool (e.g., "Keratome" vs "Keratome blade")
- When 2+ distractors belong to the same instrument family as the correct answer

NOT too similar: instruments from clearly different functional categories (e.g., forceps vs cannula vs blade vs IOL injector).

OUTPUT FORMAT:
You must output ONLY a valid JSON object exactly matching this schema:
{
  "are_options_too_similar": true/false,
  "similarity_explanation": "A short sentence explaining why they are or are not similar.",
  "new_options": {
     "A": "Option text",
     "B": "Option text",
     "C": "Option text",
     "D": "Option text"
  }
}

RULES:
1. If `are_options_too_similar` is true, provide 3 NEW distractor options from clearly different instrument categories than the correct answer. Keep the correct answer letter and text EXACTLY as they are in `new_options`.
2. If `are_options_too_similar` is false, `new_options` MUST be an empty dictionary: {}.
3. You must always include the `similarity_explanation` string.
"""


def build_user_prompt(correct_letter, correct_text, options):
    option_lines = "\n".join([f"{k}) {v}" for k, v in options.items()])
    return f"""Correct answer: {correct_letter}) {correct_text}

All options:
{option_lines}

Evaluate whether the 3 distractor options (the ones that are NOT {correct_letter}) are too visually or functionally similar to the correct answer. If they are, propose 3 replacement distractors from clearly different instrument families."""


# ══════════════════════════════════════════════════════════════════════════════
# SFT/GRPO Parsing & Writing
# ══════════════════════════════════════════════════════════════════════════════

def parse_sft_line(line_json):
    messages = line_json.get("messages", [])
    if len(messages) < 2:
        return None

    assistant_content = messages[1].get("content", "")
    user_content = messages[0].get("content", [])
    if not isinstance(user_content, list) or len(user_content) < 2:
        return None

    text_content = None
    for item in user_content:
        if isinstance(item, dict) and item.get("type") == "text":
            text_content = item.get("text", "")
            break

    if text_content is None:
        return None

    if not re.search(r'\bA\)', text_content) or not (re.search(r'Provide your reasoning', text_content) or re.search(r'Respond ONLY with a JSON', text_content)):
        return None

    options = {}
    for match in re.finditer(r'^([A-D])\)\s*(.+?)$', text_content, re.MULTILINE):
        letter = match.group(1)
        option_text = match.group(2).strip()
        if option_text:
            options[letter] = option_text

    if not options or len(options) != 4:
        return None

    answer_match = re.search(r'Therefore the answer is ([A-D])\)\s*(.+?)\.', assistant_content)
    if not answer_match:
        return None

    correct_letter = answer_match.group(1)
    correct_text = answer_match.group(2).strip()

    if correct_letter not in options:
        return None

    return text_content, options, correct_letter, correct_text, assistant_content


def build_options_text(options):
    return "\n".join([f"{k}) {v}" for k, v in sorted(options.items())])


def extract_question_only(text_content):
    match = re.match(r'^(.+?)(?=\nA\)|\n\nProvide your reasoning|\n\nRespond ONLY with a JSON)', text_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text_content


def extract_reference_reasoning(assistant_content, correct_letter, correct_text):
    answer_suffix = f" Therefore the answer is {correct_letter}) {correct_text}."
    if assistant_content.endswith(answer_suffix):
        return assistant_content[:-len(answer_suffix)]
    match = re.match(r'^(.+?) Therefore the answer is [A-D]\)\s*.+?\.\s*$', assistant_content)
    if match:
        return match.group(1).strip()
    return assistant_content


def build_sft_assistant(reasoning, letter, option_text):
    return f"{reasoning} Therefore the answer is {letter}) {option_text}."


def build_sft_entry(video_rel_path, question, options, reasoning, correct_letter, correct_text):
    options_text = build_options_text(options)
    question_text = f"{question}\n{options_text}\n\nProvide your reasoning first, then state your answer."
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "video", "video": video_rel_path},
                {"type": "text", "text": question_text},
            ]},
            {"role": "assistant", "content": build_sft_assistant(reasoning, correct_letter, correct_text)},
        ]
    }


def build_grpo_entry(video_rel_path, question, options, correct_letter, reasoning):
    options_text = build_options_text(options)
    question_text = f"{question}\n{options_text}\n\n{GRPO_JSON_INSTRUCTION}"
    return {
        "prompt": [
            {"role": "user", "content": [
                {"type": "video", "video": video_rel_path},
                {"type": "text", "text": question_text},
            ]},
        ],
        "correct_answer": correct_letter,
        "question_type": "instrument_identification",
        "reference_reasoning": reasoning,
        "reward_type": "deterministic",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Core Logic
# ══════════════════════════════════════════════════════════════════════════════

def find_sft_files(dataset_dir):
    sft_files = []
    for split in ["Train", "Validation", "Test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for vid_id in sorted(os.listdir(split_dir)):
            vid_dir = os.path.join(split_dir, vid_id)
            if not os.path.isdir(vid_dir):
                continue
            for fname in sorted(os.listdir(vid_dir)):
                if fname.startswith("clip_") and fname.endswith("_sft.jsonl"):
                    sft_files.append({
                        "split": split,
                        "vid_id": vid_id,
                        "vid_dir": vid_dir,
                        "sft_path": os.path.join(vid_dir, fname),
                        "base_name": fname.replace("_sft.jsonl", ""),
                    })
    return sft_files


def evaluate_options(client, model_name, options, correct_letter, correct_text, max_retries=3):
    user_prompt = build_user_prompt(correct_letter, correct_text, options)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1024,
            )
            return InstrumentEvaluation(**json.loads(response.choices[0].message.content))
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  [attempt {attempt + 1}] Validation error: {e}")
        except Exception as e:
            print(f"  [attempt {attempt + 1}] LLM error: {e}")
            time.sleep(2)
    return None


def refine_instrument_options(dataset_dir, base_url, model, api_key_env, max_retries=3,
                              dry_run=False, grpo_dataset_dir=None):
    """Refine too-similar instrument distractors across the final dataset.

    In separated mode (dataset_sft + dataset_grpo), set `grpo_dataset_dir` to the GRPO
    dataset root; this updates the GRPO counterpart at `grpo_dataset_dir/{split}/{vid_id}/`
    instead of the same folder.
    """
    sft_files = find_sft_files(dataset_dir)
    print(f"Found {len(sft_files)} clip SFT files across all splits.\n")

    progress_log = os.path.join(dataset_dir, "processed_clips.txt")
    processed_set = set()
    if os.path.exists(progress_log):
        with open(progress_log, "r", encoding="utf-8") as f:
            processed_set = set(line.strip() for line in f if line.strip())
        print(f"Resuming progress. Already processed {len(processed_set)} clips.\n")

    client = resolve_and_get_client(api_key_env, base_url)
    if not client:
        return None

    total_evaluated = total_replaced = total_errors = skipped_short = 0

    for idx, item in enumerate(sft_files):
        label = f"{item['split']}/{item['vid_id']}/{item['base_name']}"

        if label in processed_set:
            continue

        with open(item["sft_path"], "r", encoding="utf-8") as f:
            sft_lines = [json.loads(line) for line in f if line.strip()]

        if len(sft_lines) < 4:
            skipped_short += 1
            continue

        parsed = parse_sft_line(sft_lines[3])
        if parsed is None:
            continue

        question_text, options, correct_letter, correct_text, assistant_content = parsed
        reasoning = extract_reference_reasoning(assistant_content, correct_letter, correct_text)
        question_only = extract_question_only(question_text)
        video_rel_path = f"{item['vid_id']}/{item['base_name']}.mp4"

        print(f"[{idx + 1}/{len(sft_files)}] {label} ... ", end="", flush=True)
        total_evaluated += 1

        evaluation = evaluate_options(client, model_name=model, options=options,
                                      correct_letter=correct_letter, correct_text=correct_text,
                                      max_retries=max_retries)
        if evaluation is None:
            print("LLM FAILED")
            total_errors += 1
            continue

        if not evaluation.are_options_too_similar:
            print("OK (options are distinct enough)")
            if not dry_run:
                with open(progress_log, "a", encoding="utf-8") as f:
                    f.write(label + "\n")
            continue

        new_options = evaluation.new_options
        new_options[correct_letter] = correct_text
        if set(new_options.keys()) != {"A", "B", "C", "D"}:
            print("WARN: LLM returned invalid option keys, skipping")
            total_errors += 1
            continue

        new_options = {k: new_options[k] for k in ["A", "B", "C", "D"]}

        replaced = ", ".join(
            f"{lk}) {lo} -> {ln}"
            for lk, lo, ln in zip(sorted(options.keys()),
                                  [options[k] for k in sorted(options.keys())],
                                  [new_options[k] for k in sorted(new_options.keys())])
            if lk != correct_letter
        )
        print(f"REPLACED ({evaluation.similarity_explanation[:80]}...) Changes: {replaced}")

        total_replaced += 1

        if dry_run:
            continue

        sft_lines[3] = build_sft_entry(video_rel_path, question_only, new_options,
                                       reasoning, correct_letter, correct_text)

        with open(item["sft_path"], "w", encoding="utf-8") as f:
            for sft_line in sft_lines:
                f.write(json.dumps(sft_line) + "\n")

        # Resolve GRPO counterpart: same folder (legacy) or mirrored GRPO dataset (separated)
        if grpo_dataset_dir:
            grpo_path = os.path.join(grpo_dataset_dir, item["split"], item["vid_id"], f"{item['base_name']}_grpo.jsonl")
        else:
            grpo_path = os.path.join(item["vid_dir"], f"{item['base_name']}_grpo.jsonl")
        if os.path.exists(grpo_path):
            with open(grpo_path, "r", encoding="utf-8") as f:
                grpo_lines = [json.loads(line) for line in f if line.strip()]

            if len(grpo_lines) >= 3:
                grpo_lines[2] = build_grpo_entry(video_rel_path, question_only, new_options,
                                                 correct_letter, reasoning)
                with open(grpo_path, "w", encoding="utf-8") as f:
                    for grpo_line in grpo_lines:
                        f.write(json.dumps(grpo_line) + "\n")

        with open(progress_log, "a", encoding="utf-8") as f:
            f.write(label + "\n")

    print(f"\n{'=' * 50}")
    print(f"  INSTRUMENT OPTIONS REFINEMENT COMPLETE")
    print(f"{'=' * 50}")
    print(f"Evaluated this run: {total_evaluated}")
    print(f"Replaced (too similar): {total_replaced}")
    print(f"Skipped (short files): {skipped_short}")
    print(f"Errors: {total_errors}")
    print(f"Mode: {'DRY RUN (no files modified)' if dry_run else 'FILES MODIFIED'}")
    return {"evaluated": total_evaluated, "replaced": total_replaced,
            "skipped_short": skipped_short, "errors": total_errors}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Refine instrument_identification distractor options across the dataset.")
    parser.add_argument("dataset_dir", help="Path to the SFT dataset root (containing Train/Validation/Test)")
    parser.add_argument("--grpo-dataset-dir", default=None, help="Path to GRPO dataset root for mirroring (separated mode). If omitted, uses same dir (legacy).")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key (default: FORMATTER_API_KEY)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--retries", type=int, default=None, help="Max retries per clip")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be changed, don't modify files")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    refine_settings = settings.refine_instrument_options
    base_url = args.base_url or refine_settings.base_url
    model_name = args.model or refine_settings.model
    retries = args.retries or refine_settings.extra.get("max_retries", 3)
    # Fallback to config separated dirs if not explicitly passed and single legacy path looks like old dataset
    grpo_dir = args.grpo_dataset_dir
    if grpo_dir is None and hasattr(settings, 'dataset_grpo_dir'):
        # If dataset_dir == dataset_sft_dir (separated mode invoked with SFT path), use GRPO counterpart
        try:
            if os.path.abspath(args.dataset_dir) == os.path.abspath(settings.dataset_sft_dir):
                grpo_dir = settings.dataset_grpo_dir
        except Exception:
            pass

    refine_instrument_options(args.dataset_dir, base_url=base_url, model=model_name,
                              api_key_env=refine_settings.api_key_env,
                              max_retries=retries, dry_run=args.dry_run,
                              grpo_dataset_dir=grpo_dir)


if __name__ == "__main__":
    main()

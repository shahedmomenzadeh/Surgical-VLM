"""Merge clips and SFT/GRPO training data into a final dataset layout.

Builds separated SFT/GRPO datasets (videos mirrored in both):
    dataset_sft/
    ├── Train/Validation/Test/{video_id}/
    │   ├── clip_XX.mp4 ... clip_XX.jsonl
    │   └── clip_XX_sft.jsonl + full_video_sft.jsonl
    dataset_grpo/
    ├── Train/Validation/Test/{video_id}/
    │   ├── clip_XX.mp4 ... clip_XX.jsonl
    │   └── clip_XX_grpo.jsonl
Legacy single-dataset mode still supported via --out.

Usage:
    python -m video_pipeline.tools.merge_dataset [--clips-dir DIR]
                                                 [--training-dir DIR]
                                                 [--out DIR]  # legacy single
                                                 [--out-sft DIR --out-grpo DIR]  # separated
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

from ..common.utils import normalize_id
from ..config import build_settings

SPLIT_FOLDER_NAMES = [("train", "Train"), ("val", "Validation"), ("test", "Test")]


def _materialize_file(src: Path, dst: Path):
    """Hardlink with copy fallback (mirrors videos to save space)."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    try:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def merge_and_verify_datasets(clip_dataset_path, training_dataset_path, output_dataset_path):
    """Legacy single-dataset merge (videos + SFT + GRPO together)."""
    return merge_and_verify_datasets_separated(
        clip_dataset_path, training_dataset_path,
        output_sft_path=output_dataset_path,
        output_grpo_path=output_dataset_path,
        single_dataset_mode=True,
    )


def merge_and_verify_datasets_separated(clip_dataset_path, training_dataset_path, output_sft_path=None, output_grpo_path=None, single_dataset_mode=False):
    """
    Separated merge: videos+metadata mirrored to both SFT and GRPO datasets;
    SFT records → SFT dataset, GRPO records → GRPO dataset.
    If single_dataset_mode True and both paths equal, behaves like legacy.
    """
    clip_dir = Path(clip_dataset_path)
    train_dir = Path(training_dataset_path)
    # Resolve outputs: if one is None, use the other
    if output_sft_path is None and output_grpo_path is None:
        raise ValueError("At least one output path required")
    if output_sft_path is None:
        output_sft_path = output_grpo_path
    if output_grpo_path is None:
        output_grpo_path = output_sft_path
    sft_dir = Path(output_sft_path)
    grpo_dir = Path(output_grpo_path)
    is_separated = not single_dataset_mode and sft_dir.resolve() != grpo_dir.resolve()

    print(f"=== Starting Dataset Merge {'(Separated SFT/GRPO)' if is_separated else '(Single)'} ===")
    print(f"Input clips: {clip_dir}")
    print(f"Input training data: {train_dir}")
    if is_separated:
        print(f"Target SFT dataset: {sft_dir}")
        print(f"Target GRPO dataset: {grpo_dir}\n")
    else:
        print(f"Target dataset: {sft_dir}\n")

    split_json_path = train_dir / "dataset_split.json"
    if not split_json_path.exists():
        raise FileNotFoundError(f"❌ Split file not found at: {split_json_path}")

    with open(split_json_path, 'r', encoding='utf-8') as f:
        splits_data = json.load(f)

    normalized_split_map = {}
    for split_key, target_folder in SPLIT_FOLDER_NAMES:
        for video_id in splits_data.get(split_key, []):
            normalized_split_map[normalize_id(video_id)] = target_folder

    # Ensure split roots exist
    for d in ({sft_dir, grpo_dir} if is_separated else {sft_dir}):
        for folder in ["Train", "Validation", "Test"]:
            (d / folder).mkdir(parents=True, exist_ok=True)

    print("--- STEP 1: Copying Clips and Base JSONL (mirrored to both) ---")

    if not clip_dir.exists():
        raise FileNotFoundError(f"❌ Clip directory not found at: {clip_dir}")

    clip_folders = [f for f in clip_dir.iterdir() if f.is_dir()]
    physical_folder_map = {}

    for video_folder in tqdm(clip_folders, desc="Copying original clips"):
        files = [f for f in video_folder.iterdir() if f.is_file()]
        if not files:
            print(f"⚠️ Skipping empty clip folder: {video_folder.name}")
            continue

        actual_folder_name = video_folder.name
        norm_key = normalize_id(actual_folder_name)

        physical_folder_map[norm_key] = actual_folder_name

        if norm_key not in normalized_split_map:
            target_split = "Train"
            normalized_split_map[norm_key] = target_split
        else:
            target_split = normalized_split_map[norm_key]

        # Mirrored dirs (or single)
        target_dirs = [sft_dir / target_split / actual_folder_name, grpo_dir / target_split / actual_folder_name] if is_separated else [sft_dir / target_split / actual_folder_name]
        # Deduplicate if same path
        target_dirs = list(dict.fromkeys(target_dirs))
        for td in target_dirs:
            td.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            # Clip videos/metadatas are mirrored; SFT/GRPO not expected here but if present, handle accordingly
            for td in target_dirs:
                dest_path = td / file_path.name
                if dest_path.exists():
                    continue
                # Video -> hardlink, others -> copy2
                if file_path.suffix == ".mp4":
                    _materialize_file(file_path, dest_path)
                else:
                    shutil.copy2(file_path, dest_path)

    print("\n--- STEP 2: Distributing SFT and GRPO files ---")

    total_global_errors = 0
    total_global_success = 0

    for split_key, target_folder in SPLIT_FOLDER_NAMES:
        for format_type in ["sft", "grpo"]:
            source_jsonl = train_dir / f"{split_key}_{format_type}.jsonl"

            if not source_jsonl.exists():
                continue

            # Route to correct dataset
            if is_separated:
                if format_type == "sft":
                    out_root = sft_dir
                else:
                    out_root = grpo_dir
            else:
                out_root = sft_dir

            print(f"\nProcessing {source_jsonl.name} -> {out_root.name}/ ...")

            lines_in_source = 0
            successful_writes = 0

            with open(source_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    lines_in_source += 1

                    try:
                        data = json.loads(line)

                        if format_type == "sft":
                            video_ref = data["messages"][0]["content"][0]["video"]
                        else:
                            video_ref = data["prompt"][0]["content"][0]["video"]

                        if "/" not in video_ref:
                            print(f"❌ Format Error: Unrecognized video path '{video_ref}'")
                            total_global_errors += 1
                            continue

                        raw_parent_id, clip_filename = video_ref.rsplit("/", 1)
                        clip_base = Path(clip_filename).stem
                        norm_parent = normalize_id(raw_parent_id)

                        expected_split = normalized_split_map.get(norm_parent)
                        actual_physical_folder = physical_folder_map.get(norm_parent)

                        if not expected_split or not actual_physical_folder:
                            print(f"❌ Mapping Error: Could not determine split or physical folder for '{raw_parent_id}'")
                            total_global_errors += 1
                            continue

                        target_dir = out_root / expected_split / actual_physical_folder
                        expected_mp4 = target_dir / clip_filename

                        if not expected_mp4.exists():
                            print(f"❌ Verification Failed: Original clip missing at '{expected_mp4}'.")
                            total_global_errors += 1
                            continue

                        target_jsonl = target_dir / f"{clip_base}_{format_type}.jsonl"

                        with open(target_jsonl, 'a', encoding='utf-8') as out_f:
                            out_f.write(json.dumps(data, ensure_ascii=False) + "\n")

                        successful_writes += 1
                        total_global_success += 1

                    except json.JSONDecodeError:
                        print("❌ JSON Error: Could not decode line.")
                        total_global_errors += 1
                    except KeyError as e:
                        print(f"❌ Schema Error: Missing expected key {e} in {source_jsonl.name}")
                        total_global_errors += 1

            print(f"  -> Count Verification for {source_jsonl.name}:")
            print(f"     Expected samples (lines in source): {lines_in_source}")
            print(f"     Successfully generated files:       {successful_writes}")

            if lines_in_source == successful_writes:
                print("     ✅ Match! All samples were successfully distributed.")
            else:
                missing = lines_in_source - successful_writes
                print(f"     ❌ Mismatch! Failed to distribute {missing} samples.")

    print("\n=== Final Summary ===")
    print(f"✅ Total entries successfully mapped: {total_global_success}")
    if total_global_errors > 0:
        print(f"⚠️ Total errors encountered: {total_global_errors}")
    else:
        print("🎉 Merge completed perfectly with no errors!")
    if is_separated:
        print(f"📂 SFT dataset: {sft_dir.resolve()}")
        print(f"📂 GRPO dataset: {grpo_dir.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge clips + SFT/GRPO into final datasets (separated SFT/GRPO, videos mirrored).")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--clips-dir", default=None, help="Source clip dataset directory")
    parser.add_argument("--training-dir", default=None, help="Training datasets directory")
    parser.add_argument("--out", default=None, help="Legacy single output dataset directory (uses same for SFT+GRPO)")
    parser.add_argument("--out-sft", default=None, help="Output SFT dataset directory (mirrored videos + SFT only)")
    parser.add_argument("--out-grpo", default=None, help="Output GRPO dataset directory (mirrored videos + GRPO only)")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    clip_dataset_dir = args.clips_dir or settings.clips_dir
    training_dataset_dir = args.training_dir or settings.formatter_output_dir
    # Prefer separated outputs if provided, else fallback to legacy --out or config defaults
    if args.out_sft or args.out_grpo:
        out_sft = args.out_sft or settings.dataset_sft_dir
        out_grpo = args.out_grpo or settings.dataset_grpo_dir
        merge_and_verify_datasets_separated(clip_dataset_dir, training_dataset_dir, out_sft, out_grpo)
    elif args.out:
        merge_and_verify_datasets(clip_dataset_dir, training_dataset_dir, args.out)
    else:
        # Default: separated mode using config dataset_sft/dataset_grpo
        merge_and_verify_datasets_separated(clip_dataset_dir, training_dataset_dir, settings.dataset_sft_dir, settings.dataset_grpo_dir)


if __name__ == "__main__":
    main()

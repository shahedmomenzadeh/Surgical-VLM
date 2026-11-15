"""Command-line entry point: `python -m video_pipeline [--step STEP]`.

Steps (same interface as the original pipeline):
  download, clean, transcribe, refine, summarize, vlm,
  adverse_event, downsample, split, format, merge,
  move_full_video, narration, refine_instrument_options,
  generate_missing_grpo_sft  — or  all
"""

import argparse
import os
import sys

from .config import build_settings, load_video_links
from .stages import (
    downloader, cleaner, transcriber, refiner, summarizer,
    vlm_generator, adverse_event_detector, splitter, dataset_formatter,
)
from .tools import (
    downsample_videos, merge_dataset, move_full_video,
    video_level_dataset, refine_instrument_options, generate_missing_grpo_sft,
)

ALL_STEPS = ["download", "clean", "transcribe", "refine", "summarize",
             "vlm", "adverse_event", "downsample", "split", "format",
             "merge", "move_full_video", "narration",
             "refine_instrument_options", "generate_missing_grpo_sft"]


def _print_banner(step):
    print("============================================")
    print(f"   VIDEO PIPELINE STARTED (Step: {step.upper()})")
    print("============================================")


def run_step(step, s):
    """Run a single pipeline stage given the dataset settings."""
    if step == "download":
        print("\n[Step 1/15] Running Downloader...")
        video_urls = load_video_links(s.links_file, config_path=s.config_path)
        if video_urls:
            downloader.run_downloader_pipeline(
                video_urls, s.videos_dir, s.audio_dir,
                s.metadata_file, s.cookies_file)
        else:
            print("❌ No URLs found. Skipping.")

    elif step == "clean":
        print("\n[Step 2/15] Running Cleaner...")
        cleaner.run_cleaner_pipeline(
            s.metadata_file, s.videos_dir, s.audio_dir,
            s.download.max_duration_seconds, auto_confirm=True)

    elif step == "transcribe":
        print("\n[Step 3/15] Running Whisper Transcriber...")
        transcriber.transcribe_audio_files(
            s.audio_dir, s.transcripts_dir,
            s.whisper.model_size, s.whisper.device)

    elif step == "refine":
        print("\n[Step 4/15] Running LLM Refiner...")
        refiner.run_refiner_pipeline(
            s.transcripts_dir, s.videos_dir, s.refined_dir, s.refinement_log,
            model_name=s.refiner.model,
            base_url=s.refiner.base_url,
            api_key_env=s.refiner.api_key_env,
            delay=s.refiner.extra.get("api_call_delay_seconds", 20),
            max_files=s.refiner.extra.get("max_files_per_run", 300))

    elif step == "summarize":
        print("\n[Step 5/15] Creating Dataset Summary...")
        summarizer.create_dataset_info(s.metadata_file, s.refinement_log, s.summary_file)

    elif step == "vlm":
        print("\n[Step 6/15] Generating VLM Fine-Tuning Dataset...")
        vlm_generator.run_vlm_generation_pipeline(
            dataset_summary_path=s.summary_file,
            refined_dir=s.refined_dir,
            output_dir=s.vlm_dir,
            aggregate_filename=s.vlm.aggregate_file,
            log_filename=s.vlm.log_file,
            gatekeeper_settings=s.vlm.gatekeeper,
            analyst_model=s.vlm.analyst_model,
            analyst_api_key_env=s.vlm.analyst_api_key_env)

    elif step == "adverse_event":
        print("\n[Step 7/15] Running Adverse Event Detector...")
        adverse_event_detector.run_adverse_event_pipeline(
            vlm_input_dir=s.vlm_dir,
            output_dir=s.adverse_events_dir,
            log_filename=s.adverse_event.extra.get("log_file", "adverse_event_log.csv"),
            aggregate_filename=s.adverse_event.extra.get("aggregate_file", "adverse_events_all.jsonl"),
            model_name=s.adverse_event.model,
            base_url=s.adverse_event.base_url,
            api_key_env=s.adverse_event.api_key_env)

    elif step == "downsample":
        print("\n[Step 8/15] Downsampling Videos...")
        downsample_videos.downsample_directory(
            s.downsample.input_dir, s.downsample.output_dir, s.downsample.target_fps)

    elif step == "split":
        print("\n[Step 9/15] Running Video Splitter...")
        splitter.run_splitter_pipeline(
            vlm_input_dir=s.vlm_dir,
            videos_dir=s.splitter_videos_dir,
            output_dir=s.clips_dir,
            log_filename=os.path.basename(s.splitter_log))

    elif step == "format":
        print("\n[Step 10/15] Generating Training Datasets (SFT & GRPO)...")
        dataset_formatter.run_dataset_formatter(
            clips_dir=s.clips_dir,
            output_dir=s.formatter_output_dir,
            model_name=s.dataset_formatter.model,
            base_url=s.dataset_formatter.base_url,
            api_key_env=s.dataset_formatter.api_key_env,
            log_filename=s.formatter_log,
            split_filename=s.formatter_split_file,
            max_retries=s.formatter_max_retries)

    elif step == "merge":
        print("\n[Step 11/15] Merging Clips and Training Data into Separated Datasets (SFT/GRPO, videos mirrored)...")
        # Separated mode: videos mirrored, SFT→dataset_sft, GRPO→dataset_grpo
        if hasattr(s, 'dataset_sft_dir') and hasattr(s, 'dataset_grpo_dir') and s.dataset_sft_dir != s.dataset_grpo_dir:
            merge_dataset.merge_and_verify_datasets_separated(
                s.clips_dir, s.formatter_output_dir, s.dataset_sft_dir, s.dataset_grpo_dir)
        else:
            merge_dataset.merge_and_verify_datasets(
                s.clips_dir, s.formatter_output_dir, s.dataset_dir)

    elif step == "move_full_video":
        print("\n[Step 12/15] Copying Full Videos into Datasets (mirrored to SFT+GRPO)...")
        # Mirrored: full_video.mp4 needed in both for training
        targets = []
        if hasattr(s, 'dataset_sft_dir') and s.dataset_sft_dir:
            targets.append(s.dataset_sft_dir)
        if hasattr(s, 'dataset_grpo_dir') and s.dataset_grpo_dir:
            # avoid duplicate if same as sft
            if not targets or s.dataset_grpo_dir != targets[0]:
                targets.append(s.dataset_grpo_dir)
        # Fallback legacy
        if not targets:
            targets = [s.dataset_dir]
        for t in targets:
            move_full_video.copy_main_videos_to_dataset(s.splitter_videos_dir, t)
            move_full_video.verify_dataset_videos(t)

    elif step == "narration":
        print("\n[Step 13/15] Generating Full-Video Narration (SFT dataset)...")
        # full_video_sft.jsonl is SFT-only
        if hasattr(s, 'dataset_sft_dir') and s.dataset_sft_dir:
            video_level_dataset.process_dataset(s.dataset_sft_dir)
        else:
            video_level_dataset.process_dataset(s.dataset_dir)

    elif step == "refine_instrument_options":
        print("\n[Step 14/15] Refining Instrument Distractor Options...")
        # Operates on SFT dataset for SFT files and propagates to GRPO dataset
        sft_target = getattr(s, 'dataset_sft_dir', s.dataset_dir)
        grpo_target = getattr(s, 'dataset_grpo_dir', None)
        # If separated, pass both; otherwise single
        if grpo_target and sft_target != grpo_target and os.path.isdir(grpo_target):
            refine_instrument_options.refine_instrument_options(
                sft_target,
                base_url=s.refine_instrument_options.base_url,
                model=s.refine_instrument_options.model,
                api_key_env=s.refine_instrument_options.api_key_env,
                max_retries=s.refine_instrument_options.extra.get("max_retries", 3),
                dry_run=False,
                grpo_dataset_dir=grpo_target)
        else:
            refine_instrument_options.refine_instrument_options(
                s.dataset_dir,
                base_url=s.refine_instrument_options.base_url,
                model=s.refine_instrument_options.model,
                api_key_env=s.refine_instrument_options.api_key_env,
                max_retries=s.refine_instrument_options.extra.get("max_retries", 3),
                dry_run=False)

    elif step == "generate_missing_grpo_sft":
        print("\n[Step 15/15] Generating Missing SFT/GRPO Files...")
        # In separated mode, generate missing SFT in sft dataset and missing GRPO in grpo dataset
        if hasattr(s, 'dataset_sft_dir') and hasattr(s, 'dataset_grpo_dir') and s.dataset_sft_dir != s.dataset_grpo_dir:
            # SFT missing in sft dataset
            generate_missing_grpo_sft.generate_missing(
                s.dataset_sft_dir,
                base_url=s.generate_missing_grpo_sft.base_url,
                model=s.generate_missing_grpo_sft.model,
                api_key_env=s.generate_missing_grpo_sft.api_key_env,
                max_retries=s.generate_missing_grpo_sft.extra.get("max_retries", 3),
                dry_run=False,
                target="sft")
            # GRPO missing in grpo dataset
            generate_missing_grpo_sft.generate_missing(
                s.dataset_grpo_dir,
                base_url=s.generate_missing_grpo_sft.base_url,
                model=s.generate_missing_grpo_sft.model,
                api_key_env=s.generate_missing_grpo_sft.api_key_env,
                max_retries=s.generate_missing_grpo_sft.extra.get("max_retries", 3),
                dry_run=False,
                target="grpo")
        else:
            generate_missing_grpo_sft.generate_missing(
                s.dataset_dir,
                base_url=s.generate_missing_grpo_sft.base_url,
                model=s.generate_missing_grpo_sft.model,
                api_key_env=s.generate_missing_grpo_sft.api_key_env,
                max_retries=s.generate_missing_grpo_sft.extra.get("max_retries", 3),
                dry_run=False)


def main(argv=None, config=None):
    """Run the pipeline. `config` overrides the config file path and may be
    combined with normal command-line arguments."""
    if config is not None:
        argv = (["--config", config] + list(sys.argv[1:])) if argv is None else ["--config", config] + list(argv)

    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all"] + ALL_STEPS,
        help="Specific pipeline step to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yaml file (default: bundled video_pipeline/config.yaml "
             "or $VIDEO_PIPELINE_CONFIG).",
    )
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    _print_banner(args.step)

    if args.step == "all":
        for step in ALL_STEPS:
            run_step(step, settings)
    else:
        run_step(args.step, settings)

    print("\n============================================")
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("============================================")


if __name__ == "__main__":
    main()

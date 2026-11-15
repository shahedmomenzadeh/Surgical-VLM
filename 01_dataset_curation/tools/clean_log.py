"""Remove ERROR_GENERATION rows from the VLM process log.

The VLM generator logs failed generations as ``ERROR_GENERATION``; this
utility purges them when a clean history is wanted.

Usage:
    python -m video_pipeline.tools.clean_log [--log FILE]
"""

import os
import argparse
import pandas as pd

from ..config import build_settings


def clean_process_log(log_path):
    if not os.path.exists(log_path):
        print(f"❌ File not found at: {log_path}")
        return

    print(f"Reading {log_path}...")

    try:
        df = pd.read_csv(log_path)
        original_count = len(df)

        df_clean = df[df['status'] != 'ERROR_GENERATION']
        new_count = len(df_clean)
        removed_count = original_count - new_count

        if removed_count == 0:
            print("✅ No 'ERROR_GENERATION' rows found. File is already clean.")
            return

        df_clean.to_csv(log_path, index=False)

        print("=" * 30)
        print("   CLEANUP COMPLETE")
        print("=" * 30)
        print(f"Original rows: {original_count}")
        print(f"Rows kept:     {new_count}")
        print(f"🗑️ Removed:    {removed_count} error rows")

    except Exception as e:
        print(f"❌ An error occurred: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clean ERROR_GENERATION rows from the VLM process log.")
    parser.add_argument("--config", default=None,
                        help="Path to a config.yaml (default: bundled / $VIDEO_PIPELINE_CONFIG)")
    parser.add_argument("--log", default=None, help="Path to the process log CSV")
    args = parser.parse_args(argv)

    settings = build_settings(config_path=args.config)
    log_path = args.log or os.path.join(settings.vlm_dir, settings.vlm.log_file)
    clean_process_log(log_path)


if __name__ == "__main__":
    main()

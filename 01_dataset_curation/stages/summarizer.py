"""Summarizer stage: merges the refinement log with download metadata into
the final dataset_info.csv used by downstream stages."""

import os
import pandas as pd


def create_dataset_info(metadata_path, log_path, output_file):
    """Merge refinement_log.csv with video_metadata.csv and write
    dataset_info.csv (same columns as before, plus `video_id`)."""
    print("=== CREATING DATASET INFO ===\n")

    if not os.path.exists(metadata_path):
        print(f"❌ Error: Metadata file not found at {metadata_path}")
        return
    if not os.path.exists(log_path):
        print(f"❌ Error: Log file not found at {log_path}")
        return

    try:
        df_meta = pd.read_csv(metadata_path)
        df_log = pd.read_csv(log_path)
        print("✅ Files loaded successfully.")
    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return

    # Join on video_name == filename (both are `{video_id}.mp4`)
    print("Merging data...")
    df_log['video_name'] = df_log['video_name'].astype(str)
    df_meta['filename'] = df_meta['filename'].astype(str)

    merged_df = pd.merge(
        df_log,
        df_meta,
        left_on='video_name',
        right_on='filename',
        how='inner'
    )

    if merged_df.empty:
        print("⚠️ Warning: The merged dataset is empty. Check if filenames match between the two CSVs.")
        return

    final_df = pd.DataFrame()
    if 'video_id' in merged_df.columns:
        final_df['video_id'] = merged_df['video_id']
    else:
        # Fallback for pre-migration metadata: derive the ID from the URL
        from ..common.utils import extract_youtube_id
        final_df['video_id'] = merged_df['url'].map(extract_youtube_id)
    final_df['title'] = merged_df['title']
    final_df['video_name'] = merged_df['video_name']
    final_df['transcript_name'] = merged_df['transcript_name']
    final_df['audio_name'] = merged_df['audio_filename']
    final_df['duration_seconds'] = merged_df['duration_seconds']
    final_df['word_count'] = merged_df['total_words']
    final_df['channel_name'] = merged_df['channel_name']
    final_df['url'] = merged_df['url']
    final_df['download_name'] = merged_df['filename']
    final_df['download_date'] = merged_df['download_date']

    try:
        final_df.to_csv(output_file, index=False)
        print("\n" + "=" * 30)
        print("       SUCCESS       ")
        print("=" * 30)
        print(f"Merged {len(final_df)} records.")
        print(f"Saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"❌ Error saving summary file: {e}")

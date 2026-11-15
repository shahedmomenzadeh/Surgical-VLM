"""Cleaner stage: removes videos exceeding the configured duration threshold."""

import os
import pandas as pd


def find_videos_to_remove(metadata_path, max_seconds):
    """Find videos over the duration threshold without deleting anything."""
    if not os.path.exists(metadata_path):
        print(f"❌ Error: Metadata file not found at '{metadata_path}'. Cannot proceed.")
        return None, None

    try:
        df = pd.read_csv(metadata_path)
    except pd.errors.EmptyDataError:
        print("✅ Metadata file is empty. Nothing to clean up.")
        return None, None
    except Exception as e:
        print(f"❌ Error reading metadata file: {e}")
        return None, None

    if 'duration_seconds' not in df.columns:
        print("❌ Error: 'duration_seconds' column not found in metadata.")
        return None, None

    df['duration_numeric'] = pd.to_numeric(df['duration_seconds'], errors='coerce')
    to_keep_mask = df['duration_numeric'].fillna(0) <= max_seconds

    return df[to_keep_mask], df[~to_keep_mask]


def delete_videos(df_to_remove, df_to_keep, metadata_path, videos_dir, audio_dir):
    """Delete the physical files and update the metadata CSV."""
    print("\nProceeding with deletion...")
    removed_count = 0

    for index, row in df_to_remove.iterrows():
        video_name = row.get('filename')
        audio_name = row.get('audio_filename')
        title = row.get('title', f"URL: {row.get('url', 'N/A')}")

        print(f"\nProcessing '{title}' (Duration: {row.get('duration_seconds')}s)")

        if pd.notna(video_name) and video_name not in ["FAILED", "SKIPPED_DURATION"]:
            video_path = os.path.join(videos_dir, video_name)
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"  🗑️ Removed video: {video_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ❌ Error removing video {video_path}: {e}")
            else:
                print(f"  🤷 Video file not found: {video_path}")

        if pd.notna(audio_name) and audio_name not in ["FAILED", "SKIPPED_DURATION"]:
            audio_path = os.path.join(audio_dir, audio_name)
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"  🗑️ Removed audio: {audio_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ❌ Error removing audio {audio_path}: {e}")
            else:
                print(f"  🤷 Audio file not found: {audio_path}")

    try:
        if 'duration_numeric' in df_to_keep.columns:
            df_to_keep = df_to_keep.drop(columns=['duration_numeric'])
        df_to_keep.to_csv(metadata_path, index=False)
        print(f"\n✅ Successfully updated metadata file: {metadata_path}")
        print(f"Removed {len(df_to_remove)} entries from CSV.")
    except Exception as e:
        print(f"❌ CRITICAL: Error writing updated metadata file: {e}")
        print("   Your files may be deleted, but the CSV was not updated.")

    print(f"\n--- Cleanup Summary ---")
    print(f"Removed {len(df_to_remove)} videos from metadata.")
    print(f"Deleted {removed_count} associated files.")
    print("--- Cleanup Finished ---")


def run_cleaner_pipeline(metadata_file, videos_dir, audio_dir, max_duration_seconds,
                         auto_confirm=False):
    """Main entry point for the cleaner stage."""
    print(f"--- Video Cleanup Utility ---")
    print(f"This script will find files over {max_duration_seconds} seconds.")

    df_to_keep, df_to_remove = find_videos_to_remove(metadata_file, max_duration_seconds)

    if df_to_remove is None or df_to_remove.empty:
        if df_to_remove is not None:
            print(f"✅ No videos found exceeding the {max_duration_seconds}s threshold.")
            if df_to_keep is not None and 'duration_numeric' in df_to_keep.columns:
                df_to_keep = df_to_keep.drop(columns=['duration_numeric'])
                df_to_keep.to_csv(metadata_file, index=False)
        return

    print(f"\nFound {len(df_to_remove)} video(s) to remove (duration > {max_duration_seconds}s):")
    for index, row in df_to_remove.iterrows():
        title = row.get('title', f"URL: {row.get('url', 'N/A')}")
        duration = row.get('duration_seconds', 'N/A')
        print(f"  - {title} (Duration: {duration}s)")

    if auto_confirm:
        print("\nAuto-confirmation enabled. Proceeding...")
        proceed = True
    else:
        confirm = input("\nAre you sure you want to proceed with deleting these files and entries? (yes/no): ")
        proceed = confirm.lower() == 'yes'

    if proceed:
        delete_videos(df_to_remove, df_to_keep, metadata_file, videos_dir, audio_dir)
    else:
        print("Operation cancelled by user.")

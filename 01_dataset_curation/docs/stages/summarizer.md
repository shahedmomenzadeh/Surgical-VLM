# `video_pipeline/stages/summarizer.py` — Summarizer (Step 5)

## Purpose

Merges the refinement log with the download metadata into a single master
index (`dataset_info.csv`) consumed by the VLM generation stage.

## Flow

1. Loads `videos/video_metadata.csv` and
   `refined_transcripts/refinement_log.csv`.
2. Inner-joins on `video_name` (log) == `filename` (metadata) — both are
   `{video_id}.mp4` under YouTube-ID naming.
3. Writes `dataset_info.csv` with the canonical columns, including
   `video_id`.

## Inputs / Outputs

| | Path | Notes |
|---|---|---|
| In | `videos/video_metadata.csv` | download metadata |
| In | `refined_transcripts/refinement_log.csv` | refinement results |
| Out | `dataset_info.csv` | `video_id,title,video_name,transcript_name,audio_name,duration_seconds,word_count,channel_name,url,download_name,download_date` |

## Notes / Quirks

- Inner join: videos that were not refined (or not downloaded) are excluded.
- If `video_id` is missing from the metadata (pre-migration CSVs), it is
  derived from the URL via `extract_youtube_id()`.
- An empty merge prints a warning and exits without writing.

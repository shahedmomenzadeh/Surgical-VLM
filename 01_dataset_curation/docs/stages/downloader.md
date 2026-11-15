# `video_pipeline/stages/downloader.py` — Downloader (Step 1)

## Purpose

Ingests YouTube videos (single videos or playlists) and extracts audio.
Files are saved with **YouTube-ID naming**: `{youtube_id}.mp4` and
`{youtube_id}.wav` — never the video title (titles contain characters like
`™`, `｜`, `：` that break filenames). A metadata CSV tracks processed URLs
so reruns skip already-downloaded videos.

## Flow

1. `run_downloader_pipeline()` checks FFmpeg, expands playlists into
   individual URLs, then downloads each video.
2. `download_video_and_extract_audio()`:
   - skips URLs already present in the metadata CSV;
   - fetches video info (`id`, `title`, `uploader`, `duration`, resolution);
   - downloads via yt-dlp as `{id}.mp4` (480p mp4, merged);
   - extracts 16 kHz mono WAV audio with FFmpeg;
   - appends a row to the metadata CSV.
3. `verify_and_process_existing_videos()` extracts audio for any existing
   video that has no `.wav` yet (e.g. after an interrupted run).

## Inputs / Outputs

| | Path | Notes |
|---|---|---|
| In | `youtube_links.txt` | one URL per line (`#` comments allowed) |
| Out | `videos/{youtube_id}.mp4` | 480p, mp4 |
| Out | `audio/{youtube_id}.wav` | 16 kHz mono PCM |
| Out | `videos/video_metadata.csv` | `video_id,title,channel_name,url,filename,download_date,duration_seconds,resolution,audio_filename` |

## Configuration

- `directories.videos`, `directories.audio` — output dirs
- `files.metadata_csv`, `files.cookies`, `files.video_links`
- `files.cookies` — optional cookies file for age-restricted videos

## Dependencies

`yt-dlp`, `pandas`, FFmpeg (system binary).

## Notes / Quirks

- A URL is considered processed once it appears in the metadata CSV, even if
  the audio extraction later failed.
- `verify_and_process_existing_videos()` matches audio to videos purely by
  stem name (works with any naming scheme, including pre-migration titles).
- Playlists are flattened with `extract_flat: in_playlist` so only individual
  video URLs are downloaded.

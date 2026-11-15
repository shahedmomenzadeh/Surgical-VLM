"""Shared low-level utilities: timestamps, YouTube IDs, name normalization, FFmpeg check."""

import re
import shutil
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def time_str_to_seconds(time_str):
    """Convert 'MM:SS' or 'HH:MM:SS' to total seconds. Returns None if unparsable."""
    if not time_str or "?" in time_str:
        return None
    parts = str(time_str).strip().split(':')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    return None


def seconds_to_mm_ss(seconds):
    """Convert seconds to 'MM:SS' (or 'HH:MM:SS' when >= 1 hour)."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# YouTube IDs
# ---------------------------------------------------------------------------

def extract_youtube_id(url):
    """Extract the YouTube video ID from a standard or shortened URL."""
    if not url:
        return ""
    if "youtu.be" in url:
        return url.rstrip('/').split('/')[-1].split('?')[0]
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    return ""


# ---------------------------------------------------------------------------
# Name normalization (for robust filename matching)
# ---------------------------------------------------------------------------

def normalize_id(name):
    """
    Strip all non-alphanumeric characters (spaces, punctuation, emojis,
    full-width chars) and lowercase. Guarantees matches even when the OS
    sanitized folder/file names (e.g. '™' saved as '__').
    """
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()


# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and available in the system's PATH."""
    return shutil.which("ffmpeg") is not None

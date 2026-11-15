"""Transcriber stage: Whisper speech-to-text on audio files -> JSON segments."""

import os
import json
from tqdm import tqdm


def _load_whisper(model_size, device):
    """Load Whisper (and torch) lazily so the CLI works without them."""
    try:
        import whisper
    except ImportError as e:
        print("❌ Error: 'whisper' package not installed. "
              "Run: pip install git+https://github.com/openai/whisper.git")
        raise SystemExit(1) from e
    return whisper.load_model(model_size, device=device)


def transcribe_audio_files(input_dir, output_dir, model_size='large', device=None):
    """Transcribe all .wav files in `input_dir` (named `{video_id}.wav`) and
    write `{video_id}.json` segment files into `output_dir`."""
    import torch

    print("--- Starting Audio Transcription Process ---")

    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device.upper()}")
    if device == 'cpu':
        print("⚠️ WARNING: No GPU found. Transcription will be very slow.")

    print(f"Loading Whisper model ({model_size})...")
    try:
        model = _load_whisper(model_size, device)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return

    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return

    audio_files = {os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.wav')}
    transcribed_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.json')}
    files_to_process = sorted([f + '.wav' for f in (audio_files - transcribed_files)])

    if not files_to_process:
        print("✅ All audio files have already been transcribed.")
        return

    print(f"Found {len(files_to_process)} audio file(s) to transcribe.")

    for filename in tqdm(files_to_process, desc="Transcribing Audio"):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.json"
        output_path = os.path.join(output_dir, output_filename)

        try:
            result = model.transcribe(input_path, fp16=torch.cuda.is_available())

            segments = [
                {
                    "start": round(seg["start"], 1),
                    "end": round(seg["end"], 1),
                    "text": seg["text"].strip()
                }
                for seg in result["segments"]
            ]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=4, ensure_ascii=False)

        except Exception as e:
            tqdm.write(f"❌ Error transcribing {filename}: {e}")

    print("\n--- Audio Transcription process completed. ---")

"""Gemini API connectivity test.

Usage:
    python -m video_pipeline.tools.apitest
"""

import sys

from ..common.gemini import get_gemini_client


def generate():
    client = get_gemini_client("GEMINI_API_KEY")
    if not client:
        sys.exit(1)

    model = "gemini-2.5-flash"
    contents = [(
        "If you're functioning say exactly this: Yes I am functioning."
    )]
    config = None
    try:
        from google.genai import types
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
        )
    except ImportError:
        pass

    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=config,
    ):
        print(chunk.text, end="")
    print()


if __name__ == "__main__":
    generate()

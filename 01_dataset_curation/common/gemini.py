"""Google GenAI client (google-genai SDK).

Only the VLM video analysis stage (analyst) uses this module, because it
sends YouTube URLs as `file_data` parts, a Gemini-specific feature that is
not available through OpenAI-compatible endpoints.
"""

import os
from google import genai

from ..config import load_env


def get_gemini_client(env_var="GEMINI_API_KEY"):
    """Initialize the Google GenAI client.

    Key resolution order: Google Colab secrets -> environment / project .env.
    """
    api_key = None
    try:
        from google.colab import userdata
        api_key = userdata.get(env_var)
    except ImportError:
        pass
    except Exception:
        pass

    if not api_key:
        load_env()
        api_key = os.getenv(env_var)

    if not api_key:
        print(f"❌ Error: {env_var} not found. Please set it in .env or Colab Secrets.")
        return None
    return genai.Client(api_key=api_key)

"""OpenAI-compatible LLM clients (base_url + model + api_key).

All text-only LLM stages (refiner, vlm gatekeeper, adverse-event detector,
dataset formatter) use this module. API keys are resolved in this order:
Google Colab secrets -> environment variable -> .env file.
"""

import os
import json
import time
from openai import OpenAI

from ..config import load_env


def _resolve_api_key(env_var, colab_secret=None):
    """Resolve an API key: Colab secret first, then env var / project .env."""
    secret = colab_secret or env_var
    if secret:
        try:
            from google.colab import userdata
            key = userdata.get(secret)
            if key:
                return key
        except ImportError:
            pass
        except Exception:
            pass
    load_env()
    return os.getenv(env_var)


def get_openai_client(api_key, base_url=None):
    """Create an OpenAI-compatible client. `api_key` may be the raw key or
    the name of the environment variable holding it."""
    if not api_key:
        raise ValueError("No API key provided (set the corresponding key in .env).")
    return OpenAI(base_url=base_url, api_key=api_key)


def resolve_and_get_client(env_var, base_url=None, colab_secret=None):
    """Resolve an API key from the environment and build a client."""
    key = _resolve_api_key(env_var, colab_secret)
    if not key:
        print(f"❌ {env_var} not found! Please set it in Colab Secrets or your .env file.")
        return None
    return get_openai_client(key, base_url)


def chat_json_with_retry(client, model_name, messages, max_retries=3, temperature=0.7,
                         max_tokens=None, retry_delay=2.0, response_format=None,
                         on_retry=None):
    """
    Call an OpenAI-compatible chat completion requesting JSON, retrying on
    network/validation errors. Returns the parsed JSON dict, or None on failure.
    """
    kwargs = {"model": model_name, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if response_format:
        kwargs["response_format"] = response_format

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if response_format is not None:
                return json.loads(content)
            return content
        except Exception as e:
            if on_retry:
                on_retry(attempt, e)
            if attempt < max_retries:
                time.sleep(retry_delay)
    return None

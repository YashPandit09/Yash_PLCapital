"""
llm.py – Thin Gemini wrapper for Portfolio Analytics Agent.

Uses the google-genai SDK (``from google import genai``).
Loads GEMINI_API_KEY from environment / .env file.
Provides retry with exponential backoff for transient failures (429, 503).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

_client: Optional[Any] = None
_CANDIDATE_MODELS = [
    os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]
_MODEL_NAME = _CANDIDATE_MODELS[0]
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.5  # seconds


def _get_client() -> Any:
    """Lazy-initialise and return the genai Client."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.\n"
            "Get a key at https://ai.google.dev/gemini-api/docs"
        )

    from google import genai  # type: ignore[import-untyped]

    _client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialised (model=%s)", _MODEL_NAME)
    return _client


def generate(
    prompt: str,
    *,
    system_instruction: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> str:
    """Call Gemini and return the text response.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff on 429 / 5xx.

    Args:
        prompt: The user prompt.
        system_instruction: Optional system prompt.
        temperature: Sampling temperature (0 = deterministic).
        max_output_tokens: Max tokens in the response.

    Returns:
        The generated text, stripped of leading/trailing whitespace.

    Raises:
        RuntimeError: If API key is missing or all retries exhausted.
    """
    global _MODEL_NAME
    client = _get_client()
    from google.genai import types  # type: ignore[import-untyped]

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if system_instruction:
        config.system_instruction = system_instruction

    last_error: Optional[Exception] = None
    for model in _CANDIDATE_MODELS:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                _MODEL_NAME = model
                text = response.text or ""
                return text.strip()

            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                err_lower = err_str.lower()

                # Check for depleted credits (HTTP 402) - non-retryable
                if "402" in err_lower or "credits are depleted" in err_lower or "prepayment" in err_lower:
                    raise RuntimeError(
                        "Your Gemini API prepayment credits are depleted. "
                        "Please go to Google AI Studio at https://ai.studio/projects to manage billing."
                    )

                # Check for rate-limiting / quota exhaustion
                elif (
                    "429" in err_lower
                    or "resource_exhausted" in err_lower
                    or "quota" in err_lower
                    or "limit exceeded" in err_lower
                ):
                    import re
                    # Look for Google's recommended retry delay (e.g. retry in 23s or retryDelay: '23s')
                    match = re.search(r"retry(?:\s+in\s+|Delay['\":\s]+)(\d+)", err_str, re.IGNORECASE)
                    if match:
                        wait = float(match.group(1)) + 2.0
                    else:
                        wait = max(15.0, _INITIAL_BACKOFF * (2 ** (attempt - 1)))

                    logger.warning(
                        "Gemini rate limit on %s (attempt %d/%d): waiting %.1fs before retry...",
                        model,
                        attempt,
                        _MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)

                # Check if model not found or service unavailable -> try next candidate model
                elif "404" in err_lower or "not_found" in err_lower or "503" in err_lower:
                    logger.warning("Model %s unavailable (%s), trying next candidate", model, exc)
                    break

                else:
                    raise RuntimeError(f"Gemini API error: {exc}") from exc

    raise RuntimeError(
        f"Gemini API failed after {_MAX_RETRIES} retries: {last_error}"
    )


def is_available() -> bool:
    """Check whether a valid API key is configured (without making an API call)."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return bool(api_key) and api_key != "your_key_here"


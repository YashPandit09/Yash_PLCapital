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
_MAX_RETRIES = 2
_INITIAL_BACKOFF = 2.0  # seconds


def _get_candidate_models() -> list[str]:
    """Return candidate Gemini models in priority order, respecting GEMINI_MODEL env var."""
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    primary = env_model if env_model else "gemini-3.5-flash"
    candidates = [primary, "gemini-3.5-flash-lite", "gemini-3.8-flash"]
    seen: set[str] = set()
    return [m for m in candidates if m and not (m in seen or seen.add(m))]


def _get_client() -> Any:
    """Lazy-initialise and return the genai Client."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.\n"
            "Get a key at https://ai.google.dev/gemini-api/docs"
        )

    from google import genai  # type: ignore[import-untyped]

    _client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialised")
    return _client


def generate(
    prompt: str,
    *,
    system_instruction: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> str:
    """Call Gemini and return the text response.

    Retries transient failures with quick backoff; fails fast on quota exhaustion.

    Args:
        prompt: The user prompt.
        system_instruction: Optional system prompt.
        temperature: Sampling temperature (0 = deterministic).
        max_output_tokens: Max tokens in the response.

    Returns:
        The generated text, stripped of leading/trailing whitespace.

    Raises:
        RuntimeError: If API key is missing or quota is exhausted.
    """
    client = _get_client()
    from google.genai import types  # type: ignore[import-untyped]

    candidate_models = _get_candidate_models()
    last_error: Optional[Exception] = None

    for model in candidate_models:
        # Lite models do not support thinking_budget
        use_thinking = "lite" not in model.lower()

        for attempt in range(1, _MAX_RETRIES + 1):
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if use_thinking:
                config.thinking_config = types.ThinkingConfig(thinking_budget=0)
            if system_instruction:
                config.system_instruction = system_instruction

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                text = response.text or ""
                return text.strip()

            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                err_lower = err_str.lower()

                # 1. Depleted credits (HTTP 402) - fail immediately without hang
                if "402" in err_lower or "credits are depleted" in err_lower or "prepayment" in err_lower:
                    raise RuntimeError(
                        "Your Gemini API prepayment credits are depleted. "
                        "Please go to Google AI Studio at https://ai.studio/projects to manage billing."
                    )

                # 2. Daily request quota limit exceeded - fail immediately without hang
                if (
                    "per day" in err_lower
                    or "daily" in err_lower
                    or "limit: 20" in err_lower
                    or "requests per day" in err_lower
                ):
                    raise RuntimeError(
                        "Gemini API free-tier daily request quota exceeded for this model/project. "
                        "Please retry later or provide an API key with active quota."
                    )

                # 3. Model parameter error (400 INVALID_ARGUMENT for thinking_config): retry once without it
                if "400" in err_lower and ("thinking" in err_lower or "invalid_argument" in err_lower):
                    if use_thinking:
                        logger.warning("Thinking config unsupported on %s, retrying without thinking budget", model)
                        use_thinking = False
                        continue
                    # Non-thinking 400 error
                    raise RuntimeError(f"Gemini API parameter error: {exc}") from exc

                # 4. Transient rate-limiting (429 / RESOURCE_EXHAUSTED)
                if (
                    "429" in err_lower
                    or "resource_exhausted" in err_lower
                    or "quota" in err_lower
                    or "limit exceeded" in err_lower
                ):
                    import re
                    # Check for large wait times (e.g. retry in 19h... or retryDelay: '60000s')
                    match = re.search(r"retry(?:\s+in\s+|Delay['\":\s]+)(\d+)", err_str, re.IGNORECASE)
                    wait = float(match.group(1)) if match else _INITIAL_BACKOFF

                    # If wait is large (> 20s) or attempt limit reached, fail fast without hanging 60+ seconds
                    if wait > 20.0 or attempt >= _MAX_RETRIES:
                        raise RuntimeError(
                            f"Gemini API rate limit exceeded ({model}): {err_str[:120]}. "
                            "Please try again in a few moments."
                        )

                    logger.warning(
                        "Gemini rate limit on %s (attempt %d/%d): waiting %.1fs...",
                        model,
                        attempt,
                        _MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                # 5. Model not found or service temporarily unavailable -> try next model
                if "404" in err_lower or "not_found" in err_lower or "503" in err_lower:
                    logger.warning("Model %s unavailable (%s), trying next candidate", model, exc)
                    break

                raise RuntimeError(f"Gemini API error: {exc}") from exc

    raise RuntimeError(
        f"Gemini API request failed across all models: {last_error}"
    )


def is_available() -> bool:
    """Check whether a valid API key is configured (without making an API call)."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return bool(api_key) and api_key != "your_key_here"


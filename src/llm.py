"""
llm.py – Multi-provider LLM wrapper for Portfolio Analytics Agent.

Supports:
1. Google Gemini (default) via google-genai SDK
2. Groq / OpenAI-compatible API via openai SDK

Controlled by environment variables:
- LLM_PROVIDER: "gemini" (default), "groq", or "openai_compatible"
- GEMINI_API_KEY / GEMINI_MODEL: For Gemini provider
- GROQ_API_KEY / LLM_MODEL / LLM_BASE_URL: For Groq / OpenAI-compatible provider
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

_gemini_client: Optional[Any] = None
_openai_client: Optional[Any] = None
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds


def get_provider() -> str:
    """Return active provider: 'gemini', 'groq', or 'openai_compatible'."""
    p = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if p in ("groq", "openai_compatible"):
        return p
    return "gemini"


# ──────────────────────────────────────────────
# Gemini Client & Candidates
# ──────────────────────────────────────────────

def _get_candidate_gemini_models() -> list[str]:
    """Return candidate Gemini models in priority order, respecting GEMINI_MODEL env var."""
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    primary = env_model if env_model else "gemini-3.5-flash"
    candidates = [primary, "gemini-3.5-flash-lite", "gemini-3.8-flash"]
    seen: set[str] = set()
    return [m for m in candidates if m and not (m in seen or seen.add(m))]


# Backwards compatibility alias for tests
_get_candidate_models = _get_candidate_gemini_models


def _get_gemini_client() -> Any:
    """Lazy-initialise and return the Google GenAI Client."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.\n"
            "Get a key at https://ai.google.dev/gemini-api/docs"
        )

    from google import genai  # type: ignore[import-untyped]

    _gemini_client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialised")
    return _gemini_client


# ──────────────────────────────────────────────
# OpenAI / Groq Client & Candidates
# ──────────────────────────────────────────────

def _get_candidate_groq_models() -> list[str]:
    """Return candidate Groq / OpenAI-compatible models in priority order."""
    env_model = os.environ.get("LLM_MODEL", "").strip()
    primary = env_model if env_model else "llama-3.3-70b-versatile"
    candidates = [primary, "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    seen: set[str] = set()
    return [m for m in candidates if m and not (m in seen or seen.add(m))]


def _get_openai_client() -> Any:
    """Lazy-initialise and return the OpenAI-compatible Client (Groq/OpenAI)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    provider = get_provider()
    api_key = (
        os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not api_key or api_key == "your_key_here":
        key_var = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        raise RuntimeError(
            f"{key_var} is not set. Copy .env.example to .env and add your key.\n"
            "Get a Groq key at https://console.groq.com/keys"
        )

    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    if not base_url:
        if provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
        else:
            base_url = "https://api.openai.com/v1"

    from openai import OpenAI

    _openai_client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("OpenAI-compatible client initialised (provider=%s, base_url=%s)", provider, base_url)
    return _openai_client


# ──────────────────────────────────────────────
# Availability Check
# ──────────────────────────────────────────────

def is_available() -> bool:
    """Check whether a valid API key is configured for the active provider."""
    provider = get_provider()
    if provider in ("groq", "openai_compatible"):
        api_key = os.environ.get("GROQ_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        return bool(api_key) and api_key != "your_key_here"

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    return bool(api_key) and api_key != "your_key_here"


# ──────────────────────────────────────────────
# Unified Generate Interface
# ──────────────────────────────────────────────

def generate(
    prompt: str,
    *,
    system_instruction: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    response_format: Optional[dict[str, str]] = None,
) -> str:
    """Call active LLM provider and return stripped text response.

    Args:
        prompt: User prompt.
        system_instruction: Optional system instruction.
        temperature: Sampling temperature (default 0.0 for deterministic).
        max_output_tokens: Max tokens in output.
        response_format: Optional dict e.g. {"type": "json_object"}.

    Returns:
        Generated text stripped of surrounding whitespace.
    """
    provider = get_provider()
    if provider in ("groq", "openai_compatible"):
        return _generate_openai(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
        )

    return _generate_gemini(
        prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


# ──────────────────────────────────────────────
# Gemini Generator
# ──────────────────────────────────────────────

def _generate_gemini(
    prompt: str,
    *,
    system_instruction: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> str:
    client = _get_gemini_client()
    from google.genai import types  # type: ignore[import-untyped]

    candidate_models = _get_candidate_gemini_models()
    last_error: Optional[Exception] = None

    for model in candidate_models:
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

                # Auth / prepayment
                if "402" in err_lower or "credits are depleted" in err_lower or "prepayment" in err_lower:
                    raise RuntimeError(
                        "Gemini API prepayment credits are depleted. "
                        "Please go to Google AI Studio at https://ai.studio/projects to manage billing."
                    )

                if "401" in err_lower or "unauthenticated" in err_lower or "api_key_invalid" in err_lower:
                    raise RuntimeError("Gemini API authentication failed (HTTP 401). Please verify your GEMINI_API_KEY.")

                # Daily request quota
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

                # Parameter error (thinking budget)
                if "400" in err_lower and ("thinking" in err_lower or "invalid_argument" in err_lower):
                    if use_thinking:
                        logger.warning("Thinking config unsupported on %s, retrying without thinking budget", model)
                        use_thinking = False
                        continue
                    raise RuntimeError(f"Gemini API parameter error: {exc}") from exc

                # Transient rate-limit (429)
                if (
                    "429" in err_lower
                    or "resource_exhausted" in err_lower
                    or "quota" in err_lower
                    or "limit exceeded" in err_lower
                ):
                    match = re.search(r"retry(?:\s+in\s+|Delay['\":\s]+)(\d+)", err_str, re.IGNORECASE)
                    wait = float(match.group(1)) if match else _INITIAL_BACKOFF
                    if wait > 20.0 or attempt >= _MAX_RETRIES:
                        raise RuntimeError(
                            f"Gemini API rate limit exceeded ({model}): {err_str[:120]}. "
                            "Please try again in a few moments."
                        )
                    logger.warning("Gemini rate limit on %s (attempt %d/%d): waiting %.1fs...", model, attempt, _MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue

                # 404 / 503 candidate fallback
                if "404" in err_lower or "not_found" in err_lower or "503" in err_lower:
                    logger.warning("Model %s unavailable (%s), trying next candidate", model, exc)
                    break

                raise RuntimeError(f"Gemini API error: {exc}") from exc

    raise RuntimeError(f"Gemini API request failed across all models: {last_error}")


# ──────────────────────────────────────────────
# OpenAI / Groq Generator
# ──────────────────────────────────────────────

def _generate_openai(
    prompt: str,
    *,
    system_instruction: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    response_format: Optional[dict[str, str]] = None,
) -> str:
    client = _get_openai_client()
    candidate_models = _get_candidate_groq_models()
    last_error: Optional[Exception] = None

    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    # Auto-enable JSON object response format for router or when system instruction asks for JSON
    fmt = response_format
    if fmt is None and system_instruction and "json" in system_instruction.lower():
        fmt = {"type": "json_object"}

    for model in candidate_models:
        for attempt in range(1, _MAX_RETRIES + 1):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
            }
            if fmt is not None:
                kwargs["response_format"] = fmt

            try:
                completion = client.chat.completions.create(**kwargs)
                choice = completion.choices[0]
                text = choice.message.content or ""
                return text.strip()

            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                err_lower = err_str.lower()

                # 1. Auth errors (401) - Fail fast with clear message, NEVER log keys
                if "401" in err_lower or "authentication" in err_lower or "invalid_api_key" in err_lower:
                    raise RuntimeError(
                        "Groq/OpenAI authentication failed (HTTP 401). "
                        "Please verify that your GROQ_API_KEY is valid and not expired."
                    )

                # 2. Rate limit (429) - Parse Retry-After or 'try again in Ns'
                if (
                    "429" in err_lower
                    or "rate_limit" in err_lower
                    or "rate limit" in err_lower
                    or "tokens per minute" in err_lower
                    or "try again in" in err_lower
                    or "retry-after" in err_lower
                ):
                    wait = _INITIAL_BACKOFF
                    # Check response headers if available
                    response_obj = getattr(exc, "response", None)
                    if response_obj and hasattr(response_obj, "headers"):
                        header_wait = response_obj.headers.get("retry-after")
                        if header_wait:
                            try:
                                wait = float(header_wait)
                            except ValueError:
                                pass

                    # Check error message regex e.g. "Please try again in 1.5s"
                    match = re.search(r"try\s+again\s+in\s+([\d\.]+)\s*s", err_str, re.IGNORECASE)
                    if match:
                        try:
                            wait = float(match.group(1))
                        except ValueError:
                            pass

                    if wait > 20.0 or attempt >= _MAX_RETRIES:
                        raise RuntimeError(
                            f"Groq API rate limit exceeded ({model}): {err_str[:120]}. "
                            "Please try again in a few moments."
                        )

                    logger.warning(
                        "Groq rate limit on %s (attempt %d/%d): waiting %.1fs...",
                        model,
                        attempt,
                        _MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                # 3. Model not found (404) -> Fallback to next candidate model
                if "404" in err_lower or "model_not_found" in err_lower or "does not exist" in err_lower:
                    logger.warning("Model %s not found on provider, falling back to next candidate", model)
                    break

                # 4. Service unavailable (503) -> Retry or fallback
                if "503" in err_lower or "unavailable" in err_lower:
                    logger.warning("Model %s service unavailable (503), waiting %.1fs...", model, _INITIAL_BACKOFF)
                    time.sleep(_INITIAL_BACKOFF)
                    continue

                raise RuntimeError(f"Groq/OpenAI API error: {exc}") from exc

    raise RuntimeError(f"Groq/OpenAI request failed across all candidate models: {last_error}")

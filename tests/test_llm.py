"""
test_llm.py – Unit tests for multi-provider LLM client configuration, fallbacks, and error handling.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import src.llm as llm
from src.llm import (
    _get_candidate_gemini_models,
    _get_candidate_groq_models,
    _get_candidate_models,
    generate,
    get_provider,
    is_available,
)


@pytest.fixture(autouse=True)
def reset_llm_clients():
    """Reset cached clients before each test."""
    llm._gemini_client = None
    llm._openai_client = None
    yield
    llm._gemini_client = None
    llm._openai_client = None


# ──────────────────────────────────────────────
# Provider Selection & Availability
# ──────────────────────────────────────────────

def test_get_provider_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_provider() == "gemini"


def test_get_provider_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert get_provider() == "groq"

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    assert get_provider() == "openai_compatible"


def test_is_available_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    assert is_available() is True

    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert is_available() is False

    monkeypatch.setenv("GEMINI_API_KEY", "your_key_here")
    assert is_available() is False


def test_is_available_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    assert is_available() is True

    monkeypatch.setenv("GROQ_API_KEY", "")
    assert is_available() is False

    monkeypatch.setenv("GROQ_API_KEY", "your_key_here")
    assert is_available() is False


# ──────────────────────────────────────────────
# Candidate Models
# ──────────────────────────────────────────────

def test_gemini_candidate_models_default(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    models = _get_candidate_gemini_models()
    assert models[0] == "gemini-3.5-flash"
    assert "gemini-3.5-flash-lite" in models


def test_gemini_candidate_models_custom(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    models = _get_candidate_gemini_models()
    assert models[0] == "gemini-2.5-pro"


def test_groq_candidate_models_default(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    models = _get_candidate_groq_models()
    assert models[0] == "llama-3.3-70b-versatile"
    assert "openai/gpt-oss-120b" in models


def test_groq_candidate_models_custom(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "mixtral-8x7b-32768")
    models = _get_candidate_groq_models()
    assert models[0] == "mixtral-8x7b-32768"


# ──────────────────────────────────────────────
# Mocked Generation – Gemini
# ──────────────────────────────────────────────

def test_generate_gemini_mocked(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "SELECT COUNT(*) FROM portfolios"
    mock_client.models.generate_content.return_value = mock_resp

    with patch("src.llm._get_gemini_client", return_value=mock_client):
        res = generate("How many portfolios?", system_instruction="Output SQL")
        assert res == "SELECT COUNT(*) FROM portfolios"
        mock_client.models.generate_content.assert_called_once()


def test_gemini_missing_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        llm._get_gemini_client()


# ──────────────────────────────────────────────
# Mocked Generation – Groq / OpenAI-compatible
# ──────────────────────────────────────────────

def test_generate_groq_mocked(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "mock_groq_key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"tool": "sql_query", "portfolio_name": null}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("src.llm._get_openai_client", return_value=mock_client):
        res = generate("List portfolios", system_instruction="You are a router. Output JSON.")
        assert res == '{"tool": "sql_query", "portfolio_name": null}'

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_groq_missing_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        llm._get_openai_client()


def test_groq_auth_error_fails_fast(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "bad_key")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized: Invalid API Key")

    with patch("src.llm._get_openai_client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Groq/OpenAI authentication failed"):
            generate("Test prompt")


def test_groq_rate_limit_retry(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "SELECT 1"
    mock_success = MagicMock()
    mock_success.choices = [mock_choice]

    # First call rate limits with retry notice, second call succeeds
    mock_client.chat.completions.create.side_effect = [
        Exception("Rate limit reached. Please try again in 0.1s"),
        mock_success,
    ]

    with patch("src.llm._get_openai_client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        res = generate("Query")
        assert res == "SELECT 1"
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(0.1)

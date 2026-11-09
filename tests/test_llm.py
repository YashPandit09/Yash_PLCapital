"""
test_llm.py – Unit tests for LLM client configuration, fallbacks, and error handling.
"""

import os
import pytest

from src.llm import _get_candidate_models, is_available


def test_is_available_true(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_key_123")
    assert is_available() is True


def test_is_available_false(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert is_available() is False

    monkeypatch.setenv("GEMINI_API_KEY", "your_key_here")
    assert is_available() is False


def test_candidate_models_default(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    models = _get_candidate_models()
    assert models[0] == "gemini-3.5-flash"
    assert "gemini-3.5-flash-lite" in models


def test_candidate_models_custom_env(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    models = _get_candidate_models()
    assert models[0] == "gemini-2.5-pro"


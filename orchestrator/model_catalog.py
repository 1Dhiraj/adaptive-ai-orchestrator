"""Discover models that the current user can actually access."""

from __future__ import annotations

from typing import Any, Dict, List

import requests

from . import config


PROVIDERS = ("gemini", "anthropic", "openai", "nvidia", "ollama")


def _unique(values: List[str]) -> List[str]:
    return sorted({v.strip() for v in values if isinstance(v, str) and v.strip()})


def list_models(provider: str, timeout: float = 8) -> Dict[str, Any]:
    """Return a safe model catalogue; API keys never leave this process."""
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    settings = config.settings
    configured = True
    headers: Dict[str, str] = {}

    if provider == "gemini":
        configured = bool(settings.gemini_api_key)
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        params = {"key": settings.gemini_api_key} if configured else {}
    elif provider == "anthropic":
        configured = bool(settings.anthropic_api_key)
        url = "https://api.anthropic.com/v1/models"
        headers = {"x-api-key": settings.anthropic_api_key or "",
                   "anthropic-version": "2023-06-01"}
        params = {}
    elif provider in {"openai", "nvidia"}:
        key = settings.openai_api_key if provider == "openai" else settings.nvidia_api_key
        base = settings.openai_base_url if provider == "openai" else settings.nvidia_base_url
        configured = bool(key)
        url = base.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {key or ''}"}
        params = {}
    else:
        url = settings.ollama_base_url.rstrip("/") + "/api/tags"
        params = {}

    active = getattr(settings, f"{provider}_model")
    if not configured:
        return {"provider": provider, "configured": False, "active": active,
                "models": [], "error": "Add this provider's API key first."}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if provider == "gemini":
            models = [item.get("name", "").removeprefix("models/")
                      for item in payload.get("models", [])
                      if "generateContent" in item.get("supportedGenerationMethods", [])]
        elif provider == "ollama":
            models = [item.get("model") or item.get("name", "")
                      for item in payload.get("models", [])]
        else:
            models = [item.get("id", "") for item in payload.get("data", [])]
        models = _unique(models)[:300]
        return {"provider": provider, "configured": True, "active": active,
                "models": models, "error": None}
    except (requests.RequestException, ValueError, TypeError) as exc:
        return {"provider": provider, "configured": configured, "active": active,
                "models": [active] if active else [],
                "error": f"Could not load models: {exc}"}


def list_all_models() -> Dict[str, Any]:
    return {"active_provider": config.settings.resolve_provider(),
            "providers": [list_models(name) for name in PROVIDERS]}

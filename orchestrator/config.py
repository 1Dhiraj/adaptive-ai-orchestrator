"""Configuration, loaded from the environment with sane defaults.

Reads ``.env.local`` then ``.env`` from the project root if present, without
requiring python-dotenv (falls back to a tiny parser).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    """Writable per-user application data, overridable by the launcher."""
    configured = os.environ.get("ORCHESTRATOR_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    # Source checkouts keep state beside the source. npm always sets this
    # variable, while a standalone packaged binary uses the OS directory.
    if not getattr(sys, "frozen", False):
        return PROJECT_ROOT
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "AdaptiveAIOrchestrator"
    if os.sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AdaptiveAIOrchestrator"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / \
        "adaptive-ai-orchestrator"


def _load_env_files() -> None:
    """Populate os.environ from .env.local / .env without clobbering real vars."""
    for name in (".env.local", ".env"):
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            # Real environment variables always win over dotenv files.
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_files()


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # -- LLM ------------------------------------------------------------
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))

    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"))

    openai_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    #: NVIDIA NIM: OpenAI-compatible, 100+ hosted open models, free tier.
    nvidia_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY"))
    #: Current NVIDIA-hosted general instruction model.
    nvidia_model: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"))
    nvidia_base_url: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_BASE_URL",
                                               "https://integrate.api.nvidia.com/v1"))

    #: No API key needed -- a local server. Never auto-selected on key
    #: presence like the others; only used when explicitly requested via
    #: LLM_PROVIDER=ollama, since assuming a local server is running (and
    #: possibly hanging on a connection attempt) would be surprising.
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.1"))

    #: Explicit override: "gemini" | "anthropic" | "openai" | "ollama" | "stub".
    #: Unset -> auto-detect from whichever API key is present (see
    #: resolve_provider()).
    llm_provider: Optional[str] = field(
        default_factory=lambda: (os.environ.get("LLM_PROVIDER") or "").strip().lower() or None)

    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 3))
    llm_timeout_s: float = field(default_factory=lambda: float(os.environ.get("LLM_TIMEOUT_S", 120)))
    #: Force the deterministic offline provider even when a key is present.
    force_stub_llm: bool = field(default_factory=lambda: _flag("ORCHESTRATOR_STUB_LLM"))

    # -- orchestration ----------------------------------------------------
    max_workers: int = field(default_factory=lambda: _int("ORCHESTRATOR_MAX_WORKERS", 4))
    #: Per-dependency character budget when assembling shared context.
    context_char_budget: int = field(default_factory=lambda: _int("ORCHESTRATOR_CONTEXT_BUDGET", 1200))

    # -- persistence ------------------------------------------------------
    db_path: str = field(
        default_factory=lambda: os.environ.get(
            "ORCHESTRATOR_DB", str(default_data_dir() / "orchestrator_state.db"))
    )

    # -- tool credentials (absent -> that tool runs in simulation mode) ---
    github_token: Optional[str] = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN"))
    github_repo: Optional[str] = field(default_factory=lambda: os.environ.get("GITHUB_REPO"))
    postgres_dsn: Optional[str] = field(default_factory=lambda: os.environ.get("DATABASE_URL"))
    slack_webhook: Optional[str] = field(default_factory=lambda: os.environ.get("SLACK_WEBHOOK_URL"))
    sendgrid_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("SENDGRID_API_KEY"))

    def resolve_provider(self) -> str:
        """Which provider ``get_provider()`` will build.

        Precedence: forced stub > explicit LLM_PROVIDER > first API key found
        (gemini, then anthropic, then openai) > stub. Ollama is deliberately
        never auto-selected -- it needs no key, so key-presence can't detect
        it, and guessing a local server is running (and blocking on a
        connection attempt) would surprise a user who has no intention of
        using it. Ask for it explicitly with LLM_PROVIDER=ollama.
        """
        if self.force_stub_llm:
            return "stub"
        if self.llm_provider:
            return self.llm_provider
        if self.gemini_api_key:
            return "gemini"
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        if self.nvidia_api_key:
            return "nvidia"
        return "stub"

    @property
    def llm_is_live(self) -> bool:
        return self.resolve_provider() != "stub"

    def describe(self) -> Dict[str, object]:
        """Non-secret summary, safe to log or show in the dashboard."""
        provider = self.resolve_provider()
        model = {
            "gemini": self.gemini_model, "anthropic": self.anthropic_model,
            "openai": self.openai_model, "ollama": self.ollama_model,
            "nvidia": self.nvidia_model,
        }.get(provider, "stub")
        return {
            "llm_mode": provider if provider != "stub" else "stub (deterministic, offline)",
            "model": model,
            "max_workers": self.max_workers,
            "db_path": self.db_path,
            "live_tools": [
                name
                for name, present in {
                    "github": bool(self.github_token),
                    "postgres": bool(self.postgres_dsn),
                    "slack": bool(self.slack_webhook),
                    "email": bool(self.sendgrid_api_key),
                    "gmail": bool(os.environ.get("GMAIL_ADDRESS")
                                  and os.environ.get("GMAIL_APP_PASSWORD")),
                    "hermes_desktop": bool(
                        os.environ.get("ORCHESTRATOR_ALLOW_DESKTOP", "").strip().lower()
                        in {"1", "true", "yes", "on"}
                        and shutil.which(os.environ.get("HERMES_EXECUTABLE", "hermes"))
                    ),
                }.items()
                if present
            ],
        }


settings = Settings()


def reload_settings() -> Settings:
    """Re-read the environment. Mainly for tests that monkeypatch os.environ."""
    global settings
    settings = Settings()
    return settings

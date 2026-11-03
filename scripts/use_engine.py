"""Switch which engine answers the orchestrator's model calls.

Two are configured:

    together   Together AI, per-token billing. Tested end to end: 1050 passing
               tests and nine verified demo prompts at about $0.011 a run.
    codex      The locally signed-in Codex CLI, using the ChatGPT plan without
               an API key. Each model call is a read-only CLI turn.

Switching is one command either way. Together uses the OpenAI-compatible
Chat Completions adapter; Codex uses its locally signed-in CLI provider.
The planner, DAG, change-awareness and dashboard stay the same.

    python scripts/use_engine.py              # show which is active
    python scripts/use_engine.py codex
    python scripts/use_engine.py gemini   # backup if Codex usage is exhausted
    python scripts/use_engine.py together

The point of the script, rather than editing .env.local by hand, is that
going back is as cheap as going forward. An engine swap the evening before a
demo should never be a one-way door.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env.local"

MANAGED = ("LLM_PROVIDER", "CODEX_MODEL", "GEMINI_MODEL", "CODEX_AGENT_DISABLED",
           "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
           "OPENAI_MODEL_COMPUTER_OPERATOR", "OPENAI_MODEL_SCREEN_LOCATOR")

TOGETHER = {
    "LLM_PROVIDER": "openai",
    "OPENAI_BASE_URL": "https://api.together.xyz/v1",
    "OPENAI_MODEL": "deepseek-ai/DeepSeek-V4.1-Flash",
    "OPENAI_MODEL_SCREEN_LOCATOR": "MiniMaxAI/MiniMax-M3",
}


def read_env() -> list[str]:
    if not ENV_FILE.exists():
        raise SystemExit(f"no {ENV_FILE}")
    return ENV_FILE.read_text(encoding="utf-8").splitlines()


def current_value(lines: list[str], key: str) -> str:
    for line in lines:
        match = re.match(rf"^\s*{re.escape(key)}\s*=\s*(.*)$", line)
        if match:
            return match.group(1).strip()
    return ""


def together_key(lines: list[str]) -> str:
    """The Together key, kept under its own name so a swap cannot lose it."""
    return current_value(lines, "TOGETHER_AI")


def apply(values: dict) -> None:
    lines = read_env()
    # Drop the keys this script owns, then append the chosen set together, so
    # the file never ends up with one engine's URL and another's model.
    kept = [line for line in lines
            if not any(re.match(rf"^\s*{re.escape(key)}\s*=", line)
                       for key in MANAGED)]
    while kept and not kept[-1].strip():
        kept.pop()
    kept.append("")
    kept.append(f"# --- engine: {values.pop('_name')} "
                "(set by scripts/use_engine.py) ---")
    kept += [f"{key}={value}" for key, value in values.items()]
    ENV_FILE.write_text("\n".join(kept) + "\n", encoding="utf-8")


def describe() -> None:
    lines = read_env()
    base = current_value(lines, "OPENAI_BASE_URL")
    provider = current_value(lines, "LLM_PROVIDER")
    model = (current_value(lines, "CODEX_MODEL") if provider == "codex" else
             current_value(lines, "GEMINI_MODEL") if provider == "gemini" else
             current_value(lines, "OPENAI_MODEL"))
    engine = ("codex" if provider == "codex" else
              "gemini" if provider == "gemini" else
              "together" if "together" in base
              else "unknown")
    print(f"engine   : {engine}")
    print(f"base_url : {base or '(not needed)'}")
    print(f"model    : {model or '(unset)'}")
    for role in ("OPENAI_MODEL_COMPUTER_OPERATOR", "OPENAI_MODEL_SCREEN_LOCATOR"):
        value = current_value(lines, role)
        if value:
            print(f"{role.removeprefix('OPENAI_MODEL_').lower():9}: {value}")


def main() -> int:
    if len(sys.argv) < 2:
        describe()
        return 0

    choice = sys.argv[1].strip().lower()
    if choice == "together":
        lines = read_env()
        key = together_key(lines)
        if not key:
            raise SystemExit("TOGETHER_AI is not set in .env.local")
        apply({"_name": "together", **TOGETHER, "OPENAI_API_KEY": key})
    elif choice == "codex":
        apply({"_name": "codex", "LLM_PROVIDER": "codex", "CODEX_MODEL": "gpt-6.1-sol"})
    elif choice == "gemini":
        lines = read_env()
        if not current_value(lines, "GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY is not set in .env.local")
        apply({"_name": "gemini", "LLM_PROVIDER": "gemini",
               "GEMINI_MODEL": "gemini-2.5-flash", "CODEX_AGENT_DISABLED": "1"})
    else:
        raise SystemExit(f"unknown engine {choice!r}; use codex, gemini or together")

    print(f"switched to {choice}\n")
    describe()
    print("\nRestart anything already running (the dashboard reads this at "
          "startup):\n  python -m orchestrator.cli serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Adaptive AI Task Orchestrator - entry point.

    python main.py                      # run the four headline scenarios
    python -m orchestrator.cli --help   # full CLI (plan / run / resume / serve / bench)

The system runs with zero configuration: with no ``GEMINI_API_KEY`` set it
uses a deterministic offline LLM so every scenario, test and benchmark is
reproducible. Set the key to switch to real Gemini calls.
"""

from orchestrator.demo import run_demo

if __name__ == "__main__":
    raise SystemExit(run_demo())

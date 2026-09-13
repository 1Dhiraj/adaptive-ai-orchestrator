"""Reproducible benchmarks against baseline orchestration strategies."""

from .harness import Scenario, SystemAdapter, Trial, TrialResult, default_scenarios, run_matrix

__all__ = [
    "Scenario", "SystemAdapter", "Trial", "TrialResult",
    "default_scenarios", "run_matrix",
]

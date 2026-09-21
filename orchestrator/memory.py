"""Shared context between agents.

Every agent sees the outputs of the steps it depends on -- that is what stops
each step from starting cold. The manager also computes the *input hash* for
a step, which is what lets the orchestrator decide whether a cached result is
still valid.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .config import settings
from .graph import DependencyGraph
from .models import Step, content_hash


def truncate(text: str, budget: int) -> str:
    """Keep the head and tail of long text; the middle is usually filler."""
    if len(text) <= budget:
        return text
    head = int(budget * 0.6)
    tail = budget - head - 20
    return f"{text[:head]}\n...[{len(text) - budget} chars elided]...\n{text[-tail:]}"


class MemoryManager:
    """Thread-safe store of step outputs plus context assembly."""

    def __init__(self, char_budget: Optional[int] = None):
        self._outputs: Dict[str, str] = {}
        self._lock = threading.RLock()
        self.char_budget = char_budget or settings.context_char_budget

    # -- storage ----------------------------------------------------------

    def store(self, step_id: str, output: str) -> None:
        with self._lock:
            self._outputs[step_id] = output

    def get(self, step_id: str, default: str = "") -> str:
        with self._lock:
            return self._outputs.get(step_id, default)

    def forget(self, step_id: str) -> None:
        with self._lock:
            self._outputs.pop(step_id, None)

    def clear(self) -> None:
        with self._lock:
            self._outputs.clear()

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._outputs)

    def load(self, outputs: Dict[str, str]) -> None:
        with self._lock:
            self._outputs.update(outputs)

    def __contains__(self, step_id: object) -> bool:
        with self._lock:
            return step_id in self._outputs

    # -- context assembly --------------------------------------------------

    def dependency_outputs(self, step: Step) -> Dict[str, str]:
        with self._lock:
            return {dep: self._outputs.get(dep, "") for dep in step.depends_on}

    def build_context(self, step: Step, graph: Optional[DependencyGraph] = None,
                      char_budget: Optional[int] = None) -> str:
        """Render the prerequisite outputs as a prompt-ready block."""
        if not step.depends_on:
            return "(this step has no prerequisites)"

        blocks: List[str] = []
        for dep in step.depends_on:
            output = self.get(dep)
            label = dep
            if graph is not None and dep in graph:
                upstream = graph.get(dep)
                label = f"{dep} ({upstream.agent_role})"
            budget = char_budget or self.char_budget
            body = truncate(output, budget) if output else "(not yet produced)"
            blocks.append(f"### Output of `{label}`\n{body}")
        return "\n\n".join(blocks)

    def input_hash(self, step: Step) -> str:
        """Signature of everything feeding this step.

        Combines the step's own definition with the exact outputs of its
        dependencies. If this matches the hash recorded on the last successful
        run, the cached output is provably still valid and the step can be
        skipped -- even after an unrelated part of the graph changed.
        """
        parts = [step.definition_hash()]
        for dep in sorted(step.depends_on):
            parts.append(f"{dep}={content_hash(self.get(dep))}")
        return content_hash(*parts)

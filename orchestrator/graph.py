"""Layer 2 -- the dependency graph and impact analysis.

This module is the foundation of adaptive re-execution: it answers
"if X changed, what else is now wrong?" without running anything.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set

from .models import Step


class GraphError(Exception):
    """Base class for malformed-graph problems."""


class MissingDependencyError(GraphError):
    def __init__(self, step_id: str, missing: List[str]):
        self.step_id = step_id
        self.missing = missing
        super().__init__(f"step '{step_id}' depends on unknown step(s): {', '.join(missing)}")


class CircularDependencyError(GraphError):
    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__("circular dependency: " + " -> ".join(cycle))


class DependencyGraph:
    """A DAG of :class:`Step` objects with impact analysis.

    Ordering is deterministic (ties broken by insertion index) so that two
    runs of the same workflow schedule steps identically -- which is what
    makes the benchmark numbers comparable.
    """

    def __init__(self, steps: Optional[Iterable[Step]] = None):
        self.steps: Dict[str, Step] = {}
        self._insertion_index: Dict[str, int] = {}
        for step in steps or []:
            self.add(step)

    # -- construction -----------------------------------------------------

    def add(self, step: Step) -> "DependencyGraph":
        if step.id in self.steps:
            raise GraphError(f"duplicate step id: '{step.id}'")
        self.steps[step.id] = step
        self._insertion_index[step.id] = len(self._insertion_index)
        return self

    def remove(self, step_id: str) -> None:
        """Delete a step and drop it from every other step's dependencies."""
        self.steps.pop(step_id, None)
        self._insertion_index.pop(step_id, None)
        for step in self.steps.values():
            if step_id in step.depends_on:
                step.depends_on = [d for d in step.depends_on if d != step_id]

    def __contains__(self, step_id: object) -> bool:
        return step_id in self.steps

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps.values())

    def get(self, step_id: str) -> Step:
        if step_id not in self.steps:
            raise KeyError(f"no such step: '{step_id}'")
        return self.steps[step_id]

    # -- validation -------------------------------------------------------

    def validate(self) -> None:
        """Raise if any dependency is dangling or the graph has a cycle."""
        for step in self.steps.values():
            missing = [d for d in step.depends_on if d not in self.steps]
            if missing:
                raise MissingDependencyError(step.id, missing)
        cycle = self.find_cycle()
        if cycle:
            raise CircularDependencyError(cycle)

    def find_cycle(self) -> Optional[List[str]]:
        """Return one cycle as a node path, or ``None`` if the graph is acyclic."""
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {sid: WHITE for sid in self.steps}
        stack: List[str] = []

        def visit(node: str) -> Optional[List[str]]:
            colour[node] = GREY
            stack.append(node)
            for dep in self._sorted(self.steps[node].depends_on):
                if dep not in self.steps:
                    continue
                if colour[dep] == GREY:
                    # Found a back edge: slice the cycle out of the stack.
                    return stack[stack.index(dep):] + [dep]
                if colour[dep] == WHITE:
                    found = visit(dep)
                    if found:
                        return found
            stack.pop()
            colour[node] = BLACK
            return None

        for sid in self._sorted(self.steps):
            if colour[sid] == WHITE:
                found = visit(sid)
                if found:
                    return found
        return None

    def break_cycles(self) -> List[tuple]:
        """Make the graph acyclic by dropping the last edge of each cycle.

        Returns the list of ``(step_id, removed_dependency)`` edges dropped.
        Used by :class:`~orchestrator.planner.TaskPlanner` to repair plans an
        LLM produced with circular references.
        """
        removed = []
        while True:
            cycle = self.find_cycle()
            if not cycle:
                return removed
            # cycle looks like [a, b, c, a]; drop the edge that closes it.
            offender, dep = cycle[-2], cycle[-1]
            self.steps[offender].depends_on = [
                d for d in self.steps[offender].depends_on if d != dep
            ]
            removed.append((offender, dep))

    def prune_dangling_dependencies(self) -> List[tuple]:
        """Drop references to steps that do not exist. Returns dropped edges."""
        removed = []
        for step in self.steps.values():
            for dep in list(step.depends_on):
                if dep not in self.steps:
                    step.depends_on = [d for d in step.depends_on if d != dep]
                    removed.append((step.id, dep))
        return removed

    # -- ordering ---------------------------------------------------------

    def _sorted(self, ids: Iterable[str]) -> List[str]:
        """Deterministic ordering: insertion order, then id."""
        return sorted(ids, key=lambda s: (self._insertion_index.get(s, 1 << 30), s))

    def topological_order(self) -> List[str]:
        """All step ids in a valid execution order (Kahn's algorithm)."""
        indegree = {sid: 0 for sid in self.steps}
        for step in self.steps.values():
            for dep in step.depends_on:
                if dep in self.steps:
                    indegree[step.id] += 1

        ready = deque(self._sorted([s for s, d in indegree.items() if d == 0]))
        order: List[str] = []
        while ready:
            node = ready.popleft()
            order.append(node)
            newly_ready = []
            for sid, step in self.steps.items():
                if node in step.depends_on and sid in indegree:
                    indegree[sid] -= 1
                    if indegree[sid] == 0:
                        newly_ready.append(sid)
            for sid in self._sorted(newly_ready):
                ready.append(sid)

        if len(order) != len(self.steps):
            raise CircularDependencyError(self.find_cycle() or sorted(set(self.steps) - set(order)))
        return order

    def execution_levels(self) -> List[List[str]]:
        """Group steps into waves that may run concurrently.

        Level *n* contains every step whose dependencies all live in levels
        ``< n``. This is what enables parallel execution.
        """
        depth: Dict[str, int] = {}
        for sid in self.topological_order():
            deps = [d for d in self.steps[sid].depends_on if d in self.steps]
            depth[sid] = 0 if not deps else max(depth[d] for d in deps) + 1

        levels: List[List[str]] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
        for sid, d in depth.items():
            levels[d].append(sid)
        return [self._sorted(level) for level in levels]

    # -- impact analysis (the adaptive core) ------------------------------

    def downstream_of(self, step_id: str, inclusive: bool = True) -> Set[str]:
        """Every step that transitively consumes ``step_id``'s output."""
        if step_id not in self.steps:
            raise KeyError(f"no such step: '{step_id}'")

        # Build the reverse adjacency once, then BFS.
        dependents: Dict[str, List[str]] = {sid: [] for sid in self.steps}
        for sid, step in self.steps.items():
            for dep in step.depends_on:
                if dep in dependents:
                    dependents[dep].append(sid)

        seen: Set[str] = set()
        queue = deque([step_id])
        while queue:
            node = queue.popleft()
            for child in dependents[node]:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)

        if inclusive:
            seen.add(step_id)
        else:
            seen.discard(step_id)
        return seen

    def upstream_of(self, step_id: str, inclusive: bool = False) -> Set[str]:
        """Every step ``step_id`` transitively depends on."""
        if step_id not in self.steps:
            raise KeyError(f"no such step: '{step_id}'")
        seen: Set[str] = set()
        queue = deque(self.steps[step_id].depends_on)
        while queue:
            node = queue.popleft()
            if node in seen or node not in self.steps:
                continue
            seen.add(node)
            queue.extend(self.steps[node].depends_on)
        if inclusive:
            seen.add(step_id)
        return seen

    def affected_by(self, step_ids: Iterable[str]) -> Set[str]:
        """Union of the downstream cones of several changed steps."""
        affected: Set[str] = set()
        for sid in step_ids:
            affected |= self.downstream_of(sid)
        return affected

    def nodes_using_tool(self, tool: str) -> Set[str]:
        return {sid for sid, s in self.steps.items() if s.requires_tool == tool}

    def roots(self) -> List[str]:
        return self._sorted([s for s, step in self.steps.items() if not step.depends_on])

    def leaves(self) -> List[str]:
        has_dependents = {d for step in self.steps.values() for d in step.depends_on}
        return self._sorted([s for s in self.steps if s not in has_dependents])

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "steps": [self.steps[sid].to_dict() for sid in self.topological_order()],
            "levels": self.execution_levels(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        return cls(Step.from_dict(s) for s in data["steps"])

    def to_mermaid(self) -> str:
        """Render the DAG as a mermaid flowchart (handy for READMEs/reports)."""
        lines = ["flowchart TD"]
        for sid in self.topological_order():
            step = self.steps[sid]
            label = step.name
            if step.requires_tool:
                label += f"<br/><i>{step.requires_tool}</i>"
            lines.append(f'    {sid}["{label}"]')
        for sid in self.topological_order():
            for dep in self.steps[sid].depends_on:
                lines.append(f"    {dep} --> {sid}")
        return "\n".join(lines)

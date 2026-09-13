"""DependencyGraph: ordering, levels, validation and impact analysis."""

from __future__ import annotations

import pytest

from orchestrator.graph import (
    CircularDependencyError,
    DependencyGraph,
    GraphError,
    MissingDependencyError,
)
from orchestrator.models import Step


def s(step_id: str, *deps: str, tool: str = None) -> Step:
    return Step(id=step_id, description=f"do {step_id}", agent_role="generic",
                depends_on=list(deps), requires_tool=tool)


class TestOrdering:
    def test_topological_order_respects_dependencies(self, linear_graph):
        order = linear_graph.topological_order()
        assert order.index("frontend") < order.index("backend")
        assert order.index("backend") < order.index("database")
        assert order.index("database") < order.index("testing")

    def test_order_is_deterministic(self, diamond_graph):
        assert diamond_graph.topological_order() == diamond_graph.topological_order()

    def test_execution_levels_group_concurrent_steps(self, diamond_graph):
        assert diamond_graph.execution_levels() == [["a"], ["b", "c"], ["d"]]

    def test_linear_graph_has_one_step_per_level(self, linear_graph):
        assert all(len(level) == 1 for level in linear_graph.execution_levels())

    def test_levels_cover_every_step_exactly_once(self, diamond_graph):
        flat = [sid for level in diamond_graph.execution_levels() for sid in level]
        assert sorted(flat) == sorted(diamond_graph.topological_order())

    def test_wide_graph_collapses_to_two_levels(self):
        graph = DependencyGraph([s("root")] + [s(f"leaf{i}", "root") for i in range(6)])
        levels = graph.execution_levels()
        assert levels[0] == ["root"]
        assert len(levels) == 2 and len(levels[1]) == 6


class TestValidation:
    def test_dangling_dependency_is_rejected(self):
        graph = DependencyGraph([s("a", "nope")])
        with pytest.raises(MissingDependencyError) as excinfo:
            graph.validate()
        assert "nope" in str(excinfo.value)

    def test_cycle_is_detected(self):
        graph = DependencyGraph([s("a", "b"), s("b", "a")])
        with pytest.raises(CircularDependencyError):
            graph.validate()

    def test_find_cycle_returns_the_path(self):
        graph = DependencyGraph([s("a", "c"), s("b", "a"), s("c", "b")])
        cycle = graph.find_cycle()
        assert cycle is not None
        assert cycle[0] == cycle[-1]  # closed loop
        assert set(cycle) == {"a", "b", "c"}

    def test_acyclic_graph_has_no_cycle(self, diamond_graph):
        assert diamond_graph.find_cycle() is None

    def test_self_dependency_is_dropped_at_construction(self):
        assert Step(id="a", description="x", agent_role="generic", depends_on=["a"]).depends_on == []

    def test_duplicate_ids_are_rejected(self):
        with pytest.raises(GraphError):
            DependencyGraph([s("a"), s("a")])

    def test_break_cycles_makes_the_graph_valid(self):
        graph = DependencyGraph([s("a", "c"), s("b", "a"), s("c", "b")])
        removed = graph.break_cycles()
        assert removed
        graph.validate()

    def test_prune_dangling_reports_what_it_removed(self):
        graph = DependencyGraph([s("a", "ghost"), s("b", "a")])
        assert graph.prune_dangling_dependencies() == [("a", "ghost")]
        graph.validate()


class TestImpactAnalysis:
    def test_downstream_includes_the_changed_step(self, linear_graph):
        assert "database" in linear_graph.downstream_of("database")

    def test_downstream_is_transitive(self, linear_graph):
        assert linear_graph.downstream_of("backend") == {"backend", "database", "testing"}

    def test_downstream_of_a_leaf_is_only_itself(self, linear_graph):
        assert linear_graph.downstream_of("testing") == {"testing"}

    def test_downstream_of_root_is_everything(self, linear_graph):
        assert linear_graph.downstream_of("frontend") == set(linear_graph.topological_order())

    def test_downstream_exclusive_drops_the_seed(self, linear_graph):
        assert linear_graph.downstream_of("backend", inclusive=False) == {"database", "testing"}

    def test_sibling_branches_are_not_affected(self, diamond_graph):
        assert diamond_graph.downstream_of("b") == {"b", "d"}
        assert "c" not in diamond_graph.downstream_of("b")

    def test_upstream_walks_the_other_way(self, linear_graph):
        assert linear_graph.upstream_of("database") == {"frontend", "backend"}

    def test_affected_by_unions_multiple_seeds(self, diamond_graph):
        assert diamond_graph.affected_by(["b", "c"]) == {"b", "c", "d"}

    def test_unknown_step_raises(self, linear_graph):
        with pytest.raises(KeyError):
            linear_graph.downstream_of("ghost")

    def test_nodes_using_tool(self, linear_graph):
        assert linear_graph.nodes_using_tool("github") == {"backend"}
        assert linear_graph.nodes_using_tool("nothing") == set()

    def test_roots_and_leaves(self, diamond_graph):
        assert diamond_graph.roots() == ["a"]
        assert diamond_graph.leaves() == ["d"]


class TestSerialisation:
    def test_round_trip_preserves_structure(self, diamond_graph):
        restored = DependencyGraph.from_dict(diamond_graph.to_dict())
        assert restored.topological_order() == diamond_graph.topological_order()
        assert restored.execution_levels() == diamond_graph.execution_levels()

    def test_remove_detaches_dependents(self, linear_graph):
        linear_graph.remove("backend")
        assert "backend" not in linear_graph
        assert linear_graph.get("database").depends_on == []
        linear_graph.validate()

    def test_mermaid_contains_every_edge(self, diamond_graph):
        mermaid = diamond_graph.to_mermaid()
        assert "a --> b" in mermaid and "b --> d" in mermaid

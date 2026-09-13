"""Production boundary tests: concurrency, tenant isolation and PostgreSQL."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from orchestrator import DependencyGraph, StateManager, Step
from orchestrator.tenancy import tenant_scope


def _graph():
    return DependencyGraph([Step(id="work", description="work", agent_role="worker")])


def test_concurrent_state_writes_are_not_lost(tmp_path):
    state = StateManager(str(tmp_path / "load.db"))
    def create(index):
        state.create_run(f"load-{index:03d}", "load test", _graph())
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(create, range(100)))
    assert len(state.list_runs(limit=200)) == 100
    state.close()


@pytest.mark.postgres
def test_shared_postgres_is_tenant_isolated(monkeypatch):
    url = os.environ.get("ORCHESTRATOR_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("set ORCHESTRATOR_TEST_POSTGRES_URL for the PostgreSQL integration test")
    monkeypatch.setenv("ORCHESTRATOR_STATE_URL", url)
    state = StateManager()
    run_id = "integration-tenant-isolation"
    with tenant_scope("organization-a"):
        state.create_run(run_id, "private", _graph())
        assert state.get_run(run_id)
    with tenant_scope("organization-b"):
        assert state.get_run(run_id) is None
    with tenant_scope("organization-a"):
        state.delete_run(run_id)
    state.close()


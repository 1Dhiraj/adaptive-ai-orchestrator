"""Request and worker scoped organization identity."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_tenant: ContextVar[str] = ContextVar("orchestrator_tenant", default="default")


def current_tenant() -> str:
    return _tenant.get()


@contextmanager
def tenant_scope(tenant_id: str) -> Iterator[None]:
    token = _tenant.set(tenant_id or "default")
    try:
        yield
    finally:
        _tenant.reset(token)


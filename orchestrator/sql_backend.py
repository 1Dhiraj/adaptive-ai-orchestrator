"""Small DB-API compatibility layer for shared SQLAlchemy databases.

StateManager keeps a deliberately simple SQL surface.  This adapter lets the
same persistence code use PostgreSQL without maintaining two implementations.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence


class ResultAdapter:
    def __init__(self, result: Any, lastrowid: Optional[int] = None):
        self._result = result
        self.lastrowid = lastrowid
        self.rowcount = result.rowcount

    def fetchone(self):
        row = self._result.mappings().fetchone()
        return row

    def fetchall(self):
        return self._result.mappings().fetchall()


class SQLAlchemyConnection:
    """Subset of sqlite3.Connection used by StateManager."""

    _ID_TABLES = {"workflow_schedules", "workflow_webhooks", "execution_jobs"}

    def __init__(self, url: str):
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise RuntimeError(
                "SQLAlchemy is required for ORCHESTRATOR_STATE_URL; install production dependencies"
            ) from exc
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        self.connection = self.engine.connect()

    @staticmethod
    def _prepare(sql: str, params: Sequence[Any] = ()):
        parts = sql.split("?")
        if len(parts) - 1 != len(params):
            return sql, {}
        statement = parts[0]
        values = {}
        for index, value in enumerate(params):
            key = f"p{index}"
            statement += f":{key}" + parts[index + 1]
            values[key] = value
        return statement, values

    def execute(self, sql: str, params: Sequence[Any] = ()) -> ResultAdapter:
        from sqlalchemy import text
        statement, values = self._prepare(sql, params)
        match = re.match(r"\s*INSERT\s+INTO\s+(\w+)", statement, re.IGNORECASE)
        wants_id = bool(match and match.group(1).lower() in self._ID_TABLES)
        if wants_id and "RETURNING" not in statement.upper():
            statement += " RETURNING id"
        result = self.connection.execute(text(statement), values)
        lastrowid = int(result.scalar_one()) if wants_id else None
        return ResultAdapter(result, lastrowid)

    def executemany(self, sql: str, params: Iterable[Sequence[Any]]) -> ResultAdapter:
        from sqlalchemy import text
        rows = list(params)
        if not rows:
            class Empty:
                rowcount = 0
            return ResultAdapter(Empty())
        statement, _ = self._prepare(sql, rows[0])
        values = [self._prepare(sql, row)[1] for row in rows]
        return ResultAdapter(self.connection.execute(text(statement), values))

    def executescript(self, script: str) -> None:
        for statement in script.split(";"):
            if statement.strip():
                self.execute(statement)

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def close(self) -> None:
        self.connection.close()
        self.engine.dispose()


"""Layer 6a -- durable state, so a crash costs you one step and not a run.

Everything needed to resume lives in SQLite for local use or PostgreSQL for
multi-machine deployments: the graph, every step result, and the event log.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .events import Event, EventType
from .graph import DependencyGraph
from .models import PendingAction, StepResult, StepStatus
from .tenancy import current_tenant

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    description  TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'created',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    graph_json   TEXT NOT NULL DEFAULT '{}',
    meta_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS step_results (
    run_id       TEXT NOT NULL,
    step_id      TEXT NOT NULL,
    result_json  TEXT NOT NULL,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (run_id, step_id)
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    ts           REAL NOT NULL,
    type         TEXT NOT NULL,
    step_id      TEXT,
    message      TEXT NOT NULL DEFAULT '',
    data_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS pending_actions (
    run_id       TEXT NOT NULL,
    step_id      TEXT NOT NULL,
    action_json  TEXT NOT NULL,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (run_id, step_id)
);

CREATE TABLE IF NOT EXISTS workflow_schedules (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    interval_seconds INTEGER NOT NULL,
    enabled          INTEGER NOT NULL DEFAULT 1,
    next_run_at      REAL NOT NULL,
    last_run_at      REAL,
    created_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_webhooks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    tenant_id         TEXT NOT NULL DEFAULT 'default',
    name              TEXT NOT NULL DEFAULT 'Webhook',
    token_hash        TEXT NOT NULL UNIQUE,
    enabled           INTEGER NOT NULL DEFAULT 1,
    last_triggered_at REAL,
    created_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_templates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    graph_json  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status INTEGER NOT NULL,
    client TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS execution_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    operation TEXT NOT NULL,
    args_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_steps_run ON step_results(run_id);
CREATE INDEX IF NOT EXISTS idx_actions_run ON pending_actions(run_id);
CREATE INDEX IF NOT EXISTS idx_schedules_due ON workflow_schedules(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS idx_webhooks_run ON workflow_webhooks(run_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON execution_jobs(status, created_at);
"""

_POSTGRES_SCHEMA = _SCHEMA.replace(
    "INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"
)


class StateManager:
    """Durable persistence backed by SQLite locally or shared PostgreSQL."""

    def __init__(self, db_path: Optional[str] = None):
        state_url = os.environ.get("ORCHESTRATOR_STATE_URL") if db_path is None else None
        self.db_path = state_url or db_path or settings.db_path
        self.backend = "postgresql" if state_url else "sqlite"
        self._lock = threading.RLock()
        if state_url:
            from .sql_backend import SQLAlchemyConnection
            self._conn = SQLAlchemyConnection(state_url)
            with self._lock:
                self._conn.executescript(_POSTGRES_SCHEMA)
                self._conn.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default'")
                self._conn.execute(
                    "ALTER TABLE execution_jobs ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default'")
                for table in ("workflow_schedules", "workflow_webhooks", "workflow_templates",
                              "audit_log"):
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default'")
                self._conn.commit()
            return
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL keeps the dashboard's reads from blocking the orchestrator's
            # writes; harmless (and ignored) for in-memory databases.
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            self._conn.executescript(_SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(runs)").fetchall()}
            if "tenant_id" not in columns:
                self._conn.execute("ALTER TABLE runs ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            job_columns = {row[1] for row in self._conn.execute(
                "PRAGMA table_info(execution_jobs)").fetchall()}
            if "tenant_id" not in job_columns:
                self._conn.execute("ALTER TABLE execution_jobs ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            for table in ("workflow_schedules", "workflow_webhooks", "workflow_templates", "audit_log"):
                table_columns = {row[1] for row in self._conn.execute(
                    f"PRAGMA table_info({table})").fetchall()}
                if "tenant_id" not in table_columns:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "StateManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- runs -------------------------------------------------------------

    def create_run(self, run_id: str, description: str, graph: DependencyGraph,
                   meta: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs "
                "(run_id, tenant_id, description, status, created_at, updated_at, graph_json, meta_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
                "description=excluded.description, status=excluded.status, "
                "updated_at=excluded.updated_at, graph_json=excluded.graph_json, "
                "meta_json=excluded.meta_json WHERE runs.tenant_id=excluded.tenant_id",
                (run_id, current_tenant(), description, "created", now, now,
                 json.dumps(graph.to_dict()), json.dumps(meta or {})),
            )
            self._conn.commit()
        if self.get_run(run_id) is None:
            raise ValueError(f"run id '{run_id}' is already owned by another organization")

    def update_run(self, run_id: str, status: Optional[str] = None,
                   graph: Optional[DependencyGraph] = None,
                   meta: Optional[Dict[str, Any]] = None) -> None:
        sets, params = ["updated_at = ?"], [time.time()]
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if graph is not None:
            sets.append("graph_json = ?")
            params.append(json.dumps(graph.to_dict()))
        if meta is not None:
            sets.append("meta_json = ?")
            params.append(json.dumps(meta))
        params.extend([run_id, current_tenant()])
        with self._lock:
            self._conn.execute(
                f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ? AND tenant_id = ?", params)
            self._conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ? AND tenant_id = ?",
                                     (run_id, current_tenant())).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "description": row["description"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "graph": json.loads(row["graph_json"]),
            "meta": json.loads(row["meta_json"]),
        }

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                # rowid last, not run_id: two runs created within the same
                # clock tick tie on both timestamps, and falling back to the
                # id sorts them alphabetically -- which has nothing to do with
                # which is newer. Insertion order does.
                "SELECT run_id, description, status, created_at, updated_at "
                "FROM runs WHERE tenant_id = ? "
                "ORDER BY updated_at DESC, created_at DESC, rowid DESC LIMIT ?",
                (current_tenant(), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_run(self, run_id: str) -> None:
        if self.get_run(run_id) is None:
            raise KeyError(f"no persisted run with id '{run_id}'")
        with self._lock:
            self._conn.execute("DELETE FROM execution_jobs WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM workflow_webhooks WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM workflow_schedules WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM pending_actions WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM step_results WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._conn.commit()

    # -- persistent interval schedules -----------------------------------

    def create_schedule(self, run_id: str, interval_seconds: int,
                        next_run_at: Optional[float] = None) -> Dict[str, Any]:
        if self.get_run(run_id) is None:
            raise KeyError(f"no persisted run with id '{run_id}'")
        if interval_seconds < 60:
            raise ValueError("schedule interval must be at least 60 seconds")
        now = time.time()
        due = float(next_run_at if next_run_at is not None else now + interval_seconds)
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO workflow_schedules "
                "(run_id, tenant_id, interval_seconds, enabled, next_run_at, created_at) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (run_id, current_tenant(), int(interval_seconds), due, now),
            )
            self._conn.commit()
            schedule_id = int(cursor.lastrowid)
        return self.get_schedule(schedule_id) or {}

    def get_schedule(self, schedule_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM workflow_schedules WHERE id = ? AND tenant_id = ?",
                (schedule_id, current_tenant())
            ).fetchone()
        return dict(row) if row is not None else None

    def list_schedules(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM workflow_schedules"
        params: tuple = ()
        if run_id is not None:
            query += " WHERE run_id = ? AND tenant_id = ?"
            params = (run_id, current_tenant())
        else:
            query += " WHERE tenant_id = ?"
            params = (current_tenant(),)
        query += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def due_schedules(self, now: Optional[float] = None,
                      all_tenants: bool = False) -> List[Dict[str, Any]]:
        cutoff = time.time() if now is None else float(now)
        query = ("SELECT * FROM workflow_schedules WHERE enabled = 1 "
                 "AND next_run_at <= ? ORDER BY next_run_at" if all_tenants else
                 "SELECT * FROM workflow_schedules WHERE tenant_id = ? AND enabled = 1 "
                 "AND next_run_at <= ? ORDER BY next_run_at")
        params = (cutoff,) if all_tenants else (current_tenant(), cutoff)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def advance_schedule(self, schedule_id: int, ran_at: Optional[float] = None) -> None:
        now = time.time() if ran_at is None else float(ran_at)
        with self._lock:
            self._conn.execute(
                "UPDATE workflow_schedules SET last_run_at = ?, "
                "next_run_at = ? + interval_seconds WHERE id = ? AND tenant_id = ?",
                (now, now, schedule_id, current_tenant()),
            )
            self._conn.commit()

    def set_schedule_enabled(self, schedule_id: int, enabled: bool) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE workflow_schedules SET enabled = ? WHERE id = ? AND tenant_id = ?",
                (1 if enabled else 0, schedule_id, current_tenant()),
            )
            self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"no schedule with id '{schedule_id}'")

    def delete_schedule(self, schedule_id: int) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM workflow_schedules WHERE id = ? AND tenant_id = ?",
                (schedule_id, current_tenant())
            )
            self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"no schedule with id '{schedule_id}'")

    # -- inbound webhook triggers ---------------------------------------

    def create_webhook(self, run_id: str, name: str, token_hash: str) -> Dict[str, Any]:
        if self.get_run(run_id) is None:
            raise KeyError(f"no persisted run with id '{run_id}'")
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO workflow_webhooks "
                "(run_id, tenant_id, name, token_hash, enabled, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (run_id, current_tenant(), name.strip() or "Webhook", token_hash, now),
            )
            self._conn.commit()
            webhook_id = int(cursor.lastrowid)
        return self.get_webhook(webhook_id) or {}

    def get_webhook(self, webhook_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM workflow_webhooks WHERE id = ? AND tenant_id = ?",
                (webhook_id, current_tenant())
            ).fetchone()
        return dict(row) if row is not None else None

    def find_webhook(self, token_hash: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM workflow_webhooks WHERE token_hash = ?", (token_hash,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_webhooks(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, run_id, name, enabled, last_triggered_at, created_at "
                "FROM workflow_webhooks WHERE run_id = ? AND tenant_id = ? ORDER BY created_at DESC",
                (run_id, current_tenant()),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_webhook_triggered(self, webhook_id: int,
                               triggered_at: Optional[float] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE workflow_webhooks SET last_triggered_at = ? "
                "WHERE id = ? AND tenant_id = ?",
                (time.time() if triggered_at is None else float(triggered_at), webhook_id,
                 current_tenant()),
            )
            self._conn.commit()

    def set_webhook_enabled(self, webhook_id: int, enabled: bool) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE workflow_webhooks SET enabled = ? WHERE id = ? AND tenant_id = ?",
                (1 if enabled else 0, webhook_id, current_tenant()),
            )
            self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"no webhook with id '{webhook_id}'")

    def delete_webhook(self, webhook_id: int) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM workflow_webhooks WHERE id = ? AND tenant_id = ?",
                (webhook_id, current_tenant())
            )
            self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"no webhook with id '{webhook_id}'")

    # -- reusable workflow templates ------------------------------------

    def save_template(self, name: str, description: str,
                      graph: DependencyGraph) -> Dict[str, Any]:
        clean = name.strip()
        if not clean:
            raise ValueError("template name is required")
        now = time.time()
        stored_name = f"{current_tenant()}::{clean}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO workflow_templates "
                "(tenant_id, name, description, graph_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
                "description=excluded.description, graph_json=excluded.graph_json, "
                "updated_at=excluded.updated_at WHERE workflow_templates.tenant_id=excluded.tenant_id",
                (current_tenant(), stored_name, description, json.dumps(graph.to_dict()), now, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM workflow_templates WHERE name = ? AND tenant_id = ?",
                (stored_name, current_tenant())).fetchone()
        result = self._template_dict(row)
        result["name"] = clean
        return result

    @staticmethod
    def _template_dict(row: Any) -> Dict[str, Any]:
        if row is None:
            return {}
        data = dict(row)
        data["graph"] = json.loads(data.pop("graph_json"))
        if "::" in data.get("name", ""):
            data["name"] = data["name"].split("::", 1)[1]
        return data

    def get_template(self, template_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM workflow_templates WHERE id = ? AND tenant_id = ?",
                (template_id, current_tenant())).fetchone()
        return self._template_dict(row) if row is not None else None

    def list_templates(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM workflow_templates WHERE tenant_id = ? ORDER BY updated_at DESC",
                (current_tenant(),)).fetchall()
        return [self._template_dict(row) for row in rows]

    def delete_template(self, template_id: int) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM workflow_templates WHERE id = ? AND tenant_id = ?",
                (template_id, current_tenant()))
            self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"no template with id '{template_id}'")

    def record_audit(self, role: str, method: str, path: str,
                     status: int, client: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log (tenant_id, ts, role, method, path, status, client) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (current_tenant(), time.time(), role, method, path, int(status), client),
            )
            self._conn.commit()

    def load_audit(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_log WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
                (current_tenant(), limit)).fetchall()
        return [dict(row) for row in rows]

    # -- durable execution jobs -----------------------------------------

    def enqueue_job(self, run_id: str, operation: str,
                    args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.get_run(run_id) is None:
            raise KeyError(f"no persisted run with id '{run_id}'")
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO execution_jobs (run_id, tenant_id, operation, args_json, status, created_at) "
                "VALUES (?, ?, ?, ?, 'queued', ?)",
                (run_id, current_tenant(), operation,
                 json.dumps(args or {}, default=str), time.time()),
            )
            self._conn.commit()
            job_id = int(cursor.lastrowid)
        return self.get_job(job_id) or {}

    @staticmethod
    def _job_dict(row: Any) -> Dict[str, Any]:
        if row is None:
            return {}
        data = dict(row)
        data["args"] = json.loads(data.pop("args_json"))
        return data

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM execution_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_dict(row) if row is not None else None

    def list_jobs(self, run_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        query, params = "SELECT * FROM execution_jobs", []
        if run_id is not None:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [self._job_dict(row) for row in rows]

    def queued_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM execution_jobs WHERE status = 'queued' ORDER BY id").fetchall()
        return [self._job_dict(row) for row in rows]

    def start_job(self, job_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE execution_jobs SET status='running', started_at=? "
                "WHERE id=? AND status='queued'", (time.time(), job_id))
            self._conn.commit()
        return cursor.rowcount == 1

    def finish_job(self, job_id: int, error: Optional[str] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE execution_jobs SET status=?, error=?, finished_at=? WHERE id=?",
                ("failed" if error else "completed", error, time.time(), job_id))
            self._conn.commit()

    def recover_interrupted_jobs(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE execution_jobs SET status='queued', started_at=NULL "
                "WHERE status='running'")
            self._conn.commit()
        return cursor.rowcount

    def load_graph(self, run_id: str) -> Optional[DependencyGraph]:
        run = self.get_run(run_id)
        if not run or not run["graph"].get("steps"):
            return None
        return DependencyGraph.from_dict(run["graph"])

    # -- step results ------------------------------------------------------

    def save_step_result(self, run_id: str, result: StepResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO step_results (run_id, step_id, result_json, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(run_id, step_id) DO UPDATE SET "
                "result_json=excluded.result_json, updated_at=excluded.updated_at",
                (run_id, result.step_id, json.dumps(result.to_dict()), time.time()),
            )
            self._conn.commit()

    def load_results(self, run_id: str) -> Dict[str, StepResult]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT step_id, result_json FROM step_results WHERE run_id = ?", (run_id,)
            ).fetchall()
        return {r["step_id"]: StepResult.from_dict(json.loads(r["result_json"])) for r in rows}

    def clear_results(self, run_id: str, step_ids: Optional[List[str]] = None) -> None:
        with self._lock:
            if step_ids is None:
                self._conn.execute("DELETE FROM step_results WHERE run_id = ?", (run_id,))
            else:
                self._conn.executemany(
                    "DELETE FROM step_results WHERE run_id = ? AND step_id = ?",
                    [(run_id, sid) for sid in step_ids],
                )
            self._conn.commit()

    # -- actions awaiting human approval ---------------------------------

    def save_pending_action(self, run_id: str, action: PendingAction) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_actions "
                "(run_id, step_id, action_json, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id, step_id) DO UPDATE SET "
                "action_json=excluded.action_json, updated_at=excluded.updated_at",
                (run_id, action.step_id, json.dumps(action.to_storage_dict()), time.time()),
            )
            self._conn.commit()

    def load_pending_actions(self, run_id: str) -> Dict[str, PendingAction]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT step_id, action_json FROM pending_actions WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return {
            row["step_id"]: PendingAction.from_storage_dict(json.loads(row["action_json"]))
            for row in rows
        }

    def delete_pending_action(self, run_id: str, step_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM pending_actions WHERE run_id = ? AND step_id = ?",
                (run_id, step_id),
            )
            self._conn.commit()

    def clear_pending_actions(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM pending_actions WHERE run_id = ?", (run_id,))
            self._conn.commit()

    # -- events ------------------------------------------------------------

    def record_event(self, event: Event) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (run_id, ts, type, step_id, message, data_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (event.run_id, event.timestamp, event.type.value, event.step_id,
                 event.message, json.dumps(event.data, default=str)),
            )
            self._conn.commit()

    def load_events(self, run_id: str, limit: int = 1000,
                    after_id: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, run_id, ts, type, step_id, message, data_json FROM events "
                "WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?",
                (run_id, after_id, limit),
            ).fetchall()
        return [
            {
                "id": r["id"], "run_id": r["run_id"], "timestamp": r["ts"],
                "type": r["type"], "step_id": r["step_id"], "message": r["message"],
                "data": json.loads(r["data_json"]),
            }
            for r in rows
        ]

    def subscriber(self) -> Any:
        """An :class:`~orchestrator.events.EventBus` subscriber that persists events."""

        def persist(event: Event) -> None:
            if event.run_id:
                self.record_event(event)

        persist.__name__ = "state_manager_persist"
        return persist

    # -- recovery ----------------------------------------------------------

    def resumable(self, run_id: str) -> Tuple[List[str], List[str]]:
        """Split a persisted run into ``(completed, outstanding)`` step ids."""
        run = self.get_run(run_id)
        if not run:
            return [], []
        all_ids = [s["id"] for s in run["graph"].get("steps", [])]
        results = self.load_results(run_id)
        completed = [
            sid for sid in all_ids
            if sid in results and results[sid].status in {StepStatus.DONE, StepStatus.SKIPPED}
        ]
        return completed, [sid for sid in all_ids if sid not in completed]

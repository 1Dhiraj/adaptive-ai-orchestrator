"""Celery entry point for multi-machine workflow execution."""
from __future__ import annotations

import os
from typing import Any, Dict


def celery_app():
    try:
        from celery import Celery
    except ImportError as exc:
        raise RuntimeError("Celery is not installed; install production dependencies") from exc
    app = Celery(
        "adaptive_orchestrator",
        broker=os.environ.get("ORCHESTRATOR_BROKER_URL", "redis://localhost:6379/0"),
        backend=os.environ.get("ORCHESTRATOR_RESULT_BACKEND", "redis://localhost:6379/1"),
    )
    app.conf.update(task_acks_late=True, task_reject_on_worker_lost=True,
                    worker_prefetch_multiplier=1, broker_connection_retry_on_startup=True,
                    task_track_started=True, task_default_queue="workflows",
                    task_routes={"orchestrator.execute_job": {"queue": "workflows"}})
    return app


app = celery_app()


@app.task(name="orchestrator.execute_job", bind=True, autoretry_for=(ConnectionError,),
          retry_backoff=True, retry_jitter=True, max_retries=5)
def execute_job(self: Any, job_id: int) -> Dict[str, Any]:
    """Atomically claim one durable job and execute it."""
    from server.app import manager
    from orchestrator.events import EventType
    from orchestrator.tenancy import tenant_scope

    job = manager.state.get_job(int(job_id))
    if not job:
        return {"job_id": job_id, "status": "missing"}
    if not manager.state.start_job(int(job_id)):
        return {"job_id": job_id, "status": "already_claimed"}
    error = None
    workflow = None
    try:
        with tenant_scope(str(job.get("tenant_id") or "default")):
            from orchestrator.vault import CredentialVault
            if os.environ.get("ORCHESTRATOR_VAULT_KEY") or os.environ.get(
                    "ORCHESTRATOR_SECRET_BACKEND") == "aws":
                CredentialVault().activate()
            workflow = manager.get(str(job["run_id"]))
            operation = str(job["operation"])
            if operation not in {"run_full", "run", "resume", "handle_step_change",
                                 "handle_tool_failure", "handle_tool_repair"}:
                raise ValueError(f"unsupported operation: {operation}")
            getattr(workflow, operation)(**job.get("args", {}))
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        raise
    finally:
        manager.state.finish_job(int(job_id), error)
        if workflow:
            workflow.bus.publish(EventType.JOB_FINISHED, run_id=str(job["run_id"]),
                                 message=f"job {job_id} {'failed' if error else 'completed'}",
                                 job_id=job_id, status="failed" if error else "completed",
                                 error=error)
    return {"job_id": job_id, "status": "completed"}

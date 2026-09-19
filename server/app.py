"""FastAPI server: REST control plane + WebSocket live feed for the dashboard.

Run it with::

    python -m orchestrator.cli serve
    # or: uvicorn server.app:app --reload

Workflows execute on worker threads (the orchestrator is synchronous), and
their events are pushed onto per-connection asyncio queues, so the dashboard
updates as each step lands rather than at the end of the run.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
import threading
import base64
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, RedirectResponse
from pydantic import BaseModel, Field

from orchestrator import Step, StateManager, Workflow
import orchestrator.config as _config
from orchestrator.events import Event, EventType
from orchestrator.inputs import InputError
from orchestrator.export import export_csv, export_html, export_json
from server.security import SecurityPolicy
from server.observability import (CONTENT_TYPE_LATEST, HTTP_DURATION, HTTP_REQUESTS,
                                  configure_tracing, metrics_payload)
from orchestrator.tenancy import tenant_scope
from orchestrator.tenancy import current_tenant
from orchestrator.acquisition import AcquisitionError, CapabilityAcquirer, search_terms
from orchestrator.tools.adaptive_email import AdaptiveEmailTool, EmailDelivery, email_message, message_digest

STATIC_DIR = Path(__file__).parent / "static"
EXPORT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
security = SecurityPolicy()
_acquirers: Dict[str, CapabilityAcquirer] = {}
_acquirer_lock = threading.RLock()
_email_deliveries: Dict[str, EmailDelivery] = {}


def _capabilities() -> CapabilityAcquirer:
    root = _config.default_data_dir() / "capabilities"
    tenant = current_tenant()
    key = f"{root}:{tenant}"
    with _acquirer_lock:
        if key not in _acquirers:
            _acquirers[key] = CapabilityAcquirer(root, tenant)
        return _acquirers[key]


def _email_delivery() -> EmailDelivery:
    root = _capabilities().root / "email-delivery"
    key = str(root)
    with _acquirer_lock:
        if key not in _email_deliveries:
            _email_deliveries[key] = EmailDelivery(root)
        service = _email_deliveries[key]
        # A shared server browser must never expose one user's mailbox to
        # another tenant. Browser email is limited to the local workspace.
        local = current_tenant() == "default" and not any(os.environ.get(name) for name in (
            "ORCHESTRATOR_API_KEY", "ORCHESTRATOR_API_KEYS", "OIDC_ISSUER"))
        service.browser_tool = next((tool for tool in [*_shared_mcp_tools, *_capabilities().connected_tools()]
                                     if getattr(getattr(tool, "_mcp_tool", None), "name", "") == "browser_run_code"), None) if local else None
        return service


# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------


class RunManager:
    """Owns live Workflow objects and fans their events out to WebSockets."""

    def __init__(self) -> None:
        self.workflows: Dict[str, Workflow] = {}
        self.busy: Dict[str, bool] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.state = StateManager()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # -- registry ---------------------------------------------------------

    def register(self, workflow: Workflow) -> Workflow:
        key = f"{current_tenant()}:{workflow.run_id}"
        with self._lock:
            self.workflows[key] = workflow
            self.busy.setdefault(workflow.run_id, False)
        workflow.bus.subscribe(self._make_forwarder(workflow.run_id))
        return workflow

    def forget(self, run_id: str) -> None:
        with self._lock:
            self.workflows.pop(f"{current_tenant()}:{run_id}", None)
            self.busy.pop(run_id, None)

    def get(self, run_id: str) -> Workflow:
        with self._lock:
            workflow = self.workflows.get(f"{current_tenant()}:{run_id}")
        if workflow is not None:
            return workflow
        # Not in memory -- try to rehydrate it from SQLite.
        try:
            workflow = Workflow.resume_from(run_id, state=self.state, verbose=False, persist=True)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}") from None
        attach = globals().get("_attach_configured_extras")
        if attach is not None:
            attach(workflow)
        return self.register(workflow)

    # -- event fan-out ----------------------------------------------------

    def _make_forwarder(self, run_id: str):
        def forward(event: Event) -> None:
            payload = event.to_dict()
            loop = self._loop
            if loop is None:
                return
            with self._lock:
                queues = list(self._subscribers.get(run_id, []))
            for queue in queues:
                # Called from a worker thread, so hop back onto the loop.
                loop.call_soon_threadsafe(queue.put_nowait, payload)

        forward.__name__ = f"ws_forward_{run_id}"
        return forward

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._subscribers.get(run_id, []):
                self._subscribers[run_id].remove(queue)

    # -- execution --------------------------------------------------------

    def launch(self, run_id: str, fn_name: str, **kwargs: Any) -> None:
        """Durably queue a Workflow method, then start a local worker."""
        workflow = self.get(run_id)
        with self._lock:
            if self.busy.get(run_id):
                raise HTTPException(status_code=409, detail=f"run '{run_id}' is already executing")
            self.busy[run_id] = True
        job = self.state.enqueue_job(run_id, fn_name, kwargs)
        workflow.bus.publish(EventType.JOB_QUEUED, run_id=run_id,
                             message=f"queued {fn_name}", job_id=job["id"])
        if os.environ.get("ORCHESTRATOR_BROKER_URL"):
            try:
                from orchestrator.distributed import execute_job
                names = {(step.requires_tool or "").lower()
                         for step in workflow.graph.steps.values()}
                desktop = bool(names & {"hermes_desktop", "desktop_automation", "hermes"})
                browser = any("playwright" in name for name in names)
                execute_job.apply_async(args=[int(job["id"])],
                                        queue="desktop" if desktop else
                                              "browser" if browser else "workflows")
            except Exception as exc:
                with self._lock:
                    self.busy[run_id] = False
                raise HTTPException(status_code=503,
                                    detail=f"distributed queue unavailable: {exc}") from exc
            with self._lock:
                self.busy[run_id] = False
            return
        if os.environ.get("ORCHESTRATOR_QUEUE_ONLY", "").lower() in {"1", "true", "yes"}:
            with self._lock:
                self.busy[run_id] = False
            return
        self._start_job(workflow, job)

    def _start_job(self, workflow: Workflow, job: Dict[str, Any]) -> None:
        run_id, job_id = str(job["run_id"]), int(job["id"])
        operation = str(job["operation"])
        allowed = {"run_full", "run", "resume", "handle_step_change",
                   "handle_tool_failure", "handle_tool_repair"}
        if operation not in allowed or not self.state.start_job(job_id):
            with self._lock:
                self.busy[run_id] = False
            if operation not in allowed:
                self.state.finish_job(job_id, f"unsupported operation: {operation}")
            return

        def worker() -> None:
            error: Optional[str] = None
            try:
                getattr(workflow, operation)(**job.get("args", {}))
            except Exception as exc:  # noqa: BLE001 - surface it on the event feed
                error = str(exc)
                workflow.bus.publish(EventType.LOG, run_id=run_id,
                                     message=f"run failed: {exc}")
            finally:
                self.state.finish_job(job_id, error)
                workflow.bus.publish(
                    EventType.JOB_FINISHED, run_id=run_id,
                    message=f"job {job_id} {'failed' if error else 'completed'}",
                    job_id=job_id, status="failed" if error else "completed",
                    error=error)
                with self._lock:
                    self.busy[run_id] = False

        threading.Thread(target=worker, name=f"run-{run_id}", daemon=True).start()

    def launch_queued_jobs(self) -> int:
        """Resume durable jobs left queued by a previous process."""
        launched = 0
        for job in self.state.queued_jobs():
            run_id = str(job["run_id"])
            with self._lock:
                if self.busy.get(run_id, False):
                    continue
                self.busy[run_id] = True
            try:
                with tenant_scope(str(job.get("tenant_id") or "default")):
                    workflow = self.get(run_id)
            except HTTPException as exc:
                self.state.finish_job(int(job["id"]), str(exc.detail))
                with self._lock:
                    self.busy[run_id] = False
                continue
            self._start_job(workflow, job)
            launched += 1
        return launched

    def launch_due_schedules(self) -> int:
        """Start due recurring workflows, skipping any run already in progress."""
        launched = 0
        for schedule in self.state.due_schedules(all_tenants=True):
            run_id = str(schedule["run_id"])
            with self._lock:
                if self.busy.get(run_id, False):
                    continue
            with tenant_scope(str(schedule.get("tenant_id") or "default")):
                try:
                    self.launch(run_id, "run_full", fresh=True)
                except HTTPException:
                    continue
                self.state.advance_schedule(int(schedule["id"]))
                launched += 1
        return launched


manager = RunManager()

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ClarifyTask(BaseModel):
    description: str = Field(..., min_length=3)


class CreateRun(BaseModel):
    description: str = Field(..., min_length=3, description="Plain-English project description")
    run_id: Optional[str] = None
    #: Answers to the clarification questions, folded into the planning prompt.
    clarifications: Dict[str, Any] = Field(default_factory=dict)
    parallel: bool = True
    smart_invalidation: bool = True
    max_workers: Optional[int] = None
    #: Stop and ask for anything unmet rather than letting the step run with
    #: simulated tool output and report success it did not earn.
    ask_for_requirements: bool = True


class InputSpec(BaseModel):
    name: str
    prompt: str = ""
    type: str = "text"
    required: bool = True
    default: Optional[Any] = None
    options: List[str] = Field(default_factory=list)
    why: str = ""


class StepSpec(BaseModel):
    id: str
    description: str
    agent_role: str = "generic"
    name: str = ""
    requires_tool: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    requires_approval: bool = False
    inputs: List[InputSpec] = Field(default_factory=list)
    condition: Optional[Dict[str, Any]] = None


class CreateRunFromSteps(BaseModel):
    description: str = ""
    run_id: Optional[str] = None
    steps: List[StepSpec]
    parallel: bool = True
    smart_invalidation: bool = True
    action_approval: str = "live"


class ChangeStep(BaseModel):
    new_description: Optional[str] = None
    new_requirement: Optional[str] = None


class PrepareChange(BaseModel):
    """A plain-English requirement change, e.g. 'use PostgreSQL instead of MongoDB'."""

    request: str = Field(..., min_length=3)


class ApplyChange(BaseModel):
    """A proposal returned by /change/prepare, sent back verbatim to apply it.

    The proposal carries its own digest and the graph revision it was built
    against; the workflow rejects anything stale or edited, so a preview can
    be shown to a person and confirmed later without risk of it landing on a
    graph that moved underneath.
    """

    proposal: Dict[str, Any]
    confirmed: bool = False
    rerun: bool = True


class EditStep(BaseModel):
    """Every field is optional; only what is sent gets changed."""

    description: Optional[str] = None
    agent_role: Optional[str] = None
    requires_tool: Optional[str] = None
    depends_on: Optional[List[str]] = None
    name: Optional[str] = None
    requires_approval: Optional[bool] = None
    inputs: Optional[List[InputSpec]] = None
    condition: Optional[Dict[str, Any]] = None
    rerun: bool = False


class RunOptions(BaseModel):
    fresh: bool = False
    only: Optional[List[str]] = None


class CapabilitySearch(BaseModel):
    query: str = Field(..., min_length=2, max_length=80)
    kind: str = Field("mcp", pattern="^(mcp|skill)$")


class ConnectCapability(BaseModel):
    digest: str = Field(..., min_length=64, max_length=64)
    approved: bool = False


class BrowserEmailPermission(BaseModel):
    approved: bool = False
    digest: str = Field(..., min_length=64, max_length=64)


class ProvideInputs(BaseModel):
    values: Dict[str, Any] = Field(default_factory=dict,
                                   description="input name -> value, for one step")
    resume: bool = Field(True, description="continue the run once nothing is outstanding")


class RejectAction(BaseModel):
    reason: str = ""


class SaveCredential(BaseModel):
    env_var: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class CreateSchedule(BaseModel):
    interval_minutes: int = Field(..., ge=1, le=525600)


class UpdateSchedule(BaseModel):
    enabled: bool


class CreateWebhook(BaseModel):
    name: str = Field("Webhook", min_length=1, max_length=80)


class UpdateWebhook(BaseModel):
    enabled: bool


class SaveTemplate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class LaunchTemplate(BaseModel):
    run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Worker threads need a handle on the loop to push WebSocket events.
    manager.bind_loop(asyncio.get_running_loop())
    if os.environ.get("ORCHESTRATOR_SECRET_BACKEND") in {"keyring", "aws"} or \
            os.environ.get("ORCHESTRATOR_VAULT_KEY"):
        from orchestrator.vault import CredentialVault
        await asyncio.to_thread(CredentialVault().activate)
    await asyncio.to_thread(_connect_shared_mcp)
    manager.state.recover_interrupted_jobs()
    await asyncio.to_thread(manager.launch_queued_jobs)
    async def schedule_loop() -> None:
        while True:
            await asyncio.to_thread(manager.launch_due_schedules)
            await asyncio.sleep(5)

    scheduler = asyncio.create_task(schedule_loop())
    try:
        yield
    finally:
        scheduler.cancel()
        try:
            await scheduler
        except asyncio.CancelledError:
            pass
        if _shared_mcp_registry is not None:
            _shared_mcp_registry.close_all()
        for service in _acquirers.values():
            service.close()
        _acquirers.clear()
        _email_deliveries.clear()
        manager.state.close()


app = FastAPI(
    title="Adaptive AI Task Orchestrator",
    version="1.0.0",
    description="Plan, execute and adaptively re-run multi-agent workflows.",
    lifespan=lifespan,
)
configure_tracing(app)


@app.middleware("http")
async def record_http_metrics(request: Request, call_next: Any) -> Any:
    started = __import__("time").perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    route_name = getattr(route, "path", request.url.path)
    if HTTP_REQUESTS is not None:
        HTTP_REQUESTS.labels(request.method, route_name, str(response.status_code)).inc()
        HTTP_DURATION.labels(request.method, route_name).observe(
            __import__("time").perf_counter() - started)
    return response


def _api_key() -> str:
    return os.environ.get("ORCHESTRATOR_API_KEY", "").strip()


def _valid_api_key(value: str) -> bool:
    return security.role_for(value) is not None


def _auth_configured() -> bool:
    return bool(security.configured_keys())


def _request_key(request: Request) -> str:
    bearer = request.headers.get("authorization", "")
    if bearer.lower().startswith("bearer "):
        return bearer[7:].strip()
    return (request.headers.get("x-orchestrator-key", "").strip()
            or request.cookies.get("orchestrator_token", ""))


@app.middleware("http")
async def require_api_key(request: Request, call_next: Any) -> Any:
    """Protect the control plane when ORCHESTRATOR_API_KEY is configured.

    The dashboard shell and health check remain public so the browser can load
    and explain that a key is required. Every endpoint that reads workflow
    data or can trigger an action is protected.
    """
    if request.url.path.startswith("/hooks/"):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > 65_536:
            return JSONResponse(
                status_code=413, content={"detail": "Webhook payload is too large."})
    public_api = {"/api/health", "/api/auth/config", "/api/auth/login"}
    if request.url.path.startswith("/api/") and request.url.path not in public_api:
        identity = security.authenticate(_request_key(request))
        if identity is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "A valid orchestrator access key is required."},
                headers={"WWW-Authenticate": "Bearer"},
            )
        role = identity.role
        request.state.identity = identity
        required = security.required_role(request.method, request.url.path)
        if not security.allowed(role, required):
            return JSONResponse(status_code=403, content={
                "detail": f"This action requires the {required} role."})
        allowed, retry_after = security.check_rate(_request_key(request) or "local")
        if not allowed:
            return JSONResponse(status_code=429,
                                content={"detail": "Request limit exceeded."},
                                headers={"Retry-After": str(retry_after)})
        request.state.orchestrator_role = role
    identity = getattr(request.state, "identity", None)
    with tenant_scope(identity.tenant_id if identity else "default"):
        response = await call_next(request)
    role = getattr(request.state, "orchestrator_role", None)
    if role:
        response.headers["X-Orchestrator-Role"] = role
        if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            client = request.client.host if request.client else ""
            manager.state.record_audit(
                role, request.method.upper(), request.url.path, response.status_code, client)
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> Any:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Dashboard asset missing</h1>", status_code=500)
    return FileResponse(index)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "version": app.version,
            "api_auth_required": _auth_configured(),
            "role_access": ["viewer", "operator", "admin"],
            "config": _config.settings.describe()}


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(metrics_payload(manager.state), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/auth/me")
async def auth_me(request: Request) -> Dict[str, Any]:
    identity = request.state.identity
    return {"subject": identity.subject, "organization": identity.tenant_id,
            "role": identity.role, "provider": identity.provider}


@app.get("/api/auth/config")
async def auth_config() -> Dict[str, Any]:
    return {"oidc_enabled": bool(os.environ.get("OIDC_ISSUER")),
            "password_login": False}


@app.get("/api/auth/login")
async def oidc_login(request: Request) -> Response:
    authorization = os.environ.get("OIDC_AUTHORIZATION_ENDPOINT", "")
    client_id = os.environ.get("OIDC_CLIENT_ID", "")
    redirect_uri = os.environ.get(
        "OIDC_REDIRECT_URI", str(request.base_url).rstrip("/") + "/auth/callback")
    if not authorization or not client_id:
        raise HTTPException(status_code=503, detail="OIDC login is not configured")
    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    params = urllib.parse.urlencode({"response_type": "code", "client_id": client_id,
        "redirect_uri": redirect_uri, "scope": "openid profile email", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256"})
    response = RedirectResponse(f"{authorization}?{params}")
    response.set_cookie("oidc_state", f"{state}.{verifier}", httponly=True, secure=request.url.scheme == "https",
                        samesite="lax", max_age=600)
    return response


@app.get("/auth/callback")
async def oidc_callback(request: Request, code: str, state: str) -> Response:
    saved = request.cookies.get("oidc_state", "")
    saved_state, separator, verifier = saved.partition(".")
    if not separator or not secrets.compare_digest(saved_state, state):
        raise HTTPException(status_code=400, detail="OIDC state validation failed")
    token_endpoint = os.environ.get("OIDC_TOKEN_ENDPOINT", "")
    redirect_uri = os.environ.get(
        "OIDC_REDIRECT_URI", str(request.base_url).rstrip("/") + "/auth/callback")
    if not token_endpoint:
        raise HTTPException(status_code=503, detail="OIDC token endpoint is not configured")
    import requests as http_requests
    token_response = await asyncio.to_thread(http_requests.post, token_endpoint, data={
        "grant_type": "authorization_code", "code": code,
        "client_id": os.environ.get("OIDC_CLIENT_ID", ""),
        "client_secret": os.environ.get("OIDC_CLIENT_SECRET", ""),
        "redirect_uri": redirect_uri, "code_verifier": verifier}, timeout=20)
    if token_response.status_code >= 400:
        raise HTTPException(status_code=401, detail="OIDC code exchange failed")
    # OIDC access tokens are allowed to be opaque.  The ID token is the JWT
    # intended for identity verification, so prefer it when both are returned.
    token = token_response.json().get("id_token") or token_response.json().get("access_token")
    if not token or security.authenticate(token) is None:
        raise HTTPException(status_code=401, detail="OIDC returned an invalid token")
    response = RedirectResponse("/")
    response.delete_cookie("oidc_state")
    response.set_cookie("orchestrator_token", token, httponly=True,
                        secure=request.url.scheme == "https", samesite="lax", max_age=3600)
    return response


@app.post("/api/auth/logout")
async def logout() -> Response:
    response = JSONResponse({"logged_out": True})
    response.delete_cookie("orchestrator_token")
    return response


@app.get("/api/tools")
async def list_tools() -> List[dict]:
    from orchestrator.tools import default_tool_manager

    # Include the shared MCP tools, or the dashboard shows a tool list that
    # does not match what a run actually gets.
    manager_ = default_tool_manager()
    for tool in _shared_mcp_tools:
        manager_.register(tool)
    return manager_.describe()


@app.get("/api/templates")
async def list_templates() -> List[Dict[str, Any]]:
    return manager.state.list_templates()


@app.get("/api/audit")
async def audit_log(limit: int = 200) -> List[Dict[str, Any]]:
    return manager.state.load_audit(max(1, min(limit, 1000)))


@app.post("/api/runs/{run_id}/template", status_code=201)
async def save_run_as_template(run_id: str, body: SaveTemplate) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    return manager.state.save_template(body.name, workflow.description, workflow.graph)


@app.post("/api/templates/{template_id}/launch", status_code=201)
async def launch_template(template_id: int, body: LaunchTemplate) -> Dict[str, Any]:
    template = manager.state.get_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="template not found")
    from orchestrator.graph import DependencyGraph

    workflow = Workflow(
        graph=DependencyGraph.from_dict(template["graph"]),
        description=template["description"], run_id=body.run_id,
        state=manager.state, verbose=False)
    _attach_configured_extras(workflow)
    manager.register(workflow)
    return _snapshot(workflow)


@app.delete("/api/templates/{template_id}")
async def delete_template(template_id: int) -> Dict[str, Any]:
    manager.state.delete_template(template_id)
    return {"deleted": template_id}


@app.get("/api/runs")
async def list_runs() -> List[dict]:
    stored = manager.state.list_runs()
    prefix = f"{current_tenant()}:"
    live = {key[len(prefix):] for key in manager.workflows if key.startswith(prefix)}
    for record in stored:
        record["in_memory"] = record["run_id"] in live
        record["busy"] = manager.busy.get(record["run_id"], False)
    return stored



#: MCP servers are connected once at startup and shared. Spawning them per
#: run would mean a browser launch on every plan -- slow enough to look broken.
_shared_mcp_tools: List[Any] = []
_shared_mcp_registry: Any = None


def _connect_shared_mcp() -> None:
    """Connect the configured MCP servers once, for every run to reuse."""
    global _shared_mcp_registry
    if not Path(".mcp.json").exists():
        return
    from orchestrator.tools import ToolManager
    from orchestrator.tools.mcp import McpRegistry

    scratch = ToolManager()
    _shared_mcp_registry = McpRegistry()
    try:
        names = _shared_mcp_registry.attach(scratch, config_path=".mcp.json")
    except Exception as exc:  # noqa: BLE001 - never block server startup
        print(f"[mcp] could not connect: {exc}")
        return
    _shared_mcp_tools.extend(scratch.get(n) for n in names)
    print(f"[mcp] {len(names)} tool(s) ready: {', '.join(names[:6])}"
          + (" ..." if len(names) > 6 else ""))


def _attach_configured_extras(workflow: Workflow) -> None:
    """Attach connections.json / skills/ if the user has set them up.

    Convention over configuration: someone who has written a connections file
    expects those tools to exist, and having to call a separate endpoint first
    just means their first run fails with "not registered".
    """
    from pathlib import Path as _Path

    if _Path("connections.json").exists():
        workflow.attach_connections("connections.json")
    if _Path("skills").is_dir():
        workflow.attach_skills("skills")
    for tool in _shared_mcp_tools:
        workflow.tools.register(tool)
    _capabilities().attach(workflow.tools, workflow.skills)
    workflow.tools.register(AdaptiveEmailTool(_email_delivery(), workflow.run_id))
    mail_steps = [step for step in workflow.graph.steps.values() if step.requires_tool == "adaptive_email"]
    if mail_steps:
        from orchestrator.inputs import InputRequest
        recipients = set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", workflow.description.replace('\\@', '@')))
        for step in mail_steps:
            if len(recipients) == 1:
                recipient = next(iter(recipients))
                field = next((field for field in step.inputs if field.name == "recipient"), None)
                if field is None:
                    field = InputRequest(name="recipient", prompt="Who should receive this email?", type="email")
                    step.inputs.append(field)
                if not field.provided:
                    field.provide(recipient)
        mail_ids = {step.id for step in mail_steps}
        mail_credentials = {"SENDGRID_API_KEY", "EMAIL_FROM", "EMAIL_TO", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "email", "gmail"}
        workflow.requirements.requirements = [req for req in workflow.requirements.requirements
            if not (req.name in mail_credentials and set(req.needed_by).issubset(mail_ids))]
        if workflow.state is not None:
            workflow.state.update_run(workflow.run_id, graph=workflow.graph)


def _planning_tools():
    """The planner must see the same configured tools as execution."""
    from orchestrator.tools import default_tool_manager
    from orchestrator.connections import attach_connections
    tools = default_tool_manager()
    if Path("connections.json").exists():
        attach_connections(tools, config_path="connections.json")
    for tool in _shared_mcp_tools:
        tools.register(tool)
    _capabilities().attach(tools)
    tools.register(AdaptiveEmailTool(_email_delivery()))
    return tools


@app.get("/api/capabilities")
async def list_capabilities() -> Dict[str, Any]:
    return {"extensions": _capabilities().list()}


@app.post("/api/capabilities/search")
async def search_capabilities(body: CapabilitySearch) -> Dict[str, Any]:
    service = _capabilities()
    search = service.search_skills if body.kind == "skill" else service.search_tools
    return {"candidates": await asyncio.to_thread(search, body.query)}


@app.get("/api/capabilities/{candidate_id}/review")
async def review_capability(candidate_id: str) -> Dict[str, Any]:
    return await asyncio.to_thread(_capabilities().review, candidate_id)


@app.post("/api/capabilities/{candidate_id}/connect")
async def connect_capability(candidate_id: str, body: ConnectCapability) -> Dict[str, Any]:
    return await asyncio.to_thread(_capabilities().connect, candidate_id, body.digest, body.approved)


@app.post("/api/runs/{run_id}/discover")
async def discover_missing_capabilities(run_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    queries = list(dict.fromkeys(
        search_terms(req.name) for req in workflow.check_requirements().requirements
        if req.kind.value in {"tool", "mcp_server"} and req.status.value == "missing"
    ))[:3]
    service = _capabilities()
    candidates, errors = [], []
    for query in queries:
        if len(query) < 2:
            continue
        try:
            candidates.extend(await asyncio.to_thread(service.search_tools, query))
        except AcquisitionError as exc:
            errors.append(str(exc))
    return {"queries": queries, "candidates": list({c['id']: c for c in candidates}.values()),
            "errors": errors}


@app.post("/api/runs/clarify")
async def clarify_task(body: ClarifyTask) -> Dict[str, Any]:
    """Ask what the planner needs to know before it plans.

    An empty ``questions`` list means the task is already clear -- the client
    should go straight to planning rather than showing an empty form.
    """
    from orchestrator.llm import get_provider
    from orchestrator.planner import TaskPlanner
    from orchestrator.recall import annotate

    questions = await asyncio.to_thread(
        TaskPlanner(llm=get_provider()).clarify, body.description)
    # Previous answers are attached as suggestions, never applied silently --
    # the UI shows where each came from and lets the person change it.
    annotated = annotate(questions, body.description)
    return {"questions": annotated,
            "needs_clarification": bool(annotated),
            "has_remembered": any(q.get("remembered") is not None for q in annotated)}


@app.post("/api/runs", status_code=201)
async def create_run(body: CreateRun) -> Dict[str, Any]:
    tools = await asyncio.to_thread(_planning_tools)
    workflow = await asyncio.to_thread(
        Workflow.from_description,
        body.description,
        run_id=body.run_id,
        clarifications=body.clarifications,
        parallel=body.parallel,
        smart_invalidation=body.smart_invalidation,
        max_workers=body.max_workers,
        state=manager.state,
        verbose=False,
        tool_manager=tools,
    )
    _attach_configured_extras(workflow)
    if body.ask_for_requirements:
        # After the extras are attached, so tools registered above count as
        # satisfied rather than being asked about.
        workflow.ask_for_missing_requirements()
    manager.register(workflow)
    # Only remember answers the person actually confirmed for this plan.
    from orchestrator.recall import remember_all

    remember_all(body.clarifications, body.description)
    return _snapshot(workflow)


@app.post("/api/runs/{run_id}/replan", status_code=201)
async def replan_with_connected_tools(run_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    if manager.busy.get(run_id) or any(r.status.value != "pending" for r in workflow.results.values()):
        raise HTTPException(status_code=409, detail="This task already started. Use its step editor to change tools without repeating completed work.")
    result = await create_run(CreateRun(description=workflow.description))
    result["previous_plan"] = run_id
    return result


@app.post("/api/setup/forget")
async def forget_remembered() -> Dict[str, Any]:
    """Clear remembered answers so the next task starts from scratch."""
    from orchestrator.recall import forget_all

    forget_all()
    return {"forgotten": True}


@app.post("/api/runs/from-steps", status_code=201)
async def create_run_from_steps(body: CreateRunFromSteps) -> Dict[str, Any]:
    steps = [Step(**spec.model_dump()) for spec in body.steps]
    try:
        workflow = Workflow(
            steps, description=body.description, run_id=body.run_id,
            parallel=body.parallel, smart_invalidation=body.smart_invalidation,
            action_approval=body.action_approval,
            state=manager.state, verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - invalid graphs are a client error
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _attach_configured_extras(workflow)
    manager.register(workflow)
    return _snapshot(workflow)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    return _snapshot(manager.get(run_id))


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str) -> Dict[str, Any]:
    # Validate existence first. Deleting state under a live worker can make it
    # reappear partially when that worker persists its next event/result.
    manager.get(run_id)
    if manager.busy.get(run_id, False):
        raise HTTPException(
            status_code=409,
            detail="the workflow is running; stop it and wait for it to finish before deleting",
        )
    manager.forget(run_id)
    manager.state.delete_run(run_id)
    return {"deleted": run_id}


@app.post("/api/runs/{run_id}/run")
async def start_run(run_id: str, body: RunOptions = RunOptions()) -> Dict[str, Any]:
    if body.fresh:
        manager.launch(run_id, "run_full", fresh=True)
    else:
        manager.launch(run_id, "run", only=body.only)
    return {"started": True, "run_id": run_id}


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str) -> Dict[str, Any]:
    manager.launch(run_id, "resume")
    return {"started": True, "run_id": run_id}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    if not manager.busy.get(run_id, False):
        raise HTTPException(status_code=409, detail="the workflow is not currently running")
    accepted = workflow.request_cancel()
    return {"accepted": accepted, "run_id": run_id,
            "message": "active work will finish; later steps will not start"}


@app.get("/api/runs/{run_id}/impact/{step_id}")
async def impact(run_id: str, step_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    try:
        return workflow.impact_of(step_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/runs/{run_id}/steps/{step_id}")
async def edit_step(run_id: str, step_id: str, body: EditStep) -> Dict[str, Any]:
    """Edit a step in place. Rejects anything that would break the graph."""
    workflow = manager.get(run_id)
    try:
        return await asyncio.to_thread(
            workflow.update_step, step_id,
            **body.model_dump(exclude_unset=True, exclude={"rerun"}),
            rerun=body.rerun)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/runs/{run_id}/steps/{step_id}")
async def delete_step(run_id: str, step_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    try:
        return workflow.remove_step(step_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/steps", status_code=201)
async def add_step(run_id: str, body: StepSpec, rerun: bool = False) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    if body.id in workflow.graph:
        raise HTTPException(status_code=409, detail=f"step '{body.id}' already exists")
    try:
        data = body.model_dump()
        if not data.get("condition"):
            data["condition"] = None
        result = workflow.add_step(Step(**data), rerun=rerun)
    except Exception as exc:  # noqa: BLE001 - an invalid graph is a client error
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result if isinstance(result, dict) else result.to_dict()


@app.post("/api/runs/{run_id}/steps/{step_id}/change")
async def change_step(run_id: str, step_id: str, body: ChangeStep) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    if step_id not in workflow.graph:
        raise HTTPException(status_code=404, detail=f"no such step: {step_id}")
    manager.launch(run_id, "handle_step_change", step_id=step_id,
                   new_description=body.new_description,
                   new_requirement=body.new_requirement)
    return {"started": True, "step_id": step_id, "impact": workflow.impact_of(step_id)}


@app.post("/api/runs/{run_id}/change/prepare")
async def prepare_change(run_id: str, body: PrepareChange) -> Dict[str, Any]:
    """Translate a requirement change and show its impact. Changes nothing."""
    workflow = manager.get(run_id)
    try:
        return await asyncio.to_thread(workflow.prepare_change, body.request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/change/apply")
async def apply_change(run_id: str, body: ApplyChange) -> Dict[str, Any]:
    """Apply a previously previewed change, once a person has confirmed it."""
    workflow = manager.get(run_id)
    if not body.confirmed:
        raise HTTPException(
            status_code=400,
            detail="confirm the fact delta or plan diff before applying the change")
    try:
        await asyncio.to_thread(workflow.apply_change, body.proposal,
                                confirmed=True, rerun=body.rerun)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _snapshot(workflow)


@app.post("/api/runs/{run_id}/steps/{step_id}/inputs")
async def provide_step_inputs(run_id: str, step_id: str, body: ProvideInputs) -> Dict[str, Any]:
    """Answer the questions a step is waiting on."""
    workflow = manager.get(run_id)
    if step_id not in workflow.graph:
        raise HTTPException(status_code=404, detail=f"no such step: {step_id}")
    try:
        applied = workflow.provide_inputs(body.values, step_id=step_id)
    except (KeyError, InputError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    still_pending = [i.name for i in workflow.graph.get(step_id).pending_inputs]
    if body.resume and not still_pending:
        manager.launch(run_id, "resume")
    return {"applied": sorted(applied), "still_pending": still_pending}


@app.get("/api/setup/questions")
async def setup_questions_for_run(run_id: Optional[str] = None) -> Dict[str, Any]:
    """What the operator still needs to supply, with how-to-get guidance.

    Without ``run_id`` this reports global setup (is any AI provider
    configured at all); with one it adds whatever that specific workflow needs.
    """
    from orchestrator.requirements import RequirementsReport
    from orchestrator.setup import guide_for, setup_questions

    questions: List[Dict[str, Any]] = []

    # An AI key is the one thing every workflow needs to do real work.
    if not _config.settings.llm_is_live:
        for env_var in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            questions.append({
                **guide_for(env_var).to_dict(),
                "why_this_workflow": "Without an AI key the agents produce placeholder "
                                     "text instead of real work. Any one of these is enough.",
                "blocking": False, "group": "ai_provider",
            })

    report: Optional[RequirementsReport] = None
    if run_id:
        try:
            report = manager.get(run_id).check_requirements()
        except HTTPException:
            report = None
    if report is not None:
        questions.extend({**q, "group": "workflow"} for q in setup_questions(report))

    return {
        "ready": not questions,
        "llm_configured": _config.settings.llm_is_live,
        "llm_mode": _config.settings.describe()["llm_mode"],
        "questions": questions,
    }


@app.post("/api/setup/credentials")
async def save_setup_credential(body: SaveCredential) -> Dict[str, Any]:
    """Validate and persist one credential, then make it live immediately."""
    from orchestrator.setup import save_credential, guide_for

    try:
        if os.environ.get("ORCHESTRATOR_VAULT_KEY") or os.environ.get(
                "ORCHESTRATOR_SECRET_BACKEND") in {"aws", "keyring"}:
            error = guide_for(body.env_var).validate(body.value.strip())
            if error:
                raise ValueError(error)
            from orchestrator.vault import CredentialVault
            vault_name = (body.env_var if os.environ.get("ORCHESTRATOR_SECRET_BACKEND") in {"aws", "keyring"}
                          else f"{current_tenant()}:{body.env_var}")
            CredentialVault().set(vault_name, body.value.strip())
            os.environ[body.env_var] = body.value.strip()
            _config.reload_settings()
        else:
            save_credential(body.env_var, body.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Rebuild the provider so a newly-added AI key takes effect at once
    # instead of on the next server restart.
    from orchestrator.llm import set_provider

    set_provider(None)
    return {"saved": body.env_var, "llm_mode": _config.settings.describe()["llm_mode"]}


@app.get("/api/runs/{run_id}/inputs")
async def list_pending_inputs(run_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    return {
        step_id: [request.to_dict() for request in requests]
        for step_id, requests in workflow.pending_inputs().items()
    }


@app.get("/api/runs/{run_id}/actions")
async def list_pending_actions(run_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    return {sid: action.to_dict() for sid, action in workflow.pending_actions().items()}


@app.post("/api/runs/{run_id}/actions/{step_id}/approve")
async def approve_action(run_id: str, step_id: str, resume: bool = True,
                         body: Optional[BrowserEmailPermission] = None) -> Dict[str, Any]:
    """Authorise a held irreversible action, then perform it."""
    workflow = manager.get(run_id)
    action = workflow.pending_actions().get(step_id)
    if action and action.tool == "adaptive_email":
        details = _email_details(workflow, step_id)
        if not body or not body.approved or body.digest != details["digest"]:
            raise HTTPException(status_code=400, detail="Review and approve the exact email before sending")
        if details.get("method") == "browser":
            ready = details["status"] == "draft_ready"
        else:
            ready = details["api_available"]
        if not ready:
            raise HTTPException(status_code=409, detail="Configure email credentials or prepare a Gmail browser draft first")
    try:
        workflow.approve_action(step_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if resume:
        manager.launch(run_id, "resume")
    return {"approved": step_id}


def _email_details(workflow: Workflow, step_id: str) -> dict:
    action = workflow.pending_actions().get(step_id)
    if action is None or action.tool != "adaptive_email":
        raise HTTPException(status_code=404, detail="No email draft is awaiting review")
    step = workflow.graph.get(step_id)
    message = email_message(action.payload, {"inputs": {**workflow.all_input_values(), **step.input_values()}})
    return _email_delivery().status(workflow.run_id, step_id, message)


@app.post("/api/runs/{run_id}/email/{step_id}/open-browser")
async def open_gmail_for_email(run_id: str, step_id: str, body: BrowserEmailPermission) -> dict:
    workflow = manager.get(run_id)
    details = _email_details(workflow, step_id)
    if body.digest != details["digest"] or not body.approved:
        raise HTTPException(status_code=400, detail="Approve browser access for this email first")
    if manager.busy.get(run_id):
        raise HTTPException(status_code=409, detail="Wait until the task pauses before opening Gmail")
    result = await asyncio.to_thread(_email_delivery().open_browser, run_id, step_id, details["message"], body.approved)
    return result


@app.post("/api/runs/{run_id}/email/{step_id}/prepare-browser")
async def prepare_gmail_draft(run_id: str, step_id: str) -> dict:
    workflow = manager.get(run_id)
    details = _email_details(workflow, step_id)
    return await asyncio.to_thread(_email_delivery().prepare_browser, run_id, step_id, details["message"])


@app.post("/api/runs/{run_id}/actions/{step_id}/reject")
async def reject_action(run_id: str, step_id: str, body: RejectAction = RejectAction()) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    try:
        workflow.reject_action(step_id, reason=body.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"rejected": step_id}


@app.post("/api/runs/{run_id}/steps/{step_id}/approve")
async def approve_step(run_id: str, step_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    try:
        workflow.approve(step_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"approved": step_id}


@app.post("/api/runs/{run_id}/steps/{step_id}/pause")
async def pause_step(run_id: str, step_id: str) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    try:
        workflow.pause_before(step_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"paused_before": step_id}


@app.post("/api/runs/{run_id}/tools/{tool}/break")
async def break_tool(run_id: str, tool: str, rerun: bool = True) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    if tool not in workflow.tools:
        raise HTTPException(status_code=404, detail=f"no such tool: {tool}")
    if rerun:
        manager.launch(run_id, "handle_tool_failure", tool=tool, rerun=True)
    else:
        workflow.handle_tool_failure(tool, rerun=False)
    return {"broken": tool, "rerun": rerun}


@app.post("/api/runs/{run_id}/tools/{tool}/repair")
async def repair_tool(run_id: str, tool: str, rerun: bool = False) -> Dict[str, Any]:
    workflow = manager.get(run_id)
    if tool not in workflow.tools:
        raise HTTPException(status_code=404, detail=f"no such tool: {tool}")
    if rerun:
        manager.launch(run_id, "handle_tool_repair", tool=tool, rerun=True)
    else:
        workflow.handle_tool_repair(tool, rerun=False)
    return {"repaired": tool, "rerun": rerun}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, after_id: int = 0, limit: int = 500) -> List[dict]:
    return manager.state.load_events(run_id, limit=limit, after_id=after_id)


@app.get("/api/runs/{run_id}/jobs")
async def run_jobs(run_id: str) -> List[Dict[str, Any]]:
    manager.get(run_id)
    return manager.state.list_jobs(run_id)


@app.get("/api/runs/{run_id}/schedules")
async def list_run_schedules(run_id: str) -> List[Dict[str, Any]]:
    manager.get(run_id)
    return manager.state.list_schedules(run_id)


@app.post("/api/runs/{run_id}/schedules", status_code=201)
async def create_run_schedule(run_id: str, body: CreateSchedule) -> Dict[str, Any]:
    manager.get(run_id)
    return manager.state.create_schedule(run_id, body.interval_minutes * 60)


@app.patch("/api/runs/{run_id}/schedules/{schedule_id}")
async def update_run_schedule(run_id: str, schedule_id: int,
                              body: UpdateSchedule) -> Dict[str, Any]:
    manager.get(run_id)
    schedule = manager.state.get_schedule(schedule_id)
    if schedule is None or schedule["run_id"] != run_id:
        raise HTTPException(status_code=404, detail="no such schedule for this workflow")
    manager.state.set_schedule_enabled(schedule_id, body.enabled)
    return manager.state.get_schedule(schedule_id) or {}


@app.delete("/api/runs/{run_id}/schedules/{schedule_id}")
async def delete_run_schedule(run_id: str, schedule_id: int) -> Dict[str, Any]:
    manager.get(run_id)
    schedule = manager.state.get_schedule(schedule_id)
    if schedule is None or schedule["run_id"] != run_id:
        raise HTTPException(status_code=404, detail="no such schedule for this workflow")
    manager.state.delete_schedule(schedule_id)
    return {"deleted": schedule_id}


def _public_webhook(record: Dict[str, Any]) -> Dict[str, Any]:
    """Never expose the stored token digest through the control plane."""
    return {key: value for key, value in record.items() if key != "token_hash"}


@app.get("/api/runs/{run_id}/webhooks")
async def list_run_webhooks(run_id: str) -> List[Dict[str, Any]]:
    manager.get(run_id)
    return manager.state.list_webhooks(run_id)


@app.post("/api/runs/{run_id}/webhooks", status_code=201)
async def create_run_webhook(run_id: str, body: CreateWebhook,
                             request: Request) -> Dict[str, Any]:
    manager.get(run_id)
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode()).hexdigest()
    public = _public_webhook(manager.state.create_webhook(run_id, body.name, digest))
    # The secret appears once. Only its SHA-256 digest is stored.
    public["url"] = str(request.base_url).rstrip("/") + "/hooks/" + token
    return public


@app.patch("/api/runs/{run_id}/webhooks/{webhook_id}")
async def update_run_webhook(run_id: str, webhook_id: int,
                             body: UpdateWebhook) -> Dict[str, Any]:
    manager.get(run_id)
    webhook = manager.state.get_webhook(webhook_id)
    if webhook is None or webhook["run_id"] != run_id:
        raise HTTPException(status_code=404, detail="no such webhook for this workflow")
    manager.state.set_webhook_enabled(webhook_id, body.enabled)
    return _public_webhook(manager.state.get_webhook(webhook_id) or {})


@app.delete("/api/runs/{run_id}/webhooks/{webhook_id}")
async def delete_run_webhook(run_id: str, webhook_id: int) -> Dict[str, Any]:
    manager.get(run_id)
    webhook = manager.state.get_webhook(webhook_id)
    if webhook is None or webhook["run_id"] != run_id:
        raise HTTPException(status_code=404, detail="no such webhook for this workflow")
    manager.state.delete_webhook(webhook_id)
    return {"deleted": webhook_id}


@app.post("/hooks/{token}", status_code=202)
async def trigger_webhook(token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    digest = hashlib.sha256(token.encode()).hexdigest()
    webhook = manager.state.find_webhook(digest)
    if webhook is None or not webhook["enabled"]:
        raise HTTPException(status_code=404, detail="webhook not found")
    run_id = str(webhook["run_id"])
    with tenant_scope(str(webhook.get("tenant_id") or "default")):
        manager.launch(run_id, "run_full", fresh=True, runtime_inputs=payload)
        manager.state.mark_webhook_triggered(int(webhook["id"]))
    return {"accepted": True, "run_id": run_id}


@app.get("/api/runs/{run_id}/export")
async def export_run(run_id: str, format: str = "json") -> Any:
    workflow = manager.get(run_id)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    exporters = {"json": export_json, "html": export_html, "csv": export_csv}
    if format not in exporters:
        raise HTTPException(status_code=400, detail=f"unknown format: {format}")
    path = EXPORT_DIR / f"{run_id}.{format}"
    exporters[format](workflow, str(path))
    media = {"json": "application/json", "html": "text/html", "csv": "text/csv"}[format]
    return FileResponse(path, media_type=media, filename=path.name)


def _workspace_files(workflow: Workflow) -> List[Dict[str, Any]]:
    """List files produced inside this run's confined coding workspace."""
    tool = workflow.tools.get("workspace")
    root = getattr(tool, "workspace", None)
    if root is None or not Path(root).is_dir():
        return []
    root = Path(root).resolve()
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
        })
    return files


@app.get("/api/runs/{run_id}/files")
async def list_run_files(run_id: str) -> List[Dict[str, Any]]:
    return _workspace_files(manager.get(run_id))


@app.get("/api/runs/{run_id}/files/{relative_path:path}")
async def download_run_file(run_id: str, relative_path: str) -> Any:
    workflow = manager.get(run_id)
    tool = workflow.tools.get("workspace")
    root = getattr(tool, "workspace", None)
    if root is None:
        raise HTTPException(status_code=404, detail="this workflow has no coding workspace")
    root = Path(root).resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="file path leaves the workflow workspace")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="no such workspace file")
    return FileResponse(target, filename=target.name)


@app.websocket("/ws/{run_id}")
async def websocket_feed(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    if _auth_configured():
        try:
            auth = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        except Exception:
            await websocket.close(code=4401, reason="authentication required")
            return
        role = security.role_for(str(auth.get("token", "")))
        if auth.get("type") != "auth" or role is None or not security.allowed(role, "viewer"):
            await websocket.send_json({"type": "error", "message": "Invalid access key."})
            await websocket.close(code=4401, reason="invalid access key")
            return
    queue = manager.subscribe(run_id)
    try:
        # Send the current state immediately so a late joiner is not blank.
        try:
            await websocket.send_json({"type": "snapshot", "data": _snapshot(manager.get(run_id))})
        except HTTPException:
            await websocket.send_json({"type": "error", "message": f"no such run: {run_id}"})

        while True:
            event = await queue.get()
            await websocket.send_json({"type": "event", "data": event})
            # After anything that changes step state, push a fresh snapshot so
            # the client never has to reconstruct state from the event stream.
            if event["type"] in {
                "step_finished", "step_failed", "step_skipped", "step_cancelled",
                "run_finished", "step_awaiting_approval", "tool_broken", "tool_repaired",
                "run_cancel_requested",
                "step_awaiting_input", "step_input_provided",
                "action_awaiting_approval", "action_approved",
                "action_rejected", "action_executed",
                "job_queued", "job_finished", "agent_message",
            }:
                await websocket.send_json({"type": "snapshot", "data": _snapshot(manager.get(run_id))})
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - never let a dead socket kill the server
        pass
    finally:
        manager.unsubscribe(run_id, queue)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(workflow: Workflow) -> Dict[str, Any]:
    """Compact state payload for the dashboard."""
    return {
        "run_id": workflow.run_id,
        "description": workflow.description,
        "busy": manager.busy.get(workflow.run_id, False),
        "cancellation_requested": workflow.cancellation_requested,
        "levels": workflow.graph.execution_levels(),
        "steps": [
            {
                **workflow.graph.get(sid).to_dict(),
                "result": workflow.results[sid].to_dict(),
            }
            for sid in workflow.graph.topological_order()
        ],
        "metrics": workflow.metrics(),
        "tools": workflow.tools.describe(),
        "awaiting_approval": workflow.awaiting_approval(),
        "requirements": workflow.check_requirements().to_dict(),
        "blocked_on_human": workflow.blocked_on_human(),
        # The full form, not just the blocking subset, so optional fields and
        # defaults can still be edited in the UI.
        "pending_inputs": {
            step_id: [r.to_dict() for r in requests]
            for step_id, requests in workflow.input_form().items()
        },
        "pending_actions": {sid: {**a.to_dict(), **({"email": _email_details(workflow, sid)}
                            if a.tool == "adaptive_email" else {})}
                            for sid, a in workflow.pending_actions().items()},
        "workspace_files": _workspace_files(workflow),
        "schedules": manager.state.list_schedules(workflow.run_id),
        "webhooks": manager.state.list_webhooks(workflow.run_id),
        "agent_messages": workflow.agent_messages,
        "jobs": manager.state.list_jobs(workflow.run_id, limit=20),
    }


@app.exception_handler(KeyError)
async def _key_error_handler(request: Any, exc: KeyError) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(AcquisitionError)
async def _acquisition_error_handler(request: Any, exc: AcquisitionError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})

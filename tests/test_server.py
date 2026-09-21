"""FastAPI control plane and the WebSocket feed."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def client(tmp_path, monkeypatch, stub_llm):
    """A TestClient wired to a throwaway database and the stub LLM."""
    from orchestrator.state import StateManager
    import server.app as server_app

    server_app.manager.state = StateManager(str(tmp_path / "server.db"))
    server_app.manager.workflows.clear()
    server_app.manager.busy.clear()
    monkeypatch.setattr(server_app, "EXPORT_DIR", tmp_path / "exports")

    with TestClient(server_app.app) as test_client:
        yield test_client

    server_app.manager.state.close()


def wait_until_idle(client, run_id: str, timeout: float = 20.0,
                    headers: dict | None = None) -> dict:
    """Poll until the background run thread finishes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = client.get(f"/api/runs/{run_id}", headers=headers or {}).json()
        if not snapshot["busy"]:
            return snapshot
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} never finished")


class TestBasics:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["ok"] and "llm_mode" in body["config"]

    def test_optional_api_key_protects_control_plane(self, client, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_API_KEY", "test-control-key")
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/health").json()["api_auth_required"] is True
        denied = client.get("/api/runs")
        assert denied.status_code == 401
        assert denied.headers["www-authenticate"] == "Bearer"
        allowed = client.get(
            "/api/runs", headers={"Authorization": "Bearer test-control-key"})
        assert allowed.status_code == 200

    def test_api_key_alternate_header(self, client, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_API_KEY", "test-control-key")
        response = client.get(
            "/api/tools", headers={"X-Orchestrator-Key": "test-control-key"})
        assert response.status_code == 200

    def test_role_keys_limit_mutations(self, client, monkeypatch):
        import json
        monkeypatch.setenv("ORCHESTRATOR_API_KEYS", json.dumps({
            "read-key": "viewer", "work-key": "operator", "root-key": "admin"}))
        viewer = {"Authorization": "Bearer read-key"}
        operator = {"Authorization": "Bearer work-key"}
        assert client.get("/api/runs", headers=viewer).status_code == 200
        assert client.post("/api/runs/from-steps", headers=viewer, json={
            "steps": [{"id": "a", "description": "x", "agent_role": "writer"}]
        }).status_code == 403
        created = client.post("/api/runs/from-steps", headers=operator, json={
            "steps": [{"id": "a", "description": "x", "agent_role": "writer"}]
        })
        assert created.status_code == 201
        assert created.headers["x-orchestrator-role"] == "operator"

    def test_organizations_cannot_see_each_others_runs(self, client, monkeypatch):
        import json
        monkeypatch.setenv("ORCHESTRATOR_API_KEYS", json.dumps({
            "alpha-key": {"role": "operator", "organization": "alpha", "user": "a"},
            "beta-key": {"role": "operator", "organization": "beta", "user": "b"},
        }))
        alpha = {"Authorization": "Bearer alpha-key"}
        beta = {"Authorization": "Bearer beta-key"}
        body = {"run_id": "alpha-private", "description": "private", "steps": [
            {"id": "a", "description": "x", "agent_role": "writer"}]}
        assert client.post("/api/runs/from-steps", headers=alpha, json=body).status_code == 201
        assert [r["run_id"] for r in client.get("/api/runs", headers=alpha).json()] == ["alpha-private"]
        assert client.get("/api/runs", headers=beta).json() == []
        assert client.get("/api/runs/alpha-private", headers=beta).status_code == 404

    def test_auth_me_reports_verified_key_identity(self, client, monkeypatch):
        import json
        monkeypatch.setenv("ORCHESTRATOR_API_KEYS", json.dumps({
            "team-key": {"role": "admin", "organization": "vcet", "user": "dhiraj"}}))
        response = client.get("/api/auth/me", headers={"Authorization": "Bearer team-key"})
        assert response.status_code == 200
        assert response.json() == {"subject": "dhiraj", "organization": "vcet",
                                   "role": "admin", "provider": "api_key"}

    def test_oidc_callback_prefers_identity_token(self, client, monkeypatch):
        """Opaque OAuth access tokens must not hide the signed OIDC ID token."""
        import requests
        import server.app as server_app

        class TokenResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"access_token": "opaque-access-token", "id_token": "signed-id-token"}

        seen = []
        monkeypatch.setenv("OIDC_TOKEN_ENDPOINT", "https://identity.example/token")
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: TokenResponse())
        monkeypatch.setattr(server_app.security, "authenticate",
                            lambda token: seen.append(token) or object())
        client.cookies.set("oidc_state", "expected.verifier")
        response = client.get("/auth/callback?code=code&state=expected",
                              follow_redirects=False)
        assert response.status_code == 307
        assert seen == ["signed-id-token"]
        assert "orchestrator_token=signed-id-token" in response.headers["set-cookie"]

    def test_configurable_rate_limit_returns_retry_after(self, client, monkeypatch):
        import server.app as server_app
        monkeypatch.setenv("ORCHESTRATOR_API_KEY", "limited")
        monkeypatch.setenv("ORCHESTRATOR_RATE_LIMIT_PER_MINUTE", "1")
        server_app.security.reset()
        headers = {"Authorization": "Bearer limited"}
        assert client.get("/api/runs", headers=headers).status_code == 200
        limited = client.get("/api/runs", headers=headers)
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) >= 1
        server_app.security.reset()

    def test_only_admin_can_read_durable_audit_log(self, client, monkeypatch):
        import json
        monkeypatch.setenv("ORCHESTRATOR_API_KEYS", json.dumps({
            "work": "operator", "root": "admin"}))
        operator = {"Authorization": "Bearer work"}
        admin = {"Authorization": "Bearer root"}
        assert client.post("/api/runs/from-steps", headers=operator, json={
            "steps": [{"id": "a", "description": "x", "agent_role": "writer"}]
        }).status_code == 201
        assert client.get("/api/audit", headers=operator).status_code == 403
        audit = client.get("/api/audit", headers=admin)
        assert audit.status_code == 200
        assert audit.json()[0]["role"] == "operator"

    def test_dashboard_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Adaptive AI Task Orchestrator" in response.text
        assert 'id="artifact-viewer"' in response.text
        assert "viewWorkspaceFile" in response.text
        assert "Stored at workspace/" in response.text
        assert "Task chat" in response.text
        assert "renderTaskSetupQuestions" in response.text
        assert "Open Gmail" in response.text
        assert "Prepare Gmail draft" in response.text
        assert "const message = email.message" in response.text
        assert 'id="change-requirement-dialog"' in response.text
        assert "openRequirementChange" in response.text
        assert "window.prompt" not in response.text
        assert "prompt(" not in response.text
        assert 'id="confirmation-dialog"' in response.text
        assert "requestConfirmation" in response.text
        assert "window.confirm" not in response.text
        assert "confirm(" not in response.text
        assert 'id="btn-delete-task"' in response.text
        assert "Delete this task?" in response.text
        assert 'method: "DELETE"' in response.text

    def test_invalid_ai_plan_returns_retryable_error_without_creating_run(
            self, client, monkeypatch):
        import server.app as server_app
        from orchestrator.planner import PlanningError

        def fail_planning(*args, **kwargs):
            raise PlanningError("The AI planner did not return a valid task plan")

        before = len(client.get("/api/runs").json())
        monkeypatch.setattr(server_app.Workflow, "from_description", fail_planning)
        response = client.post("/api/runs", json={"description": "Build a CLI"})

        assert response.status_code == 502
        assert "AI planner" in response.json()["detail"]
        assert len(client.get("/api/runs").json()) == before

    def test_playwright_unsafe_code_tool_is_available_for_browser_email(self):
        import server.app as server_app

        class RawTool:
            name = "browser_run_code_unsafe"

        class WrappedTool:
            name = "web_browser_run_code_unsafe"
            _mcp_tool = RawTool()

        tool = WrappedTool()
        assert server_app._find_browser_code_tool([tool]) is tool

    def test_browser_email_failure_is_a_recoverable_conflict(self, client, monkeypatch):
        import server.app as server_app
        from orchestrator.tools.base import ToolError

        digest = "a" * 64
        workflow = object()
        monkeypatch.setattr(server_app.manager, "get", lambda run_id: workflow)
        monkeypatch.setattr(server_app, "_email_details", lambda current, step_id: {
            "digest": digest, "message": {
                "to": "friend@example.com", "subject": "Hello", "body": "Hi",
            },
        })

        class BrokenBrowser:
            def open_browser(self, *args, **kwargs):
                raise ToolError("Gmail browser connection was lost; nothing was sent")

        monkeypatch.setattr(server_app, "_email_delivery", lambda: BrokenBrowser())
        response = client.post("/api/runs/mail/email/send/open-browser", json={
            "approved": True, "digest": digest,
        })

        assert response.status_code == 409
        assert response.json()["detail"] == "Gmail browser connection was lost; nothing was sent"

    def test_already_sent_email_can_finish_without_sending_again(self, client, monkeypatch):
        import server.app as server_app

        approved = []
        digest = "b" * 64

        class WorkflowStub:
            def pending_actions(self):
                return {"send": SimpleNamespace(tool="adaptive_email")}

            def approve_action(self, step_id):
                approved.append(step_id)

        monkeypatch.setattr(server_app.manager, "get", lambda run_id: WorkflowStub())
        monkeypatch.setattr(server_app.manager, "launch", lambda *args, **kwargs: None)
        monkeypatch.setattr(server_app, "_email_details", lambda workflow, step_id: {
            "browser_available": True,
            "method": "browser",
            "status": "sent",
            "digest": digest,
            "reviewed_digest": digest,
            "api_available": False,
        })

        response = client.post("/api/runs/mail/actions/send/approve", json={
            "approved": True, "digest": digest,
        })

        assert response.status_code == 200
        assert approved == ["send"]

    def test_tools_are_listed(self, client):
        names = {tool["name"] for tool in client.get("/api/tools").json()}
        assert "github" in names and "postgres" in names

    def test_runs_start_empty(self, client):
        assert client.get("/api/runs").json() == []


class TestRunLifecycle:
    def test_create_from_description(self, client):
        response = client.post("/api/runs", json={"description": "Build a bookstore API"})
        assert response.status_code == 201
        body = response.json()
        assert body["run_id"] and len(body["steps"]) >= 3
        assert body["levels"]

    def test_short_description_is_rejected(self, client):
        assert client.post("/api/runs", json={"description": "x"}).status_code == 422

    def test_create_from_explicit_steps(self, client):
        response = client.post("/api/runs/from-steps", json={
            "description": "manual",
            "run_id": "manual-1",
            "steps": [
                {"id": "a", "description": "Do a", "agent_role": "backend"},
                {"id": "b", "description": "Do b", "agent_role": "testing", "depends_on": ["a"]},
            ],
        })
        assert response.status_code == 201
        assert [s["id"] for s in response.json()["steps"]] == ["a", "b"]

    def test_invalid_graph_is_a_client_error(self, client):
        response = client.post("/api/runs/from-steps", json={
            "steps": [
                {"id": "a", "description": "x", "agent_role": "backend", "depends_on": ["b"]},
                {"id": "b", "description": "y", "agent_role": "backend", "depends_on": ["a"]},
            ],
        })
        assert response.status_code == 400

    def test_run_executes_every_step(self, client):
        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "run-1",
            "steps": [
                {"id": "a", "description": "Do a", "agent_role": "backend"},
                {"id": "b", "description": "Do b", "agent_role": "testing", "depends_on": ["a"]},
            ],
        }).json()["run_id"]

        assert client.post(f"/api/runs/{run_id}/run", json={"fresh": True}).status_code == 200
        snapshot = wait_until_idle(client, run_id)
        assert all(s["result"]["status"] == "done" for s in snapshot["steps"])
        assert snapshot["metrics"]["usage"]["calls"] == 2
        jobs = client.get(f"/api/runs/{run_id}/jobs").json()
        assert jobs[0]["operation"] == "run_full"
        assert jobs[0]["status"] == "completed"

    def test_agent_workspace_files_can_be_listed_and_downloaded(self, client):
        run_id = client.post(
            "/api/runs", json={"description": "Build a web app for college events"}
        ).json()["run_id"]
        workflow = __import__("server.app", fromlist=["manager"]).manager.get(run_id)
        workspace = workflow.tools.get("workspace")
        workspace.execute(
            'TOOL_DIRECTIVE: {"action":"write","path":"src/app.py",'
            '"content":"print(42)"}')

        listed = client.get(f"/api/runs/{run_id}/files")
        assert listed.status_code == 200
        assert listed.json() == [{"path": "src/app.py", "size": 9}]
        downloaded = client.get(f"/api/runs/{run_id}/files/src/app.py")
        assert downloaded.status_code == 200
        assert downloaded.content == b"print(42)"
        assert client.get(f"/api/runs/{run_id}/files/../.env.local").status_code in {400, 404}

    def test_artifact_store_files_are_visible_and_downloadable(self, client):
        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "artifact-files",
            "steps": [{"id": "report", "description": "Write a report",
                       "agent_role": "writer", "requires_tool": "artifact_store"}],
        }).json()["run_id"]
        workflow = __import__("server.app", fromlist=["manager"]).manager.get(run_id)
        workflow.tools.get("artifact_store").execute(
            "report body", {"step_id": "report"})

        listed = client.get(f"/api/runs/{run_id}/files")
        assert listed.status_code == 200
        artifacts = [entry for entry in listed.json()
                     if entry["path"].startswith("artifacts/report-")]
        assert artifacts
        artifact = artifacts[-1]
        downloaded = client.get(f"/api/runs/{run_id}/files/{artifact['path']}")
        assert downloaded.status_code == 200
        assert downloaded.content == b"report body"

    def test_recurring_schedule_can_be_created_paused_and_deleted(self, client):
        run_id = client.post(
            "/api/runs/from-steps",
            json={"run_id": "scheduled-1", "steps": [
                {"id": "a", "description": "Do a", "agent_role": "writer"}
            ]},
        ).json()["run_id"]
        created = client.post(
            f"/api/runs/{run_id}/schedules", json={"interval_minutes": 15})
        assert created.status_code == 201
        schedule_id = created.json()["id"]
        assert client.get(f"/api/runs/{run_id}/schedules").json()[0]["enabled"] == 1
        paused = client.patch(
            f"/api/runs/{run_id}/schedules/{schedule_id}", json={"enabled": False})
        assert paused.status_code == 200 and paused.json()["enabled"] == 0
        assert client.delete(
            f"/api/runs/{run_id}/schedules/{schedule_id}").status_code == 200
        assert client.get(f"/api/runs/{run_id}/schedules").json() == []

    def test_due_schedule_starts_a_fresh_run_and_moves_the_next_time(self, client,
                                                                      stub_llm):
        import server.app as server_app

        run_id = client.post(
            "/api/runs/from-steps",
            json={"run_id": "scheduled-due", "steps": [
                {"id": "collect", "description": "Collect current data",
                 "agent_role": "research"}
            ]},
        ).json()["run_id"]
        schedule = server_app.manager.state.create_schedule(
            run_id, interval_seconds=60, next_run_at=0)

        assert server_app.manager.launch_due_schedules() == 1
        snapshot = wait_until_idle(client, run_id)
        updated = server_app.manager.state.get_schedule(schedule["id"])

        assert snapshot["steps"][0]["result"]["status"] == "done"
        assert stub_llm.total_usage.calls == 1
        assert updated is not None
        assert updated["last_run_at"] is not None
        assert updated["next_run_at"] > updated["last_run_at"]

    def test_queued_job_is_recovered_and_executed(self, client):
        import server.app as server_app

        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "queued-recovery", "steps": [
                {"id": "a", "description": "Do a", "agent_role": "writer"}]
        }).json()["run_id"]
        server_app.manager.state.enqueue_job(run_id, "run_full", {"fresh": True})
        assert server_app.manager.launch_queued_jobs() == 1
        snapshot = wait_until_idle(client, run_id)
        assert snapshot["steps"][0]["result"]["status"] == "done"
        assert snapshot["jobs"][0]["status"] == "completed"

    def test_queue_only_mode_leaves_job_for_external_worker(self, client, monkeypatch):
        import server.app as server_app

        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "external-worker", "steps": [
                {"id": "a", "description": "Do a", "agent_role": "writer"}]
        }).json()["run_id"]
        monkeypatch.setenv("ORCHESTRATOR_QUEUE_ONLY", "1")
        assert client.post(f"/api/runs/{run_id}/run", json={"fresh": True}).status_code == 200
        assert server_app.manager.state.queued_jobs()[0]["run_id"] == run_id
        assert not server_app.manager.busy[run_id]
        monkeypatch.delenv("ORCHESTRATOR_QUEUE_ONLY")
        assert server_app.manager.launch_queued_jobs() == 1
        assert wait_until_idle(client, run_id)["jobs"][0]["status"] == "completed"

    def test_webhook_trigger_runs_workflow_with_event_data(self, client, stub_llm):
        import server.app as server_app

        run_id = client.post(
            "/api/runs/from-steps",
            json={"run_id": "webhook-run", "steps": [
                {"id": "handle", "description": "Handle the incoming event",
                 "agent_role": "research"}
            ]},
        ).json()["run_id"]
        created = client.post(
            f"/api/runs/{run_id}/webhooks", json={"name": "New order"})
        assert created.status_code == 201
        hook = created.json()
        assert hook["url"].startswith("http://testserver/hooks/")
        assert "token_hash" not in hook

        triggered = client.post(hook["url"], json={"order_id": 42, "priority": "high"})
        assert triggered.status_code == 202
        snapshot = wait_until_idle(client, run_id)
        workflow = server_app.manager.get(run_id)
        assert snapshot["steps"][0]["result"]["status"] == "done"
        assert workflow.runtime_inputs["order_id"] == 42
        assert workflow.all_input_values()["trigger"]["priority"] == "high"
        assert client.get(f"/api/runs/{run_id}/webhooks").json()[0]["last_triggered_at"]

    def test_webhook_can_be_paused_and_does_not_expose_its_digest(self, client):
        run_id = client.post(
            "/api/runs/from-steps",
            json={"run_id": "paused-hook", "steps": [
                {"id": "a", "description": "Do a", "agent_role": "writer"}
            ]},
        ).json()["run_id"]
        hook = client.post(
            f"/api/runs/{run_id}/webhooks", json={"name": "Inbound"}).json()
        paused = client.patch(
            f"/api/runs/{run_id}/webhooks/{hook['id']}", json={"enabled": False})
        assert paused.status_code == 200
        assert "token_hash" not in paused.json()
        assert client.post(hook["url"], json={"x": 1}).status_code == 404

    def test_webhook_secret_authenticates_without_control_plane_key(self, client,
                                                                     monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_API_KEY", "control-secret")
        headers = {"Authorization": "Bearer control-secret"}
        run_id = client.post(
            "/api/runs/from-steps", headers=headers,
            json={"run_id": "secure-hook", "steps": [
                {"id": "a", "description": "Handle event", "agent_role": "writer"}
            ]},
        ).json()["run_id"]
        hook = client.post(
            f"/api/runs/{run_id}/webhooks", headers=headers,
            json={"name": "External"}).json()

        # The unguessable URL is the webhook credential. External services do
        # not need the dashboard's separate control-plane key.
        assert client.post(hook["url"], json={"event": "created"}).status_code == 202
        wait_until_idle(client, run_id, headers=headers)
        assert client.post("/hooks/wrong-token", json={}).status_code == 404

    def test_save_and_launch_workflow_template(self, client):
        run_id = client.post(
            "/api/runs/from-steps",
            json={"description": "Reusable task", "steps": [
                {"id": "a", "description": "Do a", "agent_role": "writer"},
                {"id": "b", "description": "Do b", "agent_role": "testing",
                 "depends_on": ["a"]},
            ]},
        ).json()["run_id"]
        saved = client.post(
            f"/api/runs/{run_id}/template", json={"name": "Reusable"})
        assert saved.status_code == 201
        template_id = saved.json()["id"]
        launched = client.post(
            f"/api/templates/{template_id}/launch", json={"run_id": "from-template"})
        assert launched.status_code == 201
        assert launched.json()["run_id"] == "from-template"
        assert [s["id"] for s in launched.json()["steps"]] == ["a", "b"]
        assert client.delete(f"/api/templates/{template_id}").status_code == 200

    def test_cancel_endpoint_stops_later_steps(self, client):
        from orchestrator.llm import StubProvider
        import server.app as server_app

        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "cancel-1",
            "parallel": False,
            "steps": [
                {"id": "a", "description": "Do a", "agent_role": "research"},
                {"id": "b", "description": "Do b", "agent_role": "testing",
                 "depends_on": ["a"]},
            ],
        }).json()["run_id"]
        workflow = server_app.manager.get(run_id)
        slow = StubProvider(latency_s=0.2)
        for role in ("research", "testing"):
            workflow.agents.get(role).llm = slow

        client.post(f"/api/runs/{run_id}/run", json={"fresh": True})
        deadline = time.time() + 2
        while workflow.status_of("a").value != "running" and time.time() < deadline:
            time.sleep(0.01)
        response = client.post(f"/api/runs/{run_id}/cancel")
        assert response.status_code == 200 and response.json()["accepted"]

        snapshot = wait_until_idle(client, run_id)
        statuses = {step["id"]: step["result"]["status"] for step in snapshot["steps"]}
        assert statuses == {"a": "done", "b": "cancelled"}

    def test_cancel_idle_run_is_rejected(self, client):
        run_id = client.post("/api/runs/from-steps", json={
            "steps": [{"id": "a", "description": "Do a", "agent_role": "backend"}],
        }).json()["run_id"]
        assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409

    def test_unknown_run_is_404(self, client):
        assert client.get("/api/runs/nope").status_code == 404

    def test_delete_removes_the_run(self, client):
        run_id = client.post("/api/runs", json={"description": "Build a thing"}).json()["run_id"]
        assert client.delete(f"/api/runs/{run_id}").status_code == 200
        assert client.get("/api/runs").json() == []

    def test_delete_unknown_run_is_404(self, client):
        assert client.delete("/api/runs/not-real").status_code == 404

    def test_delete_running_workflow_is_rejected(self, client):
        from orchestrator.llm import StubProvider
        import server.app as server_app

        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "delete-busy",
            "steps": [{"id": "a", "description": "Slow work", "agent_role": "research"}],
        }).json()["run_id"]
        workflow = server_app.manager.get(run_id)
        workflow.agents.get("research").llm = StubProvider(latency_s=0.25)
        client.post(f"/api/runs/{run_id}/run", json={"fresh": True})

        deadline = time.time() + 2
        while workflow.status_of("a").value != "running" and time.time() < deadline:
            time.sleep(0.01)
        response = client.delete(f"/api/runs/{run_id}")
        assert response.status_code == 409
        wait_until_idle(client, run_id)
        assert client.get(f"/api/runs/{run_id}").status_code == 200

    def test_waiting_for_input_is_persisted_as_paused(self, client):
        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "needs-input",
            "steps": [{
                "id": "send", "description": "Send report", "agent_role": "writer",
                "inputs": [{"name": "recipient", "prompt": "Who receives it?"}],
            }],
        }).json()["run_id"]
        client.post(f"/api/runs/{run_id}/run", json={"fresh": True})
        wait_until_idle(client, run_id)
        record = next(r for r in client.get("/api/runs").json() if r["run_id"] == run_id)
        assert record["status"] == "paused"


class TestAdaptiveEndpoints:
    @pytest.fixture
    def ran(self, client):
        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "adaptive-1",
            "steps": [
                {"id": "design", "description": "Design it", "agent_role": "research"},
                {"id": "build", "description": "Build it", "agent_role": "backend",
                 "requires_tool": "github", "depends_on": ["design"]},
                {"id": "test", "description": "Test it", "agent_role": "testing",
                 "depends_on": ["build"]},
            ],
        }).json()["run_id"]
        client.post(f"/api/runs/{run_id}/run", json={"fresh": True})
        wait_until_idle(client, run_id)
        return run_id

    def test_impact_endpoint_runs_nothing(self, client, ran):
        body = client.get(f"/api/runs/{ran}/impact/build").json()
        assert body["affected"] == ["build", "test"]
        assert body["reusable"] == ["design"]

    def test_impact_of_unknown_step_is_404(self, client, ran):
        assert client.get(f"/api/runs/{ran}/impact/ghost").status_code == 404

    def test_change_reruns_only_the_affected(self, client, ran):
        response = client.post(f"/api/runs/{ran}/steps/build/change",
                               json={"new_description": "Rebuild it in Rust"})
        assert response.status_code == 200
        assert response.json()["impact"]["affected_count"] == 2
        snapshot = wait_until_idle(client, ran)
        statuses = {s["id"]: s["result"]["status"] for s in snapshot["steps"]}
        assert statuses["design"] == "skipped"
        assert statuses["build"] == "done" and statuses["test"] == "done"

    def test_change_on_unknown_step_is_404(self, client, ran):
        assert client.post(f"/api/runs/{ran}/steps/ghost/change", json={}).status_code == 404

    def test_breaking_a_tool_triggers_the_fallback(self, client, ran):
        assert client.post(f"/api/runs/{ran}/tools/github/break?rerun=true").status_code == 200
        snapshot = wait_until_idle(client, ran)
        build = next(s for s in snapshot["steps"] if s["id"] == "build")
        assert build["result"]["used_fallback"]
        assert build["result"]["tool_used"] == "github_cli"

    def test_repairing_a_tool(self, client, ran):
        client.post(f"/api/runs/{ran}/tools/github/break?rerun=false")
        response = client.post(f"/api/runs/{ran}/tools/github/repair")
        assert response.status_code == 200
        snapshot = client.get(f"/api/runs/{ran}").json()
        github = next(t for t in snapshot["tools"] if t["name"] == "github")
        assert not github["broken"]

    def test_breaking_an_unknown_tool_is_404(self, client, ran):
        assert client.post(f"/api/runs/{ran}/tools/ghost/break").status_code == 404

    def test_pause_and_approve(self, client, ran):
        assert client.post(f"/api/runs/{ran}/steps/test/pause").status_code == 200
        client.post(f"/api/runs/{ran}/run", json={"fresh": True})
        snapshot = wait_until_idle(client, ran)
        assert snapshot["awaiting_approval"] == ["test"]

        assert client.post(f"/api/runs/{ran}/steps/test/approve").status_code == 200
        client.post(f"/api/runs/{ran}/resume")
        snapshot = wait_until_idle(client, ran)
        statuses = {s["id"]: s["result"]["status"] for s in snapshot["steps"]}
        assert statuses["test"] == "done"

    def test_events_are_queryable(self, client, ran):
        events = client.get(f"/api/runs/{ran}/events").json()
        assert any(e["type"] == "step_finished" for e in events)

    @pytest.mark.parametrize("fmt", ["json", "csv", "html"])
    def test_exports(self, client, ran, fmt):
        response = client.get(f"/api/runs/{ran}/export?format={fmt}")
        assert response.status_code == 200 and response.content

    def test_unknown_export_format_is_400(self, client, ran):
        assert client.get(f"/api/runs/{ran}/export?format=pdf").status_code == 400


class TestWebSocket:
    def test_snapshot_arrives_on_connect(self, client):
        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "ws-1",
            "steps": [{"id": "a", "description": "Do a", "agent_role": "backend"}],
        }).json()["run_id"]

        with client.websocket_connect(f"/ws/{run_id}") as socket:
            frame = socket.receive_json()
            assert frame["type"] == "snapshot"
            assert frame["data"]["run_id"] == run_id

    def test_websocket_authenticates_before_exposing_run_data(self, client, monkeypatch):
        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "ws-auth",
            "steps": [{"id": "a", "description": "Do a", "agent_role": "backend"}],
        }).json()["run_id"]
        monkeypatch.setenv("ORCHESTRATOR_API_KEY", "socket-secret")

        with client.websocket_connect(f"/ws/{run_id}") as socket:
            socket.send_json({"type": "auth", "token": "socket-secret"})
            frame = socket.receive_json()
            assert frame["type"] == "snapshot"
            assert frame["data"]["run_id"] == run_id

    def test_events_stream_while_a_run_executes(self, client):
        run_id = client.post("/api/runs/from-steps", json={
            "run_id": "ws-2",
            "steps": [
                {"id": "a", "description": "Do a", "agent_role": "backend"},
                {"id": "b", "description": "Do b", "agent_role": "testing", "depends_on": ["a"]},
            ],
        }).json()["run_id"]

        with client.websocket_connect(f"/ws/{run_id}") as socket:
            assert socket.receive_json()["type"] == "snapshot"
            client.post(f"/api/runs/{run_id}/run", json={"fresh": True})

            seen = []
            for _ in range(40):
                frame = socket.receive_json()
                if frame["type"] == "event":
                    seen.append(frame["data"]["type"])
                    if frame["data"]["type"] == "run_finished":
                        break
            assert "step_started" in seen and "step_finished" in seen
            assert "run_finished" in seen

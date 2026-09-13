"""Reusable API connections: auth, routing, and per-call safety."""

from __future__ import annotations

import json

import pytest

import orchestrator.tools.builtin as builtin
from orchestrator.connections import (
    ApiConnection,
    ApiConnectionTool,
    AuthSpec,
    ConnectionError_,
    _token_cache,
    attach_connections,
    connection_requirements,
    load_connections,
)
from orchestrator.models import Step, StepStatus
from orchestrator.tools import ToolManager


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


@pytest.fixture
def http_calls(monkeypatch):
    """Capture outbound requests instead of making them."""
    calls = []

    def fake_http(method, url, *, headers=None, json_body=None, timeout=20.0):
        calls.append({"method": method, "url": url, "headers": headers or {},
                      "body": json_body})
        if "token" in url:
            return FakeResponse(payload={"access_token": "oauth-tok", "expires_in": 3600})
        return FakeResponse()

    monkeypatch.setattr(builtin, "_http", fake_http)
    _token_cache.clear()
    return calls


def make_tool(**auth) -> ApiConnectionTool:
    return ApiConnectionTool(ApiConnection(
        name="svc", base_url="https://api.example.com/v1",
        auth=AuthSpec(**auth), description="Example service"))


def directive(**kwargs) -> str:
    return "do it\nTOOL_DIRECTIVE: " + json.dumps(kwargs)


class TestAuth:
    def test_bearer_sets_the_header(self, http_calls, monkeypatch):
        monkeypatch.setenv("SVC_TOKEN", "tok-123")
        tool = make_tool(type="bearer", token_env="SVC_TOKEN")
        assert tool.is_live()
        tool.execute(directive(method="GET", path="/things"))
        assert http_calls[0]["headers"]["Authorization"] == "Bearer tok-123"

    def test_basic_encodes_credentials(self, http_calls, monkeypatch):
        monkeypatch.setenv("U", "alice")
        monkeypatch.setenv("P", "hunter2")
        tool = make_tool(type="basic", username_env="U", password_env="P")
        tool.execute(directive(method="GET", path="/x"))
        import base64

        expected = base64.b64encode(b"alice:hunter2").decode()
        assert http_calls[0]["headers"]["Authorization"] == f"Basic {expected}"

    def test_api_key_as_a_header(self, http_calls, monkeypatch):
        monkeypatch.setenv("K", "abc")
        tool = make_tool(type="api_key", token_env="K", header="X-Api-Key", prefix="")
        tool.execute(directive(method="GET", path="/x"))
        assert http_calls[0]["headers"]["X-Api-Key"] == "abc"

    def test_api_key_as_a_query_parameter(self, http_calls, monkeypatch):
        monkeypatch.setenv("K", "abc")
        tool = make_tool(type="api_key", token_env="K", query_param="appid")
        tool.execute(directive(method="GET", path="/weather"))
        assert "appid=abc" in http_calls[0]["url"]

    def test_no_auth_needs_nothing(self, http_calls):
        tool = make_tool(type="none")
        assert tool.is_live()
        tool.execute(directive(method="GET", path="/public"))
        assert "Authorization" not in http_calls[0]["headers"]

    def test_oauth2_fetches_then_caches_a_token(self, http_calls, monkeypatch):
        monkeypatch.setenv("CID", "id")
        monkeypatch.setenv("CSEC", "secret")
        tool = ApiConnectionTool(ApiConnection(
            name="sf", base_url="https://api.example.com",
            auth=AuthSpec(type="oauth2_client_credentials",
                          token_url="https://api.example.com/token",
                          client_id_env="CID", client_secret_env="CSEC")))
        tool.execute(directive(method="GET", path="/a"))
        tool.execute(directive(method="GET", path="/b"))

        token_calls = [c for c in http_calls if c["url"].endswith("/token")]
        assert len(token_calls) == 1, "the token must be cached, not refetched"
        assert http_calls[-1]["headers"]["Authorization"] == "Bearer oauth-tok"

    def test_missing_credential_simulates_rather_than_failing(self, monkeypatch):
        """Consistent with every other adapter: degrade, do not break the run."""
        monkeypatch.delenv("ABSENT", raising=False)
        tool = make_tool(type="bearer", token_env="ABSENT")
        assert not tool.is_live()
        assert tool.missing_credentials() == ["ABSENT"]
        result = tool.execute(directive(method="GET", path="/x"))
        assert result.startswith("[simulated:")
        assert "ABSENT" in result, "the simulation must say what would make it real"

    def test_a_broken_connection_does_fail(self, monkeypatch):
        monkeypatch.setenv("T", "tok")
        tool = make_tool(type="bearer", token_env="T")
        tool.break_it()
        from orchestrator.tools import ToolError

        with pytest.raises(ToolError, match="broken"):
            tool.execute(directive(method="GET", path="/x"))

    def test_missing_env_name_in_config_is_an_error(self, monkeypatch):
        tool = make_tool(type="bearer")  # token_env not given
        with pytest.raises(ConnectionError_, match="environment variable name"):
            tool.connection.build_headers()


class TestRequestBuilding:
    def test_path_is_joined_to_the_base_url(self, http_calls):
        make_tool(type="none").execute(directive(method="GET", path="things/1"))
        assert http_calls[0]["url"] == "https://api.example.com/v1/things/1"

    def test_leading_slash_does_not_escape_the_base_path(self, http_calls):
        make_tool(type="none").execute(directive(method="GET", path="/things"))
        assert http_calls[0]["url"] == "https://api.example.com/v1/things"

    def test_query_parameters_are_encoded(self, http_calls):
        make_tool(type="none").execute(
            directive(method="GET", path="/s", query={"q": "a b", "n": 2}))
        assert "q=a+b" in http_calls[0]["url"] and "n=2" in http_calls[0]["url"]

    def test_body_is_sent_as_json(self, http_calls):
        make_tool(type="none").execute(
            directive(method="POST", path="/x", body={"a": 1}))
        assert http_calls[0]["body"] == {"a": 1}

    def test_default_headers_are_included(self, http_calls):
        tool = ApiConnectionTool(ApiConnection(
            name="n", base_url="https://api.example.com",
            headers={"Notion-Version": "2022-06-28"}))
        tool.execute(directive(method="GET", path="/x"))
        assert http_calls[0]["headers"]["Notion-Version"] == "2022-06-28"

    def test_missing_path_calls_nothing(self, http_calls):
        result = make_tool(type="none").execute("just prose, no directive")
        assert "no 'path'" in result
        assert http_calls == []

    def test_operator_inputs_can_steer_the_call(self, http_calls):
        tool = make_tool(type="none")
        tool.execute("x", context={"inputs": {"method": "GET", "path": "/from-input"}})
        assert http_calls[0]["url"].endswith("/from-input")


class TestPerCallSafety:
    @pytest.mark.parametrize("method,expected", [
        ("GET", False), ("HEAD", False), ("OPTIONS", False),
        ("POST", True), ("PUT", True), ("PATCH", True), ("DELETE", True),
    ])
    def test_only_writes_are_irreversible(self, method, expected):
        tool = make_tool(type="none")
        assert tool.is_irreversible(directive(method=method, path="/x")) is expected

    def test_default_method_is_a_safe_get(self):
        assert not make_tool(type="none").is_irreversible(directive(path="/x"))

    def test_safe_methods_can_be_widened_per_connection(self):
        tool = ApiConnectionTool(ApiConnection(
            name="idem", base_url="https://api.example.com",
            safe_methods=["GET", "POST"]))
        assert not tool.is_irreversible(directive(method="POST", path="/x"))
        assert tool.is_irreversible(directive(method="DELETE", path="/x"))

    def test_preview_shows_the_exact_request(self):
        preview = make_tool(type="none").preview(
            directive(method="DELETE", path="/customers/42"))
        assert "DELETE" in preview and "customers/42" in preview

    def test_prompt_hint_documents_the_call_shape(self):
        hint = make_tool(type="bearer", token_env="T").prompt_hint()
        assert "TOOL_DIRECTIVE" in hint
        assert "https://api.example.com/v1/" in hint
        assert "never include credentials" in hint


class TestConfigLoading:
    def test_missing_file_yields_nothing(self, tmp_path):
        assert load_connections(str(tmp_path / "none.json")) == []

    def test_connections_are_parsed(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"connections": {
            "gh": {"base_url": "https://api.github.com",
                   "auth": {"type": "bearer", "token_env": "GH"}}}}), encoding="utf-8")
        connections = load_connections(str(path))
        assert connections[0].name == "gh"
        assert connections[0].required_env() == ["GH"]

    def test_comment_keys_are_ignored(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"connections": {
            "_comment": ["ignore me"],
            "gh": {"base_url": "https://api.github.com"}}}), encoding="utf-8")
        assert [c.name for c in load_connections(str(path))] == ["gh"]

    def test_entry_without_base_url_is_skipped(self, tmp_path, capsys):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"connections": {"bad": {"description": "no url"}}}),
                        encoding="utf-8")
        assert load_connections(str(path)) == []
        assert "skipping 'bad'" in capsys.readouterr().out

    def test_shipped_example_config_parses(self):
        connections = load_connections("connections.json.example")
        names = {c.name for c in connections}
        assert {"github", "stripe", "notion", "salesforce"} <= names

    def test_serialisation_never_leaks_secret_values(self, monkeypatch):
        monkeypatch.setenv("SVC_TOKEN", "super-secret")
        payload = json.dumps(make_tool(type="bearer", token_env="SVC_TOKEN")
                             .connection.to_dict())
        assert "super-secret" not in payload
        assert "SVC_TOKEN" in payload  # the name is fine to expose


class TestRegistration:
    def test_connections_register_as_tools(self, tmp_path):
        manager = ToolManager()
        registered = attach_connections(manager, [
            ApiConnection(name="alpha", base_url="https://a.example.com"),
            ApiConnection(name="beta", base_url="https://b.example.com"),
        ])
        assert registered == ["api_alpha", "api_beta"]
        assert "api_alpha" in manager

    def test_each_api_gets_its_own_capability(self):
        """Falling back from one service to an unrelated one would be dangerous."""
        manager = ToolManager()
        attach_connections(manager, [
            ApiConnection(name="alpha", base_url="https://a.example.com"),
            ApiConnection(name="beta", base_url="https://b.example.com"),
        ])
        assert manager.candidates_for("api_alpha") == ["api_alpha"]

    def test_requirements_are_derived_from_auth(self):
        requirements = connection_requirements([
            ApiConnection(name="gh", base_url="https://api.github.com",
                          auth=AuthSpec(type="bearer", token_env="GH_TOKEN"))])
        assert [r.name for r in requirements] == ["GH_TOKEN"]
        assert requirements[0].optional, "a missing key degrades rather than blocks"


class TestWorkflowIntegration:
    def test_agent_can_call_a_connection(self, make_workflow, http_calls, monkeypatch):
        monkeypatch.setenv("SVC_TOKEN", "tok")
        workflow = make_workflow(
            [Step(id="fetch", description="Fetch the customer list.",
                  agent_role="backend", requires_tool="api_svc")],
            tool_manager=ToolManager())
        workflow.attach_connections(connections=[ApiConnection(
            name="svc", base_url="https://api.example.com/v1",
            auth=AuthSpec(type="bearer", token_env="SVC_TOKEN"))])

        report = workflow.run_full()
        assert report.ok
        assert workflow.results["fetch"].tool_used == "api_svc"

    def test_a_write_call_hits_the_approval_gate(self, make_workflow, http_calls, monkeypatch):
        monkeypatch.setenv("SVC_TOKEN", "tok")

        class WritingAgent:
            """Stands in for a model that decided to POST."""

            role = "backend"

            def build_system_prompt(self):
                return ""

            def build_prompt(self, step, context, tools=None):
                return ""

            def execute(self, step, context, tool_manager=None, gate=None, skills_block="",
                        workflow_inputs=None):
                from orchestrator.agents import AgentOutcome

                payload = directive(method="POST", path="/customers", body={"a": 1})
                outcome = AgentOutcome(output=payload)
                if gate is not None and not gate(step, "api_svc", payload):
                    outcome.deferred_tool = "api_svc"
                    return outcome
                outcome.tool_invocation = self.call_tool(step, "api_svc", payload, tool_manager)
                outcome.output = self.merge_tool_result(payload, "api_svc",
                                                        outcome.tool_invocation)
                return outcome

            def call_tool(self, step, tool_name, payload, tool_manager, workflow_inputs=None):
                return tool_manager.use(tool_name, payload, context={"step_id": step.id})

            @staticmethod
            def merge_tool_result(agent_output, requested, invocation):
                return f"{agent_output}\n---\n{invocation.output}"

        workflow = make_workflow(
            [Step(id="create", description="Create a customer.",
                  agent_role="backend", requires_tool="api_svc")],
            tool_manager=ToolManager())
        workflow.attach_connections(connections=[ApiConnection(
            name="svc", base_url="https://api.example.com/v1",
            auth=AuthSpec(type="bearer", token_env="SVC_TOKEN"))])
        workflow.agents._agents["backend"] = WritingAgent()  # noqa: SLF001

        report = workflow.run_full()
        assert report.awaiting_action == ["create"]
        assert http_calls == [], "nothing may be sent before approval"

        action = workflow.pending_actions()["create"]
        assert "POST" in action.preview and action.irreversible

        workflow.approve_action("create")
        workflow.resume()
        assert http_calls and http_calls[0]["method"] == "POST"
        assert workflow.status_of("create") is StepStatus.DONE

    def test_credentials_appear_in_the_requirements_report(self, make_workflow, monkeypatch):
        monkeypatch.delenv("SVC_TOKEN", raising=False)
        workflow = make_workflow(
            [Step(id="fetch", description="Fetch.", agent_role="backend",
                  requires_tool="api_svc")],
            tool_manager=ToolManager())
        workflow.attach_connections(connections=[ApiConnection(
            name="svc", base_url="https://api.example.com/v1",
            auth=AuthSpec(type="bearer", token_env="SVC_TOKEN"))])
        names = {r.name for r in workflow.check_requirements().requirements}
        assert "SVC_TOKEN" in names

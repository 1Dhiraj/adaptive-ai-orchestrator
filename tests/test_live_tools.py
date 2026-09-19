"""Live-mode tool adapters, with the network and subprocesses mocked out.

These verify that when credentials *are* configured the adapters build the
right request (URL, method, body) — the part that cannot be checked by
running them in simulation mode.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import types

import pytest

import orchestrator.tools.builtin as builtin
from orchestrator.tools.base import ToolError
from orchestrator.tools.builtin import (
    CITool,
    EmailTool,
    GitHubCLITool,
    GitHubTool,
    PostgresCLITool,
    PostgresTool,
    RestApiTool,
    SlackTool,
    S3Tool,
    StripeTool,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


@pytest.fixture
def http_calls(monkeypatch):
    """Capture every outbound HTTP call instead of making it."""
    calls = []

    def fake_http(method, url, *, headers=None, json_body=None, timeout=20.0):
        calls.append({"method": method, "url": url, "headers": headers or {},
                      "body": json_body or {}})
        if "issues" in url:
            return FakeResponse(payload={"html_url": "https://github.com/o/r/issues/1"})
        if "contents" in url:
            return FakeResponse(payload={"content": {"html_url": "https://github.com/o/r/f.md"}})
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(builtin, "_http", fake_http)
    return calls


@pytest.fixture
def github_creds(monkeypatch):
    monkeypatch.setattr(builtin.settings, "github_token", "ghp_test")
    monkeypatch.setattr(builtin.settings, "github_repo", "octocat/hello")


class TestGitHub:
    def test_simulated_without_credentials(self):
        result = GitHubTool().execute("some plan", {"step_name": "Backend"})
        assert result.startswith("[simulated:github]")

    def test_creates_an_issue_when_live(self, github_creds, http_calls):
        tool = GitHubTool()
        assert tool.is_live()
        result = tool.execute("the API design", {"step_name": "Build API"})

        assert "issues/1" in result
        call = http_calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "https://api.github.com/repos/octocat/hello/issues"
        assert call["body"]["title"] == "Build API"
        assert call["body"]["body"] == "the API design"
        assert call["headers"]["Authorization"] == "Bearer ghp_test"

    def test_directive_selects_create_file_and_encodes_content(self, github_creds, http_calls):
        payload = 'schema here\nTOOL_DIRECTIVE: {"action": "create_file", "path": "docs/api.md"}'
        GitHubTool().execute(payload, {"step_name": "Docs"})

        call = http_calls[0]
        assert call["method"] == "PUT"
        assert call["url"].endswith("/contents/docs/api.md")
        assert base64.b64decode(call["body"]["content"]).decode() == payload

    def test_unsupported_action_raises(self, github_creds, http_calls):
        with pytest.raises(ToolError, match="unsupported github action"):
            GitHubTool().execute('x\nTOOL_DIRECTIVE: {"action": "launch_rocket"}')

    def test_broken_tool_refuses_to_run(self, github_creds):
        tool = GitHubTool()
        tool.break_it()
        with pytest.raises(ToolError, match="marked broken"):
            tool.execute("x")


class TestGitHubCLI:
    def test_simulated_when_gh_is_absent(self, monkeypatch):
        monkeypatch.setattr(builtin.shutil, "which", lambda name: None)
        assert GitHubCLITool().execute("x").startswith("[simulated:github_cli]")

    def test_invokes_gh_when_available(self, monkeypatch):
        monkeypatch.setattr(builtin.shutil, "which", lambda name: "/usr/bin/gh")
        monkeypatch.setattr(builtin.settings, "github_repo", "octocat/hello")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="https://github.com/o/r/issues/2", stderr="")

        monkeypatch.setattr(builtin.subprocess, "run", fake_run)
        result = GitHubCLITool().execute("body text", {"step_name": "Ship"})

        assert "issues/2" in result
        assert captured["cmd"][:3] == ["gh", "issue", "create"]
        assert "octocat/hello" in captured["cmd"]

    def test_non_zero_exit_becomes_a_tool_error(self, monkeypatch):
        monkeypatch.setattr(builtin.shutil, "which", lambda name: "/usr/bin/gh")
        monkeypatch.setattr(builtin.settings, "github_repo", "octocat/hello")
        monkeypatch.setattr(builtin.subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "not authed"))
        with pytest.raises(ToolError, match="not authed"):
            GitHubCLITool().execute("x")


class TestPostgres:
    def test_simulated_without_a_dsn(self, monkeypatch):
        monkeypatch.setattr(builtin.settings, "postgres_dsn", None)
        payload = "```sql\nCREATE TABLE a (id INT); CREATE TABLE b (id INT);\n```"
        assert "2 statement(s)" in PostgresTool().execute(payload)

    def test_is_not_live_without_a_driver(self, monkeypatch):
        monkeypatch.setattr(builtin.settings, "postgres_dsn", "postgresql://localhost/db")
        # No psycopg installed in this environment, so it must stay simulated.
        tool = PostgresTool()
        if tool.is_live():
            pytest.skip("psycopg is installed here; the simulated path cannot be asserted")
        assert tool.execute("```sql\nCREATE TABLE a (id INT);\n```").startswith("[simulated")

    def test_psql_cli_is_simulated_when_absent(self, monkeypatch):
        monkeypatch.setattr(builtin.shutil, "which", lambda name: None)
        assert PostgresCLITool().execute("x").startswith("[simulated:postgres_cli]")

    def test_psql_cli_runs_when_available(self, monkeypatch):
        monkeypatch.setattr(builtin.shutil, "which", lambda name: "/usr/bin/psql")
        monkeypatch.setattr(builtin.settings, "postgres_dsn", "postgresql://localhost/db")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="CREATE TABLE", stderr="")

        monkeypatch.setattr(builtin.subprocess, "run", fake_run)
        result = PostgresCLITool().execute("```sql\nCREATE TABLE a (id INT);\n```")
        assert "CREATE TABLE" in result
        assert captured["cmd"][0] == "psql"
        assert "ON_ERROR_STOP=1" in captured["cmd"]


class TestCI:
    def test_simulated_without_credentials(self, monkeypatch):
        monkeypatch.setattr(builtin.settings, "github_token", None)
        assert CITool().execute("x").startswith("[simulated:ci]")

    def test_dispatches_the_named_workflow(self, github_creds, http_calls):
        CITool().execute('run them\nTOOL_DIRECTIVE: {"workflow": "tests.yml", "ref": "develop"}')
        call = http_calls[0]
        assert call["url"].endswith("/actions/workflows/tests.yml/dispatches")
        assert call["body"]["ref"] == "develop"

    def test_local_test_runner_is_opt_in(self, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_TEST_CMD", raising=False)
        result = builtin.LocalTestRunnerTool().execute("x")
        assert "set ORCHESTRATOR_TEST_CMD" in result

    def test_local_test_runner_reports_the_verdict(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_TEST_CMD", "echo hi")
        monkeypatch.setattr(builtin.subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "3 passed", ""))
        assert "passed" in builtin.LocalTestRunnerTool().execute("x")

    def test_local_test_runner_reports_failure(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_TEST_CMD", "false")
        monkeypatch.setattr(builtin.subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "1 failed", ""))
        assert "FAILED" in builtin.LocalTestRunnerTool().execute("x")


class TestNotify:
    def test_slack_simulated_without_a_webhook(self, monkeypatch):
        monkeypatch.setattr(builtin.settings, "slack_webhook", None)
        assert SlackTool().execute("hello").startswith("[simulated:slack]")

    def test_slack_posts_to_the_webhook(self, monkeypatch, http_calls):
        monkeypatch.setattr(builtin.settings, "slack_webhook", "https://hooks.slack.test/abc")
        assert "posted" in SlackTool().execute("deploy done")
        assert http_calls[0]["url"] == "https://hooks.slack.test/abc"
        assert http_calls[0]["body"]["text"] == "deploy done"

    def test_email_simulated_without_credentials(self, monkeypatch):
        monkeypatch.setattr(builtin.settings, "sendgrid_api_key", None)
        assert EmailTool().execute("body").startswith("[simulated:email]")

    def test_email_builds_a_sendgrid_payload(self, monkeypatch, http_calls):
        monkeypatch.setattr(builtin.settings, "sendgrid_api_key", "SG.test")
        monkeypatch.setenv("EMAIL_TO", "dev@example.com")
        monkeypatch.setenv("EMAIL_FROM", "sender@example.com")
        EmailTool().execute('report\nTOOL_DIRECTIVE: {"subject": "Nightly build"}')
        call = http_calls[0]
        assert call["url"] == "https://api.sendgrid.com/v3/mail/send"
        assert call["body"]["subject"] == "Nightly build"
        assert call["body"]["personalizations"][0]["to"][0]["email"] == "dev@example.com"


class TestRestApi:
    def test_no_directive_means_no_call(self, http_calls):
        assert "no url" in RestApiTool().execute("just some prose")
        assert http_calls == []

    def test_directive_drives_the_request(self, http_calls):
        RestApiTool().execute(
            'ping\nTOOL_DIRECTIVE: {"url": "https://api.test/v1/ping", "method": "post", '
            '"body": {"a": 1}}')
        call = http_calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "https://api.test/v1/ping"
        assert call["body"] == {"a": 1}


class TestStripe:
    def test_simulates_without_key(self, monkeypatch):
        monkeypatch.delenv("STRIPE_API_KEY", raising=False)
        assert StripeTool().execute("x").startswith("[simulated:stripe]")

    def test_lists_customers_with_bearer_key(self, monkeypatch):
        monkeypatch.setenv("STRIPE_API_KEY", "rk_test_example")
        calls = []
        fake = types.SimpleNamespace(status_code=200, text='{"data":[]}')
        monkeypatch.setattr("requests.request", lambda *a, **kw: calls.append((a, kw)) or fake)
        result = StripeTool().execute(
            'x\nTOOL_DIRECTIVE: {"action":"list_customers","params":{"limit":3}}')
        assert "succeeded" in result
        assert calls[0][0] == ("GET", "https://api.stripe.com/v1/customers")
        assert calls[0][1]["headers"]["Authorization"] == "Bearer rk_test_example"
        assert calls[0][1]["params"] == {"limit": 3}

    def test_only_create_actions_need_irreversible_approval(self):
        tool = StripeTool()
        assert not tool.is_irreversible('TOOL_DIRECTIVE: {"action":"list_customers"}')
        assert tool.is_irreversible('TOOL_DIRECTIVE: {"action":"create_customer"}')


class TestS3:
    def test_simulates_without_credentials(self, monkeypatch):
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        assert S3Tool().execute("x").startswith("[simulated:s3]")

    def test_uploads_to_configured_bucket(self, monkeypatch):
        monkeypatch.setenv("AWS_S3_BUCKET", "workflow-files")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTTESTTESTTEST")
        calls = []
        client = types.SimpleNamespace(
            put_object=lambda **kw: calls.append(kw),
        )
        monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(
            client=lambda *a, **kw: client))
        result = S3Tool().execute(
            'hello\nTOOL_DIRECTIVE: {"action":"put","key":"reports/result.txt","content":"done"}')
        assert "uploaded" in result
        assert calls == [{"Bucket": "workflow-files", "Key": "reports/result.txt", "Body": b"done"}]
        assert S3Tool().is_irreversible('TOOL_DIRECTIVE: {"action":"put"}')


class TestHttpErrors:
    def test_error_status_becomes_a_tool_error(self, monkeypatch):
        import orchestrator.tools.builtin as module

        class Boom:
            @staticmethod
            def request(*args, **kwargs):
                return FakeResponse(status_code=503, text="upstream down")

        monkeypatch.setitem(__import__("sys").modules, "requests", Boom)
        with pytest.raises(ToolError, match="503"):
            module._http("GET", "https://api.test/x")

    def test_transport_failure_becomes_a_tool_error(self, monkeypatch):
        import orchestrator.tools.builtin as module

        class Boom:
            @staticmethod
            def request(*args, **kwargs):
                raise ConnectionError("no route to host")

        monkeypatch.setitem(__import__("sys").modules, "requests", Boom)
        with pytest.raises(ToolError, match="no route to host"):
            module._http("GET", "https://api.test/x")

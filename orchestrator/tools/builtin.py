"""Concrete tool adapters.

Every adapter follows the same contract: if the credentials it needs are
present it does the real thing and reports ``is_live() is True``; if they are
absent it degrades to a clearly-labelled simulation. Tools that share a
``capability`` are interchangeable, so :class:`~orchestrator.tools.base.ToolManager`
can route around a failure automatically.

Each capability chain ends in a tool that is *always* live and side-effect
safe (writes to ``artifacts/``, a local SQLite file, or stdout), so a
workflow can always make forward progress.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from urllib.parse import urlencode
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PROJECT_ROOT, settings
from .base import Tool, ToolError, ToolManager

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (s[:limit] or "artifact").rstrip("-")


def _directive(task: str) -> Dict[str, Any]:
    """Pull an optional ``TOOL_DIRECTIVE: {...}`` JSON line out of the payload.

    Lets an agent steer a tool ("open a PR" vs "create an issue") without the
    orchestrator needing to know anything about that tool's operations.
    """
    match = re.search(r"TOOL_DIRECTIVE:\s*(\{.*?\})\s*$", task, re.MULTILINE | re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _http(method: str, url: str, *, headers: Optional[dict] = None,
          json_body: Optional[dict] = None, timeout: float = 20.0):
    """Thin requests wrapper that raises ToolError on any transport problem."""
    try:
        import requests  # lazy: only needed when a live tool is configured
    except ImportError as exc:  # pragma: no cover - requests ships with the deps
        raise ToolError("the 'requests' package is required for live HTTP tools") from exc
    try:
        response = requests.request(method, url, headers=headers, json=json_body, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - network errors are all routable
        raise ToolError(f"HTTP {method} {url} failed: {exc}") from exc
    if response.status_code >= 400:
        raise ToolError(f"HTTP {response.status_code} from {url}: {response.text[:300]}")
    return response


# ---------------------------------------------------------------------------
# capability: vcs
# ---------------------------------------------------------------------------


class GitHubTool(Tool):
    """GitHub REST API. Live when ``GITHUB_TOKEN`` and ``GITHUB_REPO`` are set."""

    name = "github"
    capability = "vcs"
    description = "Create issues / PRs / files in a GitHub repo via the REST API"
    fallbacks = ["github_cli", "artifact_store"]
    side_effect = True
    irreversible = False  # an issue or file can be closed/reverted

    def is_live(self) -> bool:
        return bool(settings.github_token and settings.github_repo)

    def unavailable_reason(self) -> Optional[str]:
        if self.is_broken:
            return "marked broken"
        if not self.is_live():
            return None  # simulation is still a valid way to run
        return None

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        action = directive.get("action", "create_issue")
        title = directive.get("title") or (context or {}).get("step_name") or "Orchestrator output"

        if not self.is_live():
            return f"[simulated:github] would {action} '{title}' ({len(task)} chars of payload)"

        repo = settings.github_repo
        headers = {
            "Authorization": f"Bearer {settings.github_token}",
            "Accept": "application/vnd.github+json",
        }

        if action == "create_issue":
            response = _http(
                "POST",
                f"https://api.github.com/repos/{repo}/issues",
                headers=headers,
                json_body={"title": title, "body": task},
            )
            return f"[github] issue created: {response.json().get('html_url')}"

        if action == "create_file":
            import base64

            path = directive.get("path") or f"orchestrator/{_slug(title)}.md"
            response = _http(
                "PUT",
                f"https://api.github.com/repos/{repo}/contents/{path}",
                headers=headers,
                json_body={
                    "message": directive.get("message", f"orchestrator: {title}"),
                    "content": base64.b64encode(task.encode()).decode(),
                    "branch": directive.get("branch", "main"),
                },
            )
            return f"[github] file written: {response.json().get('content', {}).get('html_url')}"

        raise ToolError(f"unsupported github action: '{action}'")


class GitHubCLITool(Tool):
    """The ``gh`` CLI -- a genuinely independent path to the same capability."""

    name = "github_cli"
    capability = "vcs"
    description = "Create GitHub issues via the gh CLI"
    fallbacks = ["artifact_store"]
    side_effect = True
    irreversible = False  # same as the REST path

    def is_live(self) -> bool:
        return shutil.which("gh") is not None and bool(settings.github_repo)

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        title = directive.get("title") or (context or {}).get("step_name") or "Orchestrator output"
        if not self.is_live():
            return f"[simulated:github_cli] would run `gh issue create --title {title!r}`"
        try:
            proc = subprocess.run(
                ["gh", "issue", "create", "--repo", str(settings.github_repo),
                 "--title", title, "--body", task],
                capture_output=True, text=True, timeout=60, check=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"gh CLI failed: {exc}") from exc
        if proc.returncode != 0:
            raise ToolError(f"gh exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        return f"[github_cli] {proc.stdout.strip()}"


class ArtifactStoreTool(Tool):
    """Always-live terminal fallback for the ``vcs`` chain: write to disk.

    Never fails for credential reasons, so a workflow can always land its
    output somewhere durable even when every remote service is down. Files
    live inside the run workspace so the dashboard can list and download them.
    """

    name = "artifact_store"
    capability = "vcs"
    description = "Persist step output to this run's local artifacts/ directory"
    fallbacks: List[str] = []
    side_effect = True
    irreversible = False  # writes a local file; safe and reversible

    def __init__(self, run_id: str = "default", root: Optional[Path] = None) -> None:
        super().__init__()
        from .workspace import DEFAULT_WORKSPACE_ROOT, workspace_for

        self.workspace = workspace_for(run_id, root or DEFAULT_WORKSPACE_ROOT)

    def is_live(self) -> bool:
        return True

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        step_id = (context or {}).get("step_id", "step")
        artifact_dir = self.workspace / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        directive = _directive(task)
        arguments = directive.get("arguments", directive)
        arguments = arguments if isinstance(arguments, dict) else {}
        requested = str(arguments.get("path") or "").replace("\\", "/").strip()
        if requested:
            if requested.startswith("/") or re.match(r"^[a-zA-Z]:", requested):
                raise ToolError("artifact path must be relative to the run workspace")
            parts = [part for part in requested.split("/") if part not in {"", "."}]
            if parts and parts[0].lower() == "artifacts":
                parts = parts[1:]
            if not parts or ".." in parts:
                raise ToolError("artifact path must stay inside artifacts/")
            path = artifact_dir.joinpath(*parts)
        else:
            path = artifact_dir / f"{_slug(str(step_id))}-{int(time.time())}.md"
        content = arguments.get("content")
        if content is None:
            content = re.sub(r"TOOL_DIRECTIVE:.*$", "", task,
                             flags=re.MULTILINE | re.DOTALL).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(str(content), encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"could not write artifact: {exc}") from exc
        return f"[artifact_store] written to {path.relative_to(self.workspace)}"


# ---------------------------------------------------------------------------
# capability: database
# ---------------------------------------------------------------------------


class PostgresTool(Tool):
    """PostgreSQL over psycopg. Live when ``DATABASE_URL`` is set."""

    name = "postgres"
    capability = "database"
    description = "Run DDL/DML against PostgreSQL"
    fallbacks = ["postgres_cli", "sqlite_local"]
    side_effect = True
    irreversible = True  # DDL/DML against a real database is not undoable

    def is_live(self) -> bool:
        if not settings.postgres_dsn:
            return False
        try:
            import psycopg  # noqa: F401
            return True
        except ImportError:
            try:
                import psycopg2  # noqa: F401
                return True
            except ImportError:
                return False

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        sql = _extract_sql(task)
        if not self.is_live():
            statements = len([s for s in sql.split(";") if s.strip()])
            return f"[simulated:postgres] would execute {statements} statement(s)"
        if not sql.strip():
            return "[postgres] no SQL found in payload; nothing executed"
        try:
            try:
                import psycopg as driver  # type: ignore
            except ImportError:
                import psycopg2 as driver  # type: ignore
            with driver.connect(settings.postgres_dsn) as conn:  # type: ignore[arg-type]
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"postgres execution failed: {exc}") from exc
        return f"[postgres] executed {len([s for s in sql.split(';') if s.strip()])} statement(s)"


class PostgresCLITool(Tool):
    name = "postgres_cli"
    capability = "database"
    description = "Run SQL through the psql CLI"
    fallbacks = ["sqlite_local"]
    side_effect = True
    irreversible = True  # same as the driver path

    def is_live(self) -> bool:
        return shutil.which("psql") is not None and bool(settings.postgres_dsn)

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        sql = _extract_sql(task)
        if not self.is_live():
            return "[simulated:postgres_cli] would pipe SQL to psql"
        try:
            proc = subprocess.run(
                ["psql", str(settings.postgres_dsn), "-v", "ON_ERROR_STOP=1", "-c", sql],
                capture_output=True, text=True, timeout=120, check=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"psql failed: {exc}") from exc
        if proc.returncode != 0:
            raise ToolError(f"psql exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        return f"[postgres_cli] {proc.stdout.strip()[:300]}"


class SQLiteLocalTool(Tool):
    """Always-live terminal fallback for the ``database`` chain."""

    name = "sqlite_local"
    capability = "database"
    description = "Apply SQL to a local SQLite scratch database"
    fallbacks: List[str] = []
    side_effect = True
    irreversible = False  # a local scratch file, deletable

    def __init__(self, db_path: Optional[str] = None):
        super().__init__()
        self.db_path = db_path or str(PROJECT_ROOT / "artifacts" / "scratch.sqlite3")

    def is_live(self) -> bool:
        return True

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        sql = _extract_sql(task)
        if not sql.strip():
            return "[sqlite_local] no SQL found in payload; nothing executed"
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # PostgreSQL-isms SQLite will reject; translated so the fallback is
        # actually usable for the common "create some tables" case.
        translated = (
            sql.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            .replace("BIGSERIAL", "INTEGER")
            .replace("TIMESTAMPTZ", "TEXT")
            .replace("JSONB", "TEXT")
            .replace("UUID", "TEXT")
        )
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(translated)
        except sqlite3.Error as exc:
            raise ToolError(f"sqlite execution failed: {exc}") from exc
        return f"[sqlite_local] applied schema to {Path(self.db_path).name}"


def _extract_sql(task: str) -> str:
    """Pull SQL out of a fenced code block, or fall back to the raw payload."""
    blocks = re.findall(r"```(?:sql)?\s*(.*?)```", task, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        if re.search(r"\b(CREATE|ALTER|INSERT|UPDATE|DELETE|DROP)\b", block, re.IGNORECASE):
            return block.strip()
    if re.search(r"\b(CREATE TABLE|ALTER TABLE|INSERT INTO)\b", task, re.IGNORECASE):
        return task
    return ""


# ---------------------------------------------------------------------------
# capability: ci
# ---------------------------------------------------------------------------


class CITool(Tool):
    """Trigger a GitHub Actions workflow. Live with a token + repo."""

    name = "ci"
    capability = "ci"
    description = "Dispatch a CI workflow run"
    fallbacks = ["local_test_runner"]
    side_effect = True
    irreversible = True  # a dispatched pipeline may deploy

    def is_live(self) -> bool:
        return bool(settings.github_token and settings.github_repo)

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        workflow = directive.get("workflow", "ci.yml")
        if not self.is_live():
            return f"[simulated:ci] would dispatch workflow '{workflow}'"
        _http(
            "POST",
            f"https://api.github.com/repos/{settings.github_repo}/actions/workflows/{workflow}/dispatches",
            headers={
                "Authorization": f"Bearer {settings.github_token}",
                "Accept": "application/vnd.github+json",
            },
            json_body={"ref": directive.get("ref", "main")},
        )
        return f"[ci] dispatched '{workflow}'"


class LocalTestRunnerTool(Tool):
    """Run the project's own test command.

    Deliberately opt-in: only executes when ``ORCHESTRATOR_TEST_CMD`` is set,
    so an agent can never trigger arbitrary local execution by accident.
    """

    name = "local_test_runner"
    capability = "ci"
    description = "Run $ORCHESTRATOR_TEST_CMD locally"
    fallbacks: List[str] = []
    side_effect = True
    irreversible = False  # runs a command, but an opt-in local one

    def is_live(self) -> bool:
        return bool(os.environ.get("ORCHESTRATOR_TEST_CMD"))

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        cmd = os.environ.get("ORCHESTRATOR_TEST_CMD")
        if not cmd:
            return "[simulated:local_test_runner] set ORCHESTRATOR_TEST_CMD to run tests for real"
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=600, check=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"test command failed to start: {exc}") from exc
        tail = (proc.stdout or proc.stderr).strip().splitlines()[-5:]
        verdict = "passed" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        return f"[local_test_runner] {verdict}\n" + "\n".join(tail)


# ---------------------------------------------------------------------------
# capability: notify
# ---------------------------------------------------------------------------


class SlackTool(Tool):
    name = "slack"
    capability = "notify"
    description = "Post a message to a Slack incoming webhook"
    fallbacks = ["email", "console_notify"]
    side_effect = True
    irreversible = True  # a posted message is seen immediately

    def is_live(self) -> bool:
        return bool(settings.slack_webhook)

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        text = task.strip()[:3000]
        inputs = (context or {}).get("inputs") or {}
        channel = inputs.get("channel") or _directive(task).get("channel")
        where = f" to {channel}" if channel else ""

        if not self.is_live():
            return f"[simulated:slack] would post {len(text)} chars{where}"
        body: Dict[str, Any] = {"text": text}
        if channel:
            body["channel"] = channel
        _http("POST", str(settings.slack_webhook), json_body=body)
        return f"[slack] message posted{where}"


class EmailTool(Tool):
    name = "email"
    capability = "notify"
    description = "Send an email via SendGrid"
    #: Prefer a configured Gmail over simulating: the router should reach a
    #: tool that can really send before falling through to stdout.
    fallbacks = ["gmail", "console_notify"]
    side_effect = True
    irreversible = True  # an email cannot be unsent

    def is_live(self) -> bool:
        return bool(settings.sendgrid_api_key and os.environ.get("EMAIL_FROM"))

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        inputs = (context or {}).get("inputs") or {}
        # An operator-supplied recipient wins over the environment default.
        recipient = (inputs.get("recipient") or inputs.get("to")
                     or inputs.get("notify_email") or directive.get("to")
                     or os.environ.get("EMAIL_TO"))
        subject = (inputs.get("subject") or directive.get("subject")
                   or (context or {}).get("step_name") or "Workflow update")

        if not self.is_live() or not recipient:
            return f"[simulated:email] would send '{subject}' to {recipient or '(no recipient)'}"
        _http(
            "POST",
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {settings.sendgrid_api_key}"},
            json_body={
                "personalizations": [{"to": [{"email": recipient}]}],
                "from": {"email": os.environ.get("EMAIL_FROM", recipient)},
                "subject": subject,
                "content": [{"type": "text/plain", "value": task.split("TOOL_DIRECTIVE:", 1)[0].strip()[:8000]}],
            },
        )
        return f"[email] sent '{subject}' to {recipient}"


class GmailTool(Tool):
    """Send real email through Gmail SMTP using an app password.

    Chosen over the Gmail API deliberately: the API needs an OAuth consent
    screen, a client secret and a browser round-trip, which is a wall for a
    user who just wants to send a message. An app password is two clicks in
    a Google account page and works with the standard library.
    """

    name = "gmail"
    capability = "notify"
    description = "Send email from a Gmail account via SMTP"
    fallbacks = ["email", "console_notify"]
    side_effect = True
    irreversible = True  # an email cannot be unsent

    def is_live(self) -> bool:
        return bool(os.environ.get("GMAIL_ADDRESS") and os.environ.get("GMAIL_APP_PASSWORD"))

    def prompt_hint(self) -> str:
        return ("Send an email. End your response with:\n"
                'TOOL_DIRECTIVE: {"to": "someone@example.com", "subject": "..."}\n'
                "The body is everything above that line.")

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        inputs = (context or {}).get("inputs") or {}
        to = inputs.get("recipient") or inputs.get("to") or directive.get("to") or "(no recipient)"
        subject = inputs.get("subject") or directive.get("subject") or "(no subject)"
        mode = "REAL" if self.is_live() else "simulated"
        return f"gmail [{mode}]: send '{subject}' to {to}"

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        inputs = (context or {}).get("inputs") or {}
        recipient = (inputs.get("recipient") or inputs.get("to") or directive.get("to")
                     or os.environ.get("EMAIL_TO"))
        subject = (inputs.get("subject") or directive.get("subject")
                   or (context or {}).get("step_name") or "Workflow update")
        body = re.sub(r"TOOL_DIRECTIVE:.*$", "", task, flags=re.MULTILINE | re.DOTALL).strip()

        if not recipient:
            return "[simulated:gmail] no recipient given; nothing sent"
        if not self.is_live():
            return (f"[simulated:gmail] would send '{subject}' to {recipient} "
                    "(set GMAIL_ADDRESS and GMAIL_APP_PASSWORD to send for real)")

        import smtplib
        from email.message import EmailMessage

        sender = os.environ["GMAIL_ADDRESS"]
        # Google shows app passwords in spaced groups; SMTP wants them joined.
        password = os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")

        message = EmailMessage()
        message["From"] = sender
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body or "(no content)")

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
                smtp.login(sender, password)
                smtp.send_message(message)
        except smtplib.SMTPAuthenticationError as exc:
            raise ToolError(
                "Gmail rejected the login. Check GMAIL_APP_PASSWORD is an *app password* "
                "(not your normal Gmail password) and that 2-Step Verification is on. "
                f"({exc.smtp_code})") from exc
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"could not send via Gmail: {exc}") from exc

        return f"[gmail] sent '{subject}' to {recipient}"


class ConsoleNotifyTool(Tool):
    """Always-live terminal fallback for the ``notify`` chain."""

    name = "console_notify"
    capability = "notify"
    description = "Print the notification to stdout"
    fallbacks: List[str] = []
    side_effect = False
    irreversible = False  # prints to stdout

    def is_live(self) -> bool:
        return True

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        first_line = task.strip().splitlines()[0][:200] if task.strip() else "(empty)"
        print(f"[notify] {first_line}")
        return f"[console_notify] {first_line}"


# ---------------------------------------------------------------------------
# capability: http
# ---------------------------------------------------------------------------


class RestApiTool(Tool):
    """Generic HTTP call, driven entirely by a TOOL_DIRECTIVE block."""

    name = "rest_api"
    capability = "http"
    description = "Generic HTTP request from a TOOL_DIRECTIVE payload"
    fallbacks: List[str] = []
    side_effect = True
    irreversible = True  # an arbitrary HTTP call could do anything

    def is_live(self) -> bool:
        return True

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        url = directive.get("url")
        if not url:
            return "[rest_api] no url in TOOL_DIRECTIVE; nothing called"
        response = _http(
            directive.get("method", "GET").upper(),
            url,
            headers=directive.get("headers"),
            json_body=directive.get("body"),
        )
        return f"[rest_api] {response.status_code} {url} -> {response.text[:300]}"


# ---------------------------------------------------------------------------
# dedicated production connectors
# ---------------------------------------------------------------------------


class StripeTool(Tool):
    """Stripe REST adapter for common read and create operations."""

    name = "stripe"
    capability = "payments"
    description = "Read Stripe objects or create customers, invoices and payment links"
    fallbacks: List[str] = []
    side_effect = True

    def is_live(self) -> bool:
        return bool(os.environ.get("STRIPE_API_KEY"))

    def prompt_hint(self) -> str:
        return ('Use TOOL_DIRECTIVE JSON with action "list_customers", "get_customer", '
                '"create_customer", or "create_payment_link". Include params and, when '
                'needed, id. Example: TOOL_DIRECTIVE: {"action":"list_customers",'
                '"params":{"limit":10}}')

    def is_irreversible(self, task: str, context: Optional[dict] = None) -> bool:
        return str(_directive(task).get("action", "")).startswith("create_")

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        action = directive.get("action", "list_customers")
        routes = {
            "list_customers": ("GET", "/customers"),
            "get_customer": ("GET", f"/customers/{directive.get('id', '')}"),
            "create_customer": ("POST", "/customers"),
            "create_payment_link": ("POST", "/payment_links"),
        }
        if action not in routes:
            raise ToolError(f"unsupported stripe action: {action}")
        if not self.is_live():
            return f"[simulated:stripe] would {action}"
        method, path = routes[action]
        params = directive.get("params") or {}
        url = f"https://api.stripe.com/v1{path}"
        headers = {"Authorization": f"Bearer {os.environ['STRIPE_API_KEY']}"}
        try:
            import requests
            response = requests.request(method, url, headers=headers,
                                        params=params if method == "GET" else None,
                                        data=params if method == "POST" else None, timeout=20)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Stripe request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ToolError(f"Stripe HTTP {response.status_code}: {response.text[:300]}")
        return f"[stripe] {action} succeeded: {response.text[:500]}"


class S3Tool(Tool):
    """Amazon S3 adapter. boto3 is loaded only when the connector is used."""

    name = "s3"
    capability = "object_storage"
    description = "List, read, upload, and delete objects in an Amazon S3 bucket"
    fallbacks = ["artifact_store"]
    side_effect = True

    def is_live(self) -> bool:
        return bool(os.environ.get("AWS_S3_BUCKET") and
                    (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")))

    def prompt_hint(self) -> str:
        return ('Use TOOL_DIRECTIVE JSON with action "list", "get", "put", or "delete"; '
                'include key and content for put.')

    def is_irreversible(self, task: str, context: Optional[dict] = None) -> bool:
        return _directive(task).get("action") in {"put", "delete"}

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        directive = _directive(task)
        action = directive.get("action", "list")
        if action not in {"list", "get", "put", "delete"}:
            raise ToolError(f"unsupported s3 action: {action}")
        if not self.is_live():
            return f"[simulated:s3] would {action} in {os.environ.get('AWS_S3_BUCKET', '(bucket)')}"
        try:
            import boto3
        except ImportError as exc:
            raise ToolError("boto3 is required for the live S3 connector") from exc
        bucket = os.environ["AWS_S3_BUCKET"]
        client = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
        key = directive.get("key", "")
        try:
            if action == "list":
                result = client.list_objects_v2(Bucket=bucket,
                                                Prefix=directive.get("prefix", ""), MaxKeys=100)
                keys = [item["Key"] for item in result.get("Contents", [])]
                return f"[s3] {len(keys)} object(s): {json.dumps(keys)}"
            if not key:
                raise ToolError("s3 action requires 'key'")
            if action == "get":
                body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
                return f"[s3] {key}: {body.decode('utf-8', errors='replace')[:2000]}"
            if action == "put":
                content = directive.get("content", task.split("TOOL_DIRECTIVE:", 1)[0].strip())
                client.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
                return f"[s3] uploaded s3://{bucket}/{key}"
            client.delete_object(Bucket=bucket, Key=key)
            return f"[s3] deleted s3://{bucket}/{key}"
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"S3 {action} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# default registry
# ---------------------------------------------------------------------------

#: Aliases so a plan that says "postgres_mcp" or "github_mcp" still routes.
TOOL_ALIASES = {
    "github_mcp": "github",
    "postgres_mcp": "postgres",
    "ci_tool": "ci",
    "slack_mcp": "slack",
    "db": "postgres",
    "vcs": "github",
    # Names models reliably invent for capabilities that do exist. Without
    # these the step plans a tool that cannot be called and fails at runtime.
    "email_service": "email", "mailer": "email", "mail": "email",
    "smtp": "gmail", "email_client": "email", "mail_service": "email",
    "database_service": "postgres", "db_service": "postgres", "sql": "postgres",
    "database": "postgres", "version_control": "github", "git": "github",
    "messaging": "slack", "chat": "slack", "notification": "slack",
    "http": "rest_api", "api": "rest_api", "web_request": "rest_api",
    "payments": "stripe", "stripe_api": "stripe",
    "aws_s3": "s3", "object_storage": "s3", "storage": "s3",
    "sendgrid": "email", "sendgrid_email": "email",
    "shell": "terminal", "bash": "terminal", "command_line": "terminal",
    "filesystem": "workspace", "file_system": "workspace", "files": "workspace",
    "desktop": "hermes_desktop", "desktop_automation": "hermes_desktop",
    # computer_use is its own tool now, not an alias for the desktop backend:
    # it picks between browser and desktop and enforces the plan, allow-list
    # and limits that the raw backends do not.
    "computer-use": "computer_use", "computer": "computer_use",
    "screen": "computer_use", "browser_use": "computer_use",
    "hermes": "hermes_desktop",
}


def default_tool_manager() -> ToolManager:
    """A ToolManager with every built-in adapter registered."""
    from .computer_task import ComputerTaskTool
    from .computer_use import ComputerUseTool
    from .desktop import DesktopTool
    from .hermes import HermesDesktopTool

    manager = ToolManager([
        ComputerTaskTool(), ComputerUseTool(), DesktopTool(),
        GitHubTool(), GitHubCLITool(), ArtifactStoreTool(),
        PostgresTool(), PostgresCLITool(), SQLiteLocalTool(),
        CITool(), LocalTestRunnerTool(),
        SlackTool(), GmailTool(), EmailTool(), ConsoleNotifyTool(),
        RestApiTool(), StripeTool(), S3Tool(), HermesDesktopTool(),
    ])
    return manager


def canonical_tool_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower()
    return TOOL_ALIASES.get(key, key)

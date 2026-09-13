"""Guided setup: turn "SENDGRID_API_KEY is missing" into something a person can act on.

The requirements layer can tell you *that* a credential is missing. That is a
developer's answer. A user needs to know what the thing is, why this workflow
wants it, where to click to get one, and whether what they pasted is even the
right shape.

This module holds that knowledge -- a catalogue of every credential the system
understands, each with plain-language guidance -- plus the ability to save an
answer to ``.env.local`` and make it live without a restart.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import PROJECT_ROOT, reload_settings

ENV_FILE = PROJECT_ROOT / ".env.local"


@dataclass
class CredentialGuide:
    """Everything a person needs in order to supply one credential."""

    env_var: str
    #: What a non-developer would call it.
    label: str
    #: One line: what it is.
    description: str = ""
    #: Where to click to get one.
    url: str = ""
    #: Ordered, concrete instructions.
    steps: List[str] = field(default_factory=list)
    #: Shown greyed in the input box.
    placeholder: str = ""
    #: Regex the value must match. None = accept anything non-empty.
    pattern: Optional[str] = None
    #: Human explanation of the expected shape, shown when validation fails.
    format_hint: str = ""
    #: True for anything that should never be echoed back.
    secret: bool = True
    #: Free-tier / cost note, so nobody is surprised by a bill.
    cost_note: str = ""

    def validate(self, value: str) -> Optional[str]:
        """Return an error message, or None if the value looks right."""
        value = (value or "").strip()
        if not value:
            return f"{self.label} cannot be empty"
        if self.pattern and not re.match(self.pattern, value):
            return self.format_hint or f"That does not look like a valid {self.label}"
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "env_var": self.env_var, "label": self.label,
            "description": self.description, "url": self.url,
            "steps": self.steps, "placeholder": self.placeholder,
            "format_hint": self.format_hint, "secret": self.secret,
            "cost_note": self.cost_note,
            "is_set": bool(os.environ.get(self.env_var)),
        }


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

CREDENTIAL_GUIDES: Dict[str, CredentialGuide] = {g.env_var: g for g in [
    # -- LLM providers --------------------------------------------------
    CredentialGuide(
        env_var="GEMINI_API_KEY", label="Google Gemini API key",
        description="Lets the agents actually think. Without any AI key the system "
                    "runs on canned placeholder text.",
        url="https://aistudio.google.com/apikey",
        steps=["Open Google AI Studio (link below) and sign in",
               "Click 'Create API key'",
               "Copy the key and paste it here"],
        placeholder="AIza...", pattern=r"^AIza[\w\-]{30,}$",
        format_hint="Gemini keys start with 'AIza' and are about 39 characters.",
        cost_note="Free tier: 20 requests/day. Paid removes the limit."),

    CredentialGuide(
        env_var="ANTHROPIC_API_KEY", label="Anthropic (Claude) API key",
        description="Use Claude to power the agents.",
        url="https://console.anthropic.com/settings/keys",
        steps=["Open the Anthropic Console and sign in",
               "Go to Settings -> API keys -> Create key",
               "Copy it and paste it here"],
        placeholder="sk-ant-...", pattern=r"^sk-ant-[\w\-]{20,}$",
        format_hint="Anthropic keys start with 'sk-ant-'.",
        cost_note="Pay as you go; no free tier."),

    CredentialGuide(
        env_var="OPENAI_API_KEY", label="OpenAI API key",
        description="Use GPT models to power the agents.",
        url="https://platform.openai.com/api-keys",
        steps=["Open the OpenAI platform and sign in",
               "Click 'Create new secret key'",
               "Copy it and paste it here"],
        placeholder="sk-...", pattern=r"^sk-[\w\-]{20,}$",
        format_hint="OpenAI keys start with 'sk-'.",
        cost_note="Pay as you go; free trial credit for new accounts."),

    CredentialGuide(
        env_var="NVIDIA_API_KEY", label="NVIDIA NIM API key",
        description="Access 100+ hosted open models (Llama, Mistral, Nemotron, Qwen) "
                    "with a generous free tier -- a good fallback when another "
                    "provider hits its daily limit.",
        url="https://build.nvidia.com/",
        steps=["Open build.nvidia.com and sign in (free account)",
               "Pick any model, e.g. Llama 3.1 8B Instruct",
               "Click 'Get API Key' and copy it here"],
        placeholder="nvapi-...", pattern=r"^nvapi-[\w\-]{20,}$",
        format_hint="NVIDIA keys start with 'nvapi-'.",
        cost_note="Free tier with credits; no card required to start."),

    # -- Email ----------------------------------------------------------
    CredentialGuide(
        env_var="GMAIL_ADDRESS", label="Your Gmail address",
        description="The account emails will be sent FROM.",
        placeholder="you@gmail.com",
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        format_hint="Enter a full email address, e.g. you@gmail.com.",
        secret=False),

    CredentialGuide(
        env_var="GMAIL_APP_PASSWORD", label="Gmail app password",
        description="A 16-character password just for this app. NOT your normal "
                    "Gmail password -- Google requires a separate one for programs.",
        url="https://myaccount.google.com/apppasswords",
        steps=["Turn on 2-Step Verification on your Google account (required first)",
               "Open the App passwords page (link below)",
               "Type any name, e.g. 'Orchestrator', and click Create",
               "Google shows a 16-letter password -- copy it here (spaces are fine)"],
        placeholder="abcd efgh ijkl mnop",
        pattern=r"^[a-zA-Z]{4}[\s]?[a-zA-Z]{4}[\s]?[a-zA-Z]{4}[\s]?[a-zA-Z]{4}$",
        format_hint="An app password is 16 letters, usually shown as 4 groups of 4.",
        cost_note="Free."),

    CredentialGuide(
        env_var="EMAIL_TO", label="Default recipient address",
        description="Where email goes if a workflow does not specify a recipient.",
        placeholder="someone@example.com",
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        format_hint="Enter a full email address.", secret=False),

    CredentialGuide(
        env_var="SENDGRID_API_KEY", label="SendGrid API key",
        description="An alternative to Gmail for sending email at volume.",
        url="https://app.sendgrid.com/settings/api_keys",
        steps=["Sign in to SendGrid",
               "Settings -> API Keys -> Create API Key",
               "Give it 'Mail Send' permission, then copy it here"],
        placeholder="SG....", pattern=r"^SG\.[\w\-]{10,}",
        format_hint="SendGrid keys start with 'SG.'.",
        cost_note="Free tier: 100 emails/day."),

    CredentialGuide(
        env_var="STRIPE_API_KEY", label="Stripe restricted API key",
        description="Lets payment workflows read Stripe data and perform approved actions.",
        url="https://dashboard.stripe.com/apikeys",
        steps=["Open Stripe Dashboard -> Developers -> API keys",
               "Create a restricted key with only the permissions this workflow needs",
               "Use a test-mode key until the workflow is validated"],
        placeholder="rk_test_...", pattern=r"^(rk|sk)_(test|live)_.+",
        format_hint="Prefer a restricted rk_test_ key during development.",
        cost_note="Stripe API access has no separate fee; transactions have processing fees."),

    CredentialGuide(
        env_var="AWS_S3_BUCKET", label="Amazon S3 bucket",
        description="Bucket used by file-storage workflow steps.",
        placeholder="my-workflow-bucket", pattern=r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$",
        format_hint="Enter the bucket name, without s3://.", secret=False),

    CredentialGuide(
        env_var="AWS_ACCESS_KEY_ID", label="AWS access key ID",
        description="Identity for the S3 connector. Use an IAM role in deployed environments.",
        url="https://console.aws.amazon.com/iam/home#/security_credentials",
        steps=["Create an IAM policy limited to the required S3 bucket",
               "Create an access key for local development only",
               "Use a workload IAM role in production"],
        placeholder="AKIA...", pattern=r"^[A-Z0-9]{16,128}$"),

    CredentialGuide(
        env_var="AWS_SECRET_ACCESS_KEY", label="AWS secret access key",
        description="Secret paired with the AWS access key ID.",
        placeholder="AWS secret", pattern=r"^.{20,}$"),

    # -- Code / repos ---------------------------------------------------
    CredentialGuide(
        env_var="GITHUB_TOKEN", label="GitHub personal access token",
        description="Lets agents create issues, commit files and open pull requests.",
        url="https://github.com/settings/tokens",
        steps=["Open GitHub -> Settings -> Developer settings -> Personal access tokens",
               "Click 'Generate new token (classic)'",
               "Tick the 'repo' scope",
               "Generate, then copy the token here (GitHub shows it only once)"],
        placeholder="ghp_...", pattern=r"^(ghp_|github_pat_)[\w]{20,}$",
        format_hint="GitHub tokens start with 'ghp_' or 'github_pat_'.",
        cost_note="Free."),

    CredentialGuide(
        env_var="GITHUB_REPO", label="Target GitHub repository",
        description="Which repository the agents should work in.",
        placeholder="your-username/your-repo",
        pattern=r"^[\w.\-]+/[\w.\-]+$",
        format_hint="Use the owner/repository form, e.g. octocat/hello-world.",
        secret=False),

    # -- Messaging ------------------------------------------------------
    CredentialGuide(
        env_var="SLACK_WEBHOOK_URL", label="Slack incoming webhook",
        description="Lets agents post messages into a Slack channel.",
        url="https://api.slack.com/messaging/webhooks",
        steps=["Create a Slack app at api.slack.com/apps",
               "Enable 'Incoming Webhooks'",
               "Click 'Add New Webhook to Workspace' and pick a channel",
               "Copy the webhook URL here"],
        placeholder="https://hooks.slack.com/services/...",
        pattern=r"^https://hooks\.slack\.com/services/.+",
        format_hint="It should start with https://hooks.slack.com/services/.",
        cost_note="Free."),

    # -- Database -------------------------------------------------------
    CredentialGuide(
        env_var="DATABASE_URL", label="PostgreSQL connection string",
        description="Lets agents create tables and run migrations for real. "
                    "Without it, schema work is written but applied to a local "
                    "SQLite file instead.",
        placeholder="postgresql://user:password@localhost:5432/dbname",
        pattern=r"^postgres(ql)?://.+",
        format_hint="Should start with postgresql:// or postgres://.",
        cost_note="Free if you run Postgres locally."),

    # -- Desktop automation --------------------------------------------
    CredentialGuide(
        env_var="ORCHESTRATOR_ALLOW_DESKTOP", label="Enable Hermes desktop automation",
        description="Allows approved workflow steps to ask a locally installed Hermes "
                    "Agent to operate desktop applications.",
        url="https://hermes-agent.nousresearch.com/docs/getting-started/installation",
        steps=["Install Hermes Agent for Windows using the official installer",
               "Run 'hermes setup' and configure its model provider",
               "Run 'hermes computer-use install'",
               "Run 'hermes computer-use doctor' and resolve reported problems",
               "Enter true here to enable desktop steps in this orchestrator"],
        placeholder="true", pattern=r"^(1|true|yes|on)$",
        format_hint="Enter true after Hermes and its computer-use driver are ready.",
        secret=False, cost_note="Hermes is MIT licensed; model usage may have a cost."),

    CredentialGuide(
        env_var="ORCHESTRATOR_ALLOW_TERMINAL", label="Enable the restricted project terminal",
        description="Allows approved coding workflow steps to run build and test commands "
                    "inside their own workspace without administrator privileges.",
        steps=["Review the generated workflow and its terminal steps",
               "Enter true to allow commands inside workspace/<run-id>",
               "Approve each real terminal action when the workflow pauses"],
        placeholder="true", pattern=r"^(1|true|yes|on)$",
        format_hint="Enter true to opt in. Leave empty to keep command execution simulated.",
        secret=False, cost_note="Runs locally; no API cost."),
]}


def guide_for(env_var: str) -> CredentialGuide:
    """The guide for a variable, or a generic one if it is not catalogued."""
    known = CREDENTIAL_GUIDES.get(env_var)
    if known:
        return known
    return CredentialGuide(
        env_var=env_var,
        label=env_var.replace("_", " ").title(),
        description=f"Required by this workflow. Set the {env_var} environment variable.",
        secret="KEY" in env_var or "TOKEN" in env_var or "SECRET" in env_var
               or "PASSWORD" in env_var)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _read_env_file() -> List[str]:
    if not ENV_FILE.exists():
        return []
    return ENV_FILE.read_text(encoding="utf-8").splitlines()


def save_credential(env_var: str, value: str, validate: bool = True) -> None:
    """Write one credential to ``.env.local`` and make it live immediately.

    Updates the line in place if the variable is already there, so the file
    never accumulates duplicates. Also sets ``os.environ`` and refreshes the
    settings singleton, so a tool that was simulated a moment ago starts
    working without restarting the server.
    """
    value = (value or "").strip()
    if validate:
        error = guide_for(env_var).validate(value)
        if error:
            raise ValueError(error)

    lines = _read_env_file()
    pattern = re.compile(rf"^\s*{re.escape(env_var)}\s*=")
    replaced = False
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = f"{env_var}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{env_var}={value}")

    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        # Owner-only, so a pasted API key is not world-readable. Best effort:
        # on Windows this is largely a no-op, which is why it is not relied on.
        ENV_FILE.chmod(0o600)
    except OSError:
        pass

    os.environ[env_var] = value
    reload_settings()


def clear_credential(env_var: str) -> None:
    """Remove a credential from ``.env.local`` and the current process."""
    pattern = re.compile(rf"^\s*{re.escape(env_var)}\s*=")
    lines = [line for line in _read_env_file() if not pattern.match(line)]
    ENV_FILE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.environ.pop(env_var, None)
    reload_settings()


# ---------------------------------------------------------------------------
# What still needs answering
# ---------------------------------------------------------------------------


def setup_questions(report: object) -> List[Dict[str, object]]:
    """Turn a RequirementsReport into an ordered list of guided questions.

    Blockers come first (they stop the run), then things that merely degrade
    it. Each entry carries the full guidance the UI needs to explain itself.
    """
    from .requirements import RequirementKind, RequirementStatus

    questions: List[Dict[str, object]] = []
    seen: set = set()

    ordered = (getattr(report, "blockers", []) + getattr(report, "degraded", []))
    for requirement in ordered:
        if requirement.kind is not RequirementKind.CREDENTIAL:
            continue
        if requirement.name in seen or os.environ.get(requirement.name):
            continue
        seen.add(requirement.name)
        guide = guide_for(requirement.name)
        questions.append({
            **guide.to_dict(),
            "why_this_workflow": requirement.why,
            "blocking": requirement.status is RequirementStatus.MISSING,
        })
    return questions

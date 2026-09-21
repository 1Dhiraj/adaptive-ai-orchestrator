"""Real email delivery with explicit API/browser selection and exact approval.

Browser access opens Gmail first, lets the person sign in themselves, then
prepares a draft. Sending is a separate workflow approval. Ambiguous delivery
is never retried automatically, including after process restart.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .base import ActionReviewRequiredError, Tool, ToolError
from .builtin import EmailTool, GmailTool


def email_message(payload: str, context: dict | None = None) -> dict:
    # Agents occasionally emit both the legacy flat directive and the newer
    # {"arguments": ...} form.  Parse each directive line independently and
    # use the last valid one; the shared greedy parser cannot represent two
    # adjacent JSON objects and previously fell back to "Project update".
    directive: dict = {}
    directive_line: int | None = None
    lines = (payload or "").splitlines()
    for line_number, line in enumerate(lines):
        if not line.lstrip().startswith("TOOL_DIRECTIVE:"):
            continue
        try:
            candidate = json.loads(line.split("TOOL_DIRECTIVE:", 1)[1].strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(candidate, dict):
            nested = candidate.get("arguments")
            directive = nested if isinstance(nested, dict) else candidate
            directive_line = line_number
    values = (context or {}).get("inputs") or {}
    recipient = str(values.get("recipient") or values.get("to") or directive.get("to") or "").strip()
    subject = str(values.get("subject") or directive.get("subject") or "Project update").strip()
    explicit_body = values.get("body") or values.get("email_body") or directive.get("body")
    if explicit_body is not None:
        body = str(explicit_body).strip()
    else:
        # Use content before the *last valid* directive.  A reasoning trace can
        # mention TOOL_DIRECTIVE verbatim, so splitting on its first occurrence
        # truncates the actual message and can leak model reasoning into Gmail.
        body = "\n".join(lines[:directive_line] if directive_line is not None else lines).strip()
        if "</think>" in body:
            body = body.rsplit("</think>", 1)[1].strip()
        body = re.sub(r"(?is)<think>.*?</think>", "", body).strip()
    # ASSUMPTIONS is model-to-orchestrator metadata, never message content.
    body = re.sub(r"(?m)^ASSUMPTIONS:\s*.*(?:\r?\n)?", "", body).strip()
    if not re.fullmatch(r"[^\s@,;<>]+@[^\s@,;<>]+\.[^\s@,;<>]+", recipient):
        raise ToolError("Provide one valid recipient before sending")
    if any(c in subject for c in "\r\n") or not body or len(body) > 8000 or len(subject) > 200:
        raise ToolError("Email needs a subject and a body of at most 8,000 characters")
    return {"to": recipient, "subject": subject, "body": body}


def message_digest(message: dict) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()


class EmailDelivery:
    def __init__(self, root: Path, browser_tool: Tool | None = None):
        self.root = root
        self.browser_tool = browser_tool
        self.lock = threading.RLock()

    def _path(self, run_id: str, step_id: str) -> Path:
        return self.root / (hashlib.sha256(f"{run_id}:{step_id}".encode()).hexdigest() + ".json")

    def read(self, run_id: str, step_id: str) -> dict:
        path = self._path(run_id, step_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "choose_method"}

    def write(self, run_id: str, step_id: str, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(run_id, step_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def api_tool() -> Tool | None:
        for tool in (GmailTool(), EmailTool()):
            if tool.is_live():
                return tool
        return None

    def status(self, run_id: str, step_id: str, message: dict) -> dict:
        record = self.read(run_id, step_id)
        api = self.api_tool()
        digest = message_digest(message)
        reviewed_digest = record.get("digest")
        stale = bool(reviewed_digest and reviewed_digest != digest)
        effective_status = "choose_method" if stale else record.get("status", "choose_method")
        return {**record, "status": effective_status, "message": message, "digest": digest,
                "reviewed_digest": reviewed_digest, "stale_draft": stale,
                "api_available": api is not None, "api_name": api.name if api else None,
                "browser_available": self.browser_tool is not None}

    def _browser(self, code: str) -> dict:
        if self.browser_tool is None:
            raise ToolError("Connect the Playwright browser tool first")
        output = self.browser_tool.execute("TOOL_DIRECTIVE: " + json.dumps({"arguments": {"code": code}}))
        # Playwright MCP wraps return values under a '### Result' heading.
        text = output.split("### Result", 1)[-1].strip()
        try:
            result, _ = json.JSONDecoder().raw_decode(text)
        except (ValueError, TypeError) as exc:
            raise ToolError("Browser did not return a verifiable result; nothing will be retried automatically") from exc
        if not isinstance(result, dict) or "status" not in result:
            raise ToolError("Browser result was incomplete")
        return result

    def open_browser(self, run_id: str, step_id: str, message: dict, approved: bool) -> dict:
        if approved is not True:
            raise ToolError("Permission to open Gmail is required")
        with self.lock:
            record = self.read(run_id, step_id)
            if record["status"] in {"sending", "sent", "delivery_unknown"}:
                raise ToolError("This message was already submitted; verify delivery before starting another email task")
            result = self._browser('''async (page) => {
              await page.goto('https://mail.google.com/mail/u/0/', {waitUntil:'domcontentloaded'});
              return {status:'sign_in_needed'};
            }''')
            if result["status"] != "sign_in_needed":
                raise ToolError("Gmail could not be opened")
            record = {"status": "sign_in_needed", "method": "browser", "digest": message_digest(message)}
            self.write(run_id, step_id, record)
            return record

    def prepare_browser(self, run_id: str, step_id: str, message: dict) -> dict:
        with self.lock:
            record = self.read(run_id, step_id)
            if record["status"] != "sign_in_needed" or record.get("digest") != message_digest(message):
                raise ToolError("Approve browser access for this exact message first")
            data = json.dumps(message)
            result = self._browser('''async (page) => {
              if (!page.url().startsWith('https://mail.google.com/mail/')) return {status:'sign_in_needed'};
              const message = ''' + data + ''';
              const compose = page.getByRole('button', {name:'Compose', exact:true});
              if (await compose.count() !== 1) return {status:'sign_in_needed'};
              if (await page.locator('input[name="subjectbox"]').count()) return {status:'existing_draft'};
              await compose.click();
              const dialog = page.locator('div[role="dialog"]').filter({has:page.locator('input[name="subjectbox"]')});
              if (await dialog.count() !== 1) return {status:'unsupported_layout'};
              const to = dialog.locator('input[name="to"],input[aria-label="To recipients"]');
              if (await to.count() !== 1) return {status:'unsupported_layout'};
              await to.fill(message.to); await to.press('Enter');
              await dialog.locator('input[name="subjectbox"]').fill(message.subject);
              await dialog.locator('[contenteditable="true"][role="textbox"]').fill(message.body);
              return {status:'draft_ready'};
            }''')
            if result["status"] != "draft_ready":
                explanations = {"sign_in_needed": "Sign in to Gmail yourself, then try preparing the draft again. The Gmail interface must be in English.",
                                "existing_draft": "A Gmail compose window is already open. Save or close it yourself first.",
                                "unsupported_layout": "Gmail's layout was not recognized. Review the browser manually; nothing was sent."}
                raise ToolError(explanations.get(result["status"], "Could not prepare a Gmail draft"))
            record["status"] = "draft_ready"
            self.write(run_id, step_id, record)
            return record

    def send(self, run_id: str, step_id: str, message: dict) -> str:
        with self.lock:
            record = self.read(run_id, step_id)
            digest = message_digest(message)
            if record["status"] == "sent" and record.get("digest") == digest:
                return "[adaptive_email] Already sent; duplicate delivery prevented."
            if record["status"] in {"sending", "sent", "delivery_unknown"}:
                raise ToolError("Delivery may already have happened. Check Sent mail; automatic retry is blocked.")
            use_browser = record.get("method") == "browser"
            if self.browser_tool is not None and not use_browser:
                raise ToolError("Open Gmail and prepare the reviewed browser draft before sending")
            if use_browser:
                if record["status"] != "draft_ready" or record.get("digest") != digest:
                    raise ToolError("Prepare and review the current Gmail draft before sending")
            elif not self.api_tool():
                raise ToolError("No email API is configured. Add credentials or approve opening Gmail in the browser.")
            # Persist before dispatch so retries/crash recovery cannot send twice.
            record.update(status="sending", digest=digest)
            self.write(run_id, step_id, record)
            try:
                if use_browser:
                    result = self._send_browser(message)
                    if result["status"] == "draft_changed":
                        record["status"] = "draft_ready"
                        self.write(run_id, step_id, record)
                        reason = str(result.get("reason") or "the reviewed fields did not match")
                        raise ActionReviewRequiredError(
                            f"Gmail draft changed or could not be verified ({reason}). "
                            "Nothing was sent; restore the reviewed draft.")
                    if result["status"] != "sent":
                        raise ToolError("Gmail did not confirm delivery. Check Sent mail before doing anything else.")
                else:
                    tool = self.api_tool()
                    payload = message["body"] + '\nTOOL_DIRECTIVE: ' + json.dumps({"to": message["to"], "subject": message["subject"]})
                    output = tool.execute(payload, {"inputs": {"recipient": message["to"], "subject": message["subject"]}})
                    if output.startswith("[simulated:"):
                        raise ToolError("Email was not sent: credentials became unavailable")
                record["status"] = "sent"
                self.write(run_id, step_id, record)
                return f"[adaptive_email] Sent '{message['subject']}' to {message['to']} via {'Gmail browser' if use_browser else tool.name}."
            except Exception:
                if record["status"] == "sending":
                    record["status"] = "delivery_unknown"
                    self.write(run_id, step_id, record)
                raise

    def _send_browser(self, message: dict) -> dict:
        # Re-read the actual compose UI immediately before the single Send click.
        # Never trust model claims or a stale saved draft as delivery evidence.
        return self._browser(r'''async (page) => {
          const expected = ''' + json.dumps(message) + r''';
          if (!page.url().startsWith('https://mail.google.com/mail/')) return {status:'draft_changed', reason:'Gmail is not the active page'};
          const dialog = page.locator('div[role="dialog"]').filter({has:page.locator('input[name="subjectbox"]')});
          const dialogCount = await dialog.count();
          if (dialogCount !== 1) return {status:'draft_changed', reason:`expected one compose window, found ${dialogCount}`};
          const chipRecipients = await dialog.locator('[email], [data-hovercard-id]').evaluateAll(els => els.map(e => e.getAttribute('email') || e.getAttribute('data-hovercard-id')).filter(v => v && v.includes('@')));
          const typedRecipients = await dialog.locator('input[name="to"],input[aria-label="To recipients"]').evaluateAll(els => els.flatMap(e => e.value.split(/[,;]/)).map(v => v.trim()).filter(v => v.includes('@')));
          const recipients = [...new Set([...chipRecipients, ...typedRecipients].map(v => v.toLowerCase()))];
          const subject = await dialog.locator('input[name="subjectbox"]').inputValue();
          const body = await dialog.locator('[contenteditable="true"][role="textbox"]').innerText();
          const extra = await dialog.locator('input[name="cc"],input[name="bcc"]').evaluateAll(els => els.some(e => e.value.trim()));
          const attachments = await dialog.locator('[download_url], [aria-label^="Remove attachment"]').count();
          const mismatches = [];
          if (recipients.length !== 1 || recipients[0] !== expected.to.toLowerCase()) mismatches.push('recipient mismatch');
          if (subject !== expected.subject) mismatches.push('subject mismatch');
          if (body.replace(/\r\n/g,'\n').trim() !== expected.body.replace(/\r\n/g,'\n').trim()) mismatches.push('body mismatch');
          if (extra) mismatches.push('CC or BCC is not empty');
          if (attachments) mismatches.push('unexpected attachment');
          if (mismatches.length) return {status:'draft_changed', reason:mismatches.join(', ')};
          // Gmail renders Send as a div[role=button]. Its accessible label can
          // contain invisible bidi characters around the keyboard shortcut,
          // which makes a strict getByRole name regex incorrectly miss it.
          const buttons = dialog.locator('[role="button"]');
          const sendMatches = [];
          for (let i = 0; i < await buttons.count(); i++) {
            const button = buttons.nth(i);
            if (!await button.isVisible()) continue;
            const text = (await button.innerText().catch(() => '')).trim();
            const label = ((await button.getAttribute('aria-label')) ||
                           (await button.getAttribute('data-tooltip')) || '').trim();
            if (text === 'Send' || (/^Send(?:\s|[()\u202a-\u202e]|$)/.test(label) &&
                                    !/(?:option|schedule)/i.test(label))) {
              sendMatches.push(button);
            }
          }
          if (sendMatches.length !== 1) return {status:'draft_changed', reason:`expected one Send button, found ${sendMatches.length}`};
          const send = sendMatches[0];
          await send.click();
          try { await page.getByText('Message sent', {exact:true}).waitFor({timeout:10000}); }
          catch (_) { return {status:'delivery_unknown'}; }
          return {status:'sent'};
        }''')


class AdaptiveEmailTool(Tool):
    name = "adaptive_email"
    capability = "reviewed_email_delivery"
    description = "Draft and send email in Gmail's browser UI, with a configured API only as fallback. Never simulates delivery."
    side_effect = True
    irreversible = True
    require_approval = True

    def __init__(self, delivery: EmailDelivery, run_id: str = "planning"):
        super().__init__()
        self.delivery, self.run_id = delivery, run_id

    def is_live(self) -> bool:
        return self.delivery.api_tool() is not None or self.delivery.browser_tool is not None

    def prompt_hint(self) -> str:
        return ('Return one TOOL_DIRECTIVE JSON line with arguments containing exactly "to", "subject", and "body", '
                'for example TOOL_DIRECTIVE: {"arguments":{"to":"recipient@example.com","subject":"Subject","body":"Email text"}}. '
                'Use the recipient supplied by the person. Do not ask for API keys: delivery setup and browser permission happen after drafting. Do not claim the email was sent.')

    def preview(self, task: str, context: dict | None = None) -> str:
        message = email_message(task, context)
        return f"Compose '{message['subject']}' to {message['to']} in Gmail, then review it before sending"

    def _run(self, task: str, context: dict | None = None) -> str:
        step_id = (context or {}).get("step_id", "email")
        return self.delivery.send(self.run_id, step_id, email_message(task, context))

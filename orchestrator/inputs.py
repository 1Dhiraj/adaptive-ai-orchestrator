"""Human input: what the workflow needs *from you* to finish the job.

A step can declare inputs it cannot invent — a recipient address, which repo
to push to, a deadline, an approval on wording. When such a step is reached
and an input is still unanswered, the run pauses, reports exactly what it
needs, and waits. Supply the values and resume, and it carries on.

This is the difference between a system that writes an email and one that
sends it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class InputType(str, Enum):
    TEXT = "text"
    MULTILINE = "multiline"
    EMAIL = "email"
    URL = "url"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    #: Never echoed back, never written to disk, never shown in exports.
    SECRET = "secret"


class InputError(ValueError):
    """A supplied value failed validation."""


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_TRUE = {"y", "yes", "true", "1", "on"}
_FALSE = {"n", "no", "false", "0", "off"}


@dataclass
class InputRequest:
    """One question the workflow needs answered before a step can run."""

    name: str
    prompt: str
    type: InputType = InputType.TEXT
    required: bool = True
    default: Optional[Any] = None
    #: Valid values when ``type`` is CHOICE.
    options: List[str] = field(default_factory=list)
    #: Why the step needs it -- shown to the operator.
    why: str = ""

    # Filled in when answered.
    value: Optional[Any] = None
    provided: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            try:
                self.type = InputType(self.type)
            except ValueError:
                self.type = InputType.TEXT
        if self.type is InputType.CHOICE and not self.options:
            # A choice with nothing to choose from is just text.
            self.type = InputType.TEXT

    # -- state -------------------------------------------------------------

    @property
    def satisfied(self) -> bool:
        """True when this no longer blocks the step."""
        if self.provided:
            return True
        if self.default is not None:
            return True
        return not self.required

    @property
    def effective_value(self) -> Any:
        if self.provided:
            return self.value
        return self.default

    @property
    def is_secret(self) -> bool:
        return self.type is InputType.SECRET

    # -- validation --------------------------------------------------------

    def coerce(self, raw: Any) -> Any:
        """Validate and convert a supplied value. Raises :class:`InputError`."""
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            if self.required and self.default is None:
                raise InputError(f"'{self.name}' is required")
            return self.default

        if self.type is InputType.NUMBER:
            try:
                text = str(raw).strip()
                return int(text) if re.fullmatch(r"[-+]?\d+", text) else float(text)
            except (TypeError, ValueError):
                raise InputError(f"'{self.name}' must be a number, got {raw!r}") from None

        if self.type is InputType.BOOLEAN:
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise InputError(f"'{self.name}' must be yes/no, got {raw!r}")

        text = str(raw).strip()

        if self.type is InputType.EMAIL and not _EMAIL_RE.match(text):
            raise InputError(f"'{self.name}' must be an email address, got {text!r}")
        if self.type is InputType.URL and not _URL_RE.match(text):
            raise InputError(f"'{self.name}' must be an http(s) URL, got {text!r}")
        if self.type is InputType.CHOICE and text not in self.options:
            raise InputError(
                f"'{self.name}' must be one of {', '.join(self.options)}, got {text!r}")
        return text

    def provide(self, raw: Any) -> Any:
        """Validate then record an answer. Returns the coerced value."""
        self.value = self.coerce(raw)
        self.provided = True
        return self.value

    def clear(self) -> None:
        self.value = None
        self.provided = False

    # -- rendering ---------------------------------------------------------

    def describe(self) -> str:
        """One-line summary for a prompt or form label."""
        bits = [self.type.value]
        if self.type is InputType.CHOICE:
            bits = [" | ".join(self.options)]
        if not self.required:
            bits.append("optional")
        if self.default is not None and not self.is_secret:
            bits.append(f"default: {self.default}")
        return f"{self.name} ({', '.join(bits)})"

    def to_dict(self, redact_secrets: bool = True) -> Dict[str, Any]:
        value = self.value
        if self.is_secret and redact_secrets and self.provided:
            value = "***"
        return {
            "name": self.name,
            "prompt": self.prompt,
            "type": self.type.value,
            "required": self.required,
            "default": None if self.is_secret else self.default,
            "options": self.options,
            "why": self.why,
            "provided": self.provided,
            "satisfied": self.satisfied,
            "value": value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputRequest":
        default = data.get("default")
        # Models frequently emit the *string* "null"/"none" instead of JSON
        # null, which would otherwise be shown to the user as a real default.
        if isinstance(default, str) and default.strip().lower() in {"null", "none", ""}:
            default = None

        request = cls(
            name=str(data.get("name", "")).strip(),
            prompt=str(data.get("prompt") or data.get("why") or data.get("name", "")).strip(),
            type=data.get("type", "text"),
            required=bool(data.get("required", True)),
            default=default,
            options=[str(o) for o in (data.get("options") or [])],
            why=str(data.get("why", "")).strip(),
        )
        if data.get("provided") and data.get("value") not in (None, "***"):
            request.value = data["value"]
            request.provided = True
        return request


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def unsatisfied(requests: Iterable[InputRequest]) -> List[InputRequest]:
    return [r for r in requests if not r.satisfied]


def values_of(requests: Iterable[InputRequest]) -> Dict[str, Any]:
    """The answers, keyed by name. Unanswered optional inputs are omitted."""
    collected: Dict[str, Any] = {}
    for request in requests:
        value = request.effective_value
        if value is not None:
            collected[request.name] = value
    return collected


def render_for_prompt(requests: Iterable[InputRequest]) -> str:
    """Render answers for inclusion in an agent's prompt.

    Secrets are deliberately withheld: an agent never needs the value of an
    API key, and putting one in a prompt would send it to the model provider
    and store it in the run's event log.
    """
    lines = []
    for request in requests:
        value = request.effective_value
        if value is None:
            continue
        if request.is_secret:
            lines.append(f"- {request.name}: (provided, withheld from this prompt)")
        else:
            lines.append(f"- {request.name}: {value}")
    return "\n".join(lines)


def parse_declared_inputs(raw: Any) -> List[InputRequest]:
    """Build InputRequests from planner JSON, skipping malformed entries."""
    requests: List[InputRequest] = []
    seen: set = set()
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        requests.append(InputRequest.from_dict(entry))
    return requests

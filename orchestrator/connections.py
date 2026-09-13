"""Reusable API connections -- the n8n "credentials + HTTP node" idea.

Declare a service once (base URL, auth, default headers) and every agent can
call it by name, without the model ever seeing a token:

    {
      "connections": {
        "stripe": {
          "base_url": "https://api.stripe.com/v1",
          "auth": {"type": "bearer", "token_env": "STRIPE_API_KEY"}
        }
      }
    }

An agent then emits::

    TOOL_DIRECTIVE: {"method": "GET", "path": "/customers", "query": {"limit": 3}}

**Secrets live in environment variables, never in the config file and never in
a prompt.** The connection file is safe to commit; it names the variables it
needs, and :mod:`orchestrator.requirements` reports any that are unset.

Safety follows the HTTP method rather than the service: a ``GET`` runs freely,
while ``POST``/``PUT``/``PATCH``/``DELETE`` are treated as irreversible and go
through the approval gate.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin

from .tools.base import Tool, ToolError

#: HTTP methods that only read. Everything else is gated as irreversible.
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

DEFAULT_CONFIG_PATH = "connections.json"


class ConnectionError_(ToolError):
    """Raised when a connection is misconfigured or its credentials are absent."""


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@dataclass
class AuthSpec:
    """How to authenticate. Values always come from the environment."""

    type: str = "none"  # none | bearer | basic | api_key | oauth2_client_credentials

    # bearer / api_key
    token_env: Optional[str] = None
    header: str = "Authorization"
    prefix: str = "Bearer "
    #: For api_key auth delivered as a query parameter instead of a header.
    query_param: Optional[str] = None

    # basic
    username_env: Optional[str] = None
    password_env: Optional[str] = None

    # oauth2 client credentials
    token_url: Optional[str] = None
    client_id_env: Optional[str] = None
    client_secret_env: Optional[str] = None
    scope: Optional[str] = None

    def __post_init__(self) -> None:
        self.type = (self.type or "none").strip().lower()

    # -- what this needs from the environment -----------------------------

    def required_env(self) -> List[str]:
        names = {
            "bearer": [self.token_env],
            "api_key": [self.token_env],
            "basic": [self.username_env, self.password_env],
            "oauth2_client_credentials": [self.client_id_env, self.client_secret_env],
        }.get(self.type, [])
        return [n for n in names if n]

    def is_satisfied(self) -> bool:
        if self.type == "none":
            return True
        return all(os.environ.get(name) for name in self.required_env())

    def missing_env(self) -> List[str]:
        return [name for name in self.required_env() if not os.environ.get(name)]

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable form. Deliberately contains variable *names* only."""
        return {"type": self.type, "required_env": self.required_env(),
                "satisfied": self.is_satisfied()}


class _TokenCache:
    """Caches OAuth2 tokens so every call does not re-authenticate."""

    def __init__(self) -> None:
        self._tokens: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._tokens.get(key)
        if entry and entry[1] > time.time():
            return entry[0]
        return None

    def set(self, key: str, token: str, expires_in: float) -> None:
        with self._lock:
            # Expire a minute early so a token never dies mid-flight.
            self._tokens[key] = (token, time.time() + max(30.0, expires_in - 60))

    def clear(self) -> None:
        with self._lock:
            self._tokens.clear()


_token_cache = _TokenCache()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


@dataclass
class ApiConnection:
    """One named external HTTP service."""

    name: str
    base_url: str
    auth: AuthSpec = field(default_factory=AuthSpec)
    headers: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    timeout_s: float = 30.0
    #: Extra query parameters sent on every request.
    default_query: Dict[str, str] = field(default_factory=dict)
    #: Methods to treat as safe, overriding the default read-only set. Use
    #: sparingly -- it removes the approval gate for those calls.
    safe_methods: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError(f"connection '{self.name}': base_url is required")
        self.base_url = self.base_url.rstrip("/") + "/"
        if isinstance(self.auth, dict):
            self.auth = AuthSpec(**self.auth)

    def is_safe(self, method: str) -> bool:
        allowed = {m.upper() for m in (self.safe_methods or SAFE_METHODS)}
        return method.upper() in allowed

    def url_for(self, path: str) -> str:
        return urljoin(self.base_url, str(path).lstrip("/"))

    def required_env(self) -> List[str]:
        return self.auth.required_env()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "base_url": self.base_url,
            "description": self.description, "auth": self.auth.to_dict(),
            "default_headers": sorted(self.headers),  # names only, not values
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ApiConnection":
        return cls(
            name=name,
            base_url=str(data.get("base_url", "")),
            auth=AuthSpec(**(data.get("auth") or {})),
            headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
            description=str(data.get("description", "")),
            timeout_s=float(data.get("timeout_s", 30.0)),
            default_query={str(k): str(v) for k, v in (data.get("default_query") or {}).items()},
            safe_methods=data.get("safe_methods"),
        )

    # -- request construction ---------------------------------------------

    def build_headers(self) -> Dict[str, str]:
        """Default headers plus whatever the auth scheme contributes."""
        headers = dict(self.headers)

        if self.auth.type == "bearer":
            token = self._env(self.auth.token_env)
            headers[self.auth.header] = f"{self.auth.prefix}{token}"

        elif self.auth.type == "api_key" and not self.auth.query_param:
            token = self._env(self.auth.token_env)
            headers[self.auth.header] = f"{self.auth.prefix}{token}".strip()

        elif self.auth.type == "basic":
            import base64

            user = self._env(self.auth.username_env)
            password = self._env(self.auth.password_env)
            encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        elif self.auth.type == "oauth2_client_credentials":
            headers["Authorization"] = f"Bearer {self._oauth_token()}"

        return headers

    def build_query(self) -> Dict[str, str]:
        query = dict(self.default_query)
        if self.auth.type == "api_key" and self.auth.query_param:
            query[self.auth.query_param] = self._env(self.auth.token_env)
        return query

    def _env(self, name: Optional[str]) -> str:
        if not name:
            raise ConnectionError_(
                f"connection '{self.name}': auth type '{self.auth.type}' is missing its "
                "environment variable name")
        value = os.environ.get(name)
        if not value:
            raise ConnectionError_(
                f"connection '{self.name}': environment variable '{name}' is not set")
        return value

    def _oauth_token(self) -> str:
        cached = _token_cache.get(self.name)
        if cached:
            return cached
        if not self.auth.token_url:
            raise ConnectionError_(f"connection '{self.name}': oauth2 needs a token_url")

        from .tools.builtin import _http

        payload = {
            "grant_type": "client_credentials",
            "client_id": self._env(self.auth.client_id_env),
            "client_secret": self._env(self.auth.client_secret_env),
        }
        if self.auth.scope:
            payload["scope"] = self.auth.scope

        response = _http("POST", self.auth.token_url, json_body=payload,
                         timeout=self.timeout_s)
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError_(
                f"connection '{self.name}': token endpoint returned non-JSON") from exc

        token = body.get("access_token")
        if not token:
            raise ConnectionError_(
                f"connection '{self.name}': token response had no access_token")
        _token_cache.set(self.name, token, float(body.get("expires_in", 3600)))
        return token


# ---------------------------------------------------------------------------
# The Tool wrapper
# ---------------------------------------------------------------------------


class ApiConnectionTool(Tool):
    """Exposes one :class:`ApiConnection` as a tool agents can call."""

    side_effect = True
    irreversible = True  # overridden per call by is_irreversible()
    fallbacks: List[str] = []

    def __init__(self, connection: ApiConnection):
        super().__init__()
        self.connection = connection
        self.name = f"api_{connection.name}"
        # Each API gets its own capability on purpose: falling back from one
        # service to an unrelated one would be nonsense, and dangerous.
        self.capability = f"api:{connection.name}"
        self.description = (connection.description
                            or f"HTTP calls to {connection.base_url}")

    # -- introspection -----------------------------------------------------

    def is_live(self) -> bool:
        return self.connection.auth.is_satisfied()

    def unavailable_reason(self) -> Optional[str]:
        # Absent credentials do NOT make the tool unavailable -- like every
        # other adapter here it degrades to a labelled simulation, and the
        # requirements report is what tells you it is not doing real work.
        return "marked broken" if self.is_broken else None

    def missing_credentials(self) -> List[str]:
        return self.connection.auth.missing_env()

    def is_irreversible(self, task: str, context: Optional[dict] = None) -> bool:
        return not self.connection.is_safe(self._directive(task).get("method", "GET"))

    def preview(self, task: str, context: Optional[dict] = None) -> str:
        directive = self._directive(task)
        method = directive.get("method", "GET").upper()
        url = self.connection.url_for(directive.get("path", "/"))
        mode = "REAL" if self.is_live() else "simulated"
        body = directive.get("body")
        summary = f" body={json.dumps(body)[:120]}" if body else ""
        return f"{method} {url} [{mode}]{summary}"

    def prompt_hint(self) -> str:
        auth = self.connection.auth
        auth_note = ("no authentication" if auth.type == "none"
                     else f"authenticated automatically ({auth.type}); "
                          "never include credentials yourself")
        return (
            f"{self.description}\n"
            f"Base URL: {self.connection.base_url} ({auth_note}).\n"
            'Call it with: TOOL_DIRECTIVE: {"method": "GET|POST|PUT|PATCH|DELETE", '
            '"path": "/resource", "query": {...}, "body": {...}}\n'
            "Give 'path' relative to the base URL. Non-GET calls need human approval."
        )

    # -- execution ---------------------------------------------------------

    @staticmethod
    def _directive(task: str) -> Dict[str, Any]:
        from .tools.builtin import _directive

        return _directive(task)

    def _run(self, task: str, context: Optional[dict] = None) -> str:
        from .tools.builtin import _http

        directive = self._directive(task)
        # Operator-supplied inputs may steer the call too.
        directive.update({k: v for k, v in ((context or {}).get("inputs") or {}).items()
                          if k in {"method", "path", "query", "body"}})

        path = directive.get("path")
        if not path:
            return (f"[{self.name}] no 'path' in TOOL_DIRECTIVE; nothing called. "
                    f"{self.prompt_hint().splitlines()[1]}")

        method = str(directive.get("method", "GET")).upper()
        url = self.connection.url_for(path)

        query = self.connection.build_query()
        query.update({str(k): str(v) for k, v in (directive.get("query") or {}).items()})
        if query:
            url = f"{url}{'&' if '?' in url else '?'}{urlencode(query)}"

        if not self.is_live():
            missing = ", ".join(self.connection.auth.missing_env()) or "credentials"
            return (f"[simulated:{self.name}] would call {method} {url} "
                    f"(set {missing} to make this real)")

        response = _http(method, url, headers=self.connection.build_headers(),
                         json_body=directive.get("body"), timeout=self.connection.timeout_s)
        body = response.text or ""
        return f"[{self.name}] {response.status_code} {method} {url}\n{body[:2000]}"


# ---------------------------------------------------------------------------
# Loading and registration
# ---------------------------------------------------------------------------


def load_connections(path: str = DEFAULT_CONFIG_PATH) -> List[ApiConnection]:
    """Read connection definitions. A missing file means none are configured."""
    config_path = Path(path)
    if not config_path.exists():
        return []

    data = json.loads(config_path.read_text(encoding="utf-8"))
    entries = data.get("connections", data if isinstance(data, dict) else {})
    connections: List[ApiConnection] = []
    for name, entry in entries.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue  # comment keys
        try:
            connections.append(ApiConnection.from_dict(name, entry))
        except (ValueError, TypeError) as exc:
            print(f"[connections] skipping '{name}': {exc}")
    return connections


def attach_connections(tool_manager: Any, connections: Optional[List[ApiConnection]] = None,
                       config_path: str = DEFAULT_CONFIG_PATH) -> List[str]:
    """Register every configured API connection as a tool. Returns tool names."""
    connections = connections if connections is not None else load_connections(config_path)
    registered = []
    for connection in connections:
        tool = ApiConnectionTool(connection)
        tool_manager.register(tool)
        registered.append(tool.name)
    return registered


def connection_requirements(connections: List[ApiConnection]) -> List[Any]:
    """The credentials these connections need, as Requirement objects."""
    from .requirements import Requirement, RequirementKind

    requirements = []
    for connection in connections:
        for env_name in connection.required_env():
            requirements.append(Requirement(
                name=env_name,
                kind=RequirementKind.CREDENTIAL,
                why=f"authenticates the '{connection.name}' API connection",
                setup=f"set {env_name} in .env.local",
                # Without it the connection simulates rather than failing, so
                # the run is degraded rather than blocked.
                optional=True,
            ))
    return requirements

"""Small control-plane security policy with no external dependency."""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple
from .identity import Identity, verify_oidc_token

ROLES = {"viewer": 0, "operator": 1, "admin": 2}


class SecurityPolicy:
    """Resolve API keys to roles and enforce a per-identity sliding window."""

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def configured_keys(self) -> Dict[str, str]:
        keys: Dict[str, str] = {}
        legacy = os.environ.get("ORCHESTRATOR_API_KEY", "").strip()
        if legacy:
            keys[legacy] = "admin"
        raw = os.environ.get("ORCHESTRATOR_API_KEYS", "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                for key, value in parsed.items():
                    role = value.get("role") if isinstance(value, dict) else value
                    if str(key).strip() and role in ROLES:
                        keys[str(key)] = role
            except (json.JSONDecodeError, AttributeError):
                pass
        return keys

    def api_key_identity(self, supplied: str) -> Optional[Identity]:
        raw = os.environ.get("ORCHESTRATOR_API_KEYS", "").strip()
        if raw:
            try:
                for key, value in json.loads(raw).items():
                    if supplied and hmac.compare_digest(supplied, str(key)):
                        if isinstance(value, dict):
                            role = str(value.get("role", "viewer"))
                            if role in ROLES:
                                return Identity(str(value.get("user") or "api-key"),
                                                str(value.get("organization") or "default"),
                                                role, "api_key")
            except (json.JSONDecodeError, AttributeError):
                pass
        role = self.role_for(supplied)
        return Identity("api-key", "default", role, "api_key") if role else None

    def role_for(self, supplied: str) -> Optional[str]:
        keys = self.configured_keys()
        if not keys:
            return "admin"
        for key, role in keys.items():
            if supplied and hmac.compare_digest(supplied, key):
                return role
        return None

    def authenticate(self, supplied: str) -> Optional[Identity]:
        key_identity = self.api_key_identity(supplied)
        if key_identity:
            return key_identity
        identity = verify_oidc_token(supplied)
        return identity if identity and identity.role in ROLES else None

    @staticmethod
    def required_role(method: str, path: str) -> str:
        if path.startswith("/api/capabilities/") and path.endswith("/connect"):
            return "admin"
        if path.startswith("/api/audit"):
            return "admin"
        if method.upper() in {"GET", "HEAD", "OPTIONS"}:
            return "viewer"
        if path.startswith("/api/setup/credentials") or path.startswith("/api/templates/") and method.upper() == "DELETE":
            return "admin"
        return "operator"

    def allowed(self, role: str, required: str) -> bool:
        return ROLES.get(role, -1) >= ROLES[required]

    def check_rate(self, identity: str) -> Tuple[bool, int]:
        limit = int(os.environ.get("ORCHESTRATOR_RATE_LIMIT_PER_MINUTE", "0") or 0)
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            hits = self._hits[identity]
            while hits and hits[0] <= now - 60:
                hits.popleft()
            if len(hits) >= limit:
                return False, max(1, int(60 - (now - hits[0])))
            hits.append(now)
        return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

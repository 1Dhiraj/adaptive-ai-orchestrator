"""Verified API-key and OIDC identities for the control plane."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class Identity:
    subject: str
    tenant_id: str
    role: str
    provider: str


@lru_cache(maxsize=4)
def _jwk_client(url: str):
    import jwt
    return jwt.PyJWKClient(url)


def verify_oidc_token(token: str) -> Optional[Identity]:
    """Verify signature, issuer, audience and expiry against an OIDC JWKS."""
    issuer = os.environ.get("OIDC_ISSUER", "").rstrip("/")
    audience = os.environ.get("OIDC_AUDIENCE", "")
    jwks_url = os.environ.get("OIDC_JWKS_URL", "") or (f"{issuer}/.well-known/jwks.json" if issuer else "")
    if not issuer or not audience or not jwks_url or token.count(".") != 2:
        return None
    try:
        import jwt
        key = _jwk_client(jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["RS256", "ES256"],
                            audience=audience, issuer=issuer,
                            options={"require": ["exp", "iat", "sub"]})
    except Exception:  # invalid identity is deliberately indistinguishable
        return None
    tenant_claim = os.environ.get("OIDC_TENANT_CLAIM", "org_id")
    role_claim = os.environ.get("OIDC_ROLE_CLAIM", "role")
    return Identity(subject=str(claims["sub"]),
                    tenant_id=str(claims.get(tenant_claim) or "default"),
                    role=str(claims.get(role_claim) or "viewer").lower(), provider="oidc")


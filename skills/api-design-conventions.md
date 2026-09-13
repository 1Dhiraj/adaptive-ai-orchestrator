---
name: api-design-conventions
description: House style for REST API design
roles: backend
keywords: api, endpoint, rest
---

# API design conventions

- Resource paths are plural nouns: `/users`, not `/user` or `/getUsers`.
- Use standard HTTP verbs: GET (read), POST (create), PUT (replace),
  PATCH (partial update), DELETE (remove). Never a verb in the path.
- Every list endpoint supports `?limit=` and `?cursor=` for pagination.
  Do not use offset-based pagination.
- Errors are returned as RFC 7807 problem+json:
  `{"type": "...", "title": "...", "status": 400, "detail": "..."}`.
- Auth: bearer tokens in the `Authorization` header. Never accept a token
  as a query parameter.
- Version in the path: `/v1/...`. Breaking changes get a new version,
  never a breaking change to an existing one.
- Idempotency: POST endpoints that create resources must accept an
  `Idempotency-Key` header.

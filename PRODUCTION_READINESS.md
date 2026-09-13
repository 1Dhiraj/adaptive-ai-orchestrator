# Production readiness record

This document separates implemented code from checks that need external infrastructure.

| Requirement | Implemented | Verification in this repository |
|---|---|---|
| Shared PostgreSQL persistence | Yes | SQLAlchemy/PostgreSQL backend, Alembic schema, tenant isolation test; real test runs in CI with PostgreSQL |
| Distributed workers | Yes | Durable jobs, Celery/Redis broker, late acknowledgement, retry and queue routing |
| Users and organizations | Yes | Tenant-scoped runs, jobs, templates, schedules, webhooks, audit records and role-based access |
| OAuth/OIDC and SSO | Yes | Authorization Code flow with PKCE, JWKS validation, secure session cookie and logout |
| Encrypted credentials and rotation | Yes | Fernet vault with tenant isolation and key rotation tests |
| Production secret manager | Yes | AWS Secrets Manager backend and workload configuration boundary |
| Dedicated connectors | Yes | GitHub, PostgreSQL, CI, Slack, Gmail, SendGrid, Stripe and S3, plus generic REST and MCP |
| Worker isolation | Yes | Non-root standard and Playwright images, restricted browser queue, non-admin Windows desktop worker |
| Deployment and migrations | Yes | Docker Compose, Kubernetes resources and Alembic migration job |
| Load/security/recovery testing | Yes | Concurrent state writes, tenant boundaries, rate limits, durable queue recovery and failure tests |
| Monitoring | Yes | Prometheus endpoint and OpenTelemetry OTLP export configuration |
| CrewAI and AutoGen benchmarks | Yes | Real framework adapters using the same deterministic workload and fresh six-system reports |
| Browser/desktop workers | Yes | Isolated Playwright worker and opt-in Hermes desktop worker script |

## Verified locally

- 663 non-slow tests passed; two PostgreSQL-dependent tests were skipped because no local PostgreSQL service was available.
- 38 real MCP subprocess tests passed.
- Four npm launcher tests passed.
- The npm launcher selected a free port, started the standalone Windows binary, loaded the dashboard and MCP tools, and used a private per-user SQLite/workspace directory.
- An installable Windows npm archive was built as `adaptive-ai-orchestrator-1.0.0.tgz`; the release workflow builds Windows, macOS and Linux runtimes before publishing.
- Docker Compose, Kubernetes, Prometheus and OpenTelemetry YAML parsed successfully.
- Dashboard JavaScript passed Node syntax validation.
- Production Python imports passed, including Celery, PostgreSQL, Alembic and boto3.
- The live local dashboard health and Prometheus endpoints returned HTTP 200.
- The six-system benchmark completed and wrote reports under `artifacts/final-benchmark`.

## Requires deployment infrastructure

- Run the PostgreSQL integration tests and Alembic migration against the intended managed database.
- Build and scan the two container images, then run the Docker Compose or Kubernetes smoke test.
- Configure an OIDC provider and verify its real claims and logout behavior.
- Configure AWS workload identity, Stripe, SendGrid, S3 and any other connector accounts used by the deployment.
- Run load and recovery tests at the expected production traffic level.
- Validate the Hermes worker on the Windows machine that will run desktop automation.

These are environment acceptance checks. The project does not claim they passed on this workstation, where Docker and a PostgreSQL service are not installed.

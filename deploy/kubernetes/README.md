# Kubernetes deployment

1. Build and push both `Dockerfile` and `Dockerfile.playwright` images.
2. Replace the image names and managed Redis hostname in `orchestrator.yaml`.
3. Create `orchestrator-runtime` through External Secrets, Sealed Secrets, or your cloud secret operator. It must contain `ORCHESTRATOR_STATE_URL`, OIDC settings, and the secret-manager workload identity configuration. Do not commit those values.
4. Run the migration Job and wait for completion before deploying the API and workers.
5. Deploy the API, standard workers, and the isolated browser-worker image declared in the manifest. Place browser workers on a separate node pool when possible.
6. Run Windows desktop workers with `scripts/run_desktop_worker.ps1`; they consume only queue `desktop` and refuse Administrator sessions.

PostgreSQL, Redis, and object storage should be managed services with private networking, backups, TLS, and provider-managed encryption.

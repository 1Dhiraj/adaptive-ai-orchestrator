"""production persistence schema"""
from alembic import op
from orchestrator.state import _POSTGRES_SCHEMA

revision = "0001_production_schema"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    for statement in _POSTGRES_SCHEMA.split(";"):
        if statement.strip():
            op.execute(statement)

def downgrade():
    for table in ("execution_jobs", "audit_log", "workflow_templates", "workflow_webhooks",
                  "workflow_schedules", "pending_actions", "events", "step_results", "runs"):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


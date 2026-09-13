from __future__ import annotations
import os
from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
config.set_main_option("sqlalchemy.url", os.environ.get(
    "ORCHESTRATOR_STATE_URL", config.get_main_option("sqlalchemy.url")))

def run_migrations_offline():
    context.configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True)
    with context.begin_transaction(): context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section),
                                     prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction(): context.run_migrations()

run_migrations_offline() if context.is_offline_mode() else run_migrations_online()


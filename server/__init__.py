"""FastAPI control plane and live dashboard for the orchestrator.

Deliberately empty of re-exports: binding ``app`` here would shadow the
``server.app`` submodule and break ``import server.app``. Import from the
submodule instead::

    from server.app import app, manager
"""

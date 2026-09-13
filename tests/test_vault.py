from __future__ import annotations

import os

from cryptography.fernet import Fernet

from orchestrator.tenancy import tenant_scope
from orchestrator.vault import CredentialVault


def test_keyring_backend_is_tenant_scoped(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from orchestrator.tenancy import tenant_scope

    stored = {}
    fake = SimpleNamespace(
        set_password=lambda service, name, value: stored.__setitem__((service, name), value),
        get_password=lambda service, name: stored.get((service, name)),
    )
    monkeypatch.setitem(sys.modules, "keyring", fake)
    monkeypatch.setenv("ORCHESTRATOR_SECRET_BACKEND", "keyring")
    vault = CredentialVault(tmp_path / "unused.enc")
    with tenant_scope("alpha"):
        vault.set("GITHUB_TOKEN", "alpha-secret")
    with tenant_scope("beta"):
        assert vault.get("GITHUB_TOKEN") is None
        vault.set("GITHUB_TOKEN", "beta-secret")
    with tenant_scope("alpha"):
        assert vault.get("GITHUB_TOKEN") == "alpha-secret"


def test_local_vault_encrypts_and_activates_only_current_tenant(tmp_path, monkeypatch):
    path = tmp_path / "credentials.enc"
    monkeypatch.setenv("ORCHESTRATOR_VAULT_KEY", Fernet.generate_key().decode())
    vault = CredentialVault(path)
    with tenant_scope("alpha"):
        vault.set("alpha:SERVICE_TOKEN", "alpha-secret")
    with tenant_scope("beta"):
        vault.set("beta:SERVICE_TOKEN", "beta-secret")
        assert vault.activate() == 1
        assert os.environ["SERVICE_TOKEN"] == "beta-secret"
    assert b"alpha-secret" not in path.read_bytes()
    assert b"beta-secret" not in path.read_bytes()


def test_vault_key_rotation_preserves_values(tmp_path, monkeypatch):
    old_key, new_key = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    monkeypatch.setenv("ORCHESTRATOR_VAULT_KEY", old_key)
    vault = CredentialVault(tmp_path / "credentials.enc")
    vault.set("default:TOKEN", "kept")
    vault.rotate(new_key)
    monkeypatch.setenv("ORCHESTRATOR_VAULT_KEY", new_key)
    assert vault.get("default:TOKEN") == "kept"


def test_missing_master_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_VAULT_KEY", raising=False)
    vault = CredentialVault(tmp_path / "credentials.enc")
    try:
        vault.set("TOKEN", "secret")
    except RuntimeError as exc:
        assert "ORCHESTRATOR_VAULT_KEY" in str(exc)
    else:
        raise AssertionError("plaintext storage must never be used without a vault key")

"""Encrypted credential vault with a production secret-manager boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .config import default_data_dir


class CredentialVault:
    """Encrypt local secrets with Fernet; optionally read AWS Secrets Manager."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or default_data_dir() / ".credentials.enc"

    @staticmethod
    def _keyring_name(name: str) -> str:
        from .tenancy import current_tenant
        return f"{current_tenant()}:{name}"

    @staticmethod
    def _keyring():
        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError(
                "OS credential storage is unavailable; install the local dependency 'keyring'"
            ) from exc
        return keyring

    @staticmethod
    def _cipher():
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise RuntimeError("cryptography is required for encrypted credentials") from exc
        key = os.environ.get("ORCHESTRATOR_VAULT_KEY", "")
        if not key:
            raise RuntimeError("ORCHESTRATOR_VAULT_KEY must be supplied by the deployment secret manager")
        return Fernet(key.encode())

    def _read(self) -> Dict[str, str]:
        if not self.path.exists():
            return {}
        return json.loads(self._cipher().decrypt(self.path.read_bytes()).decode())

    def set(self, name: str, value: str) -> None:
        backend = os.environ.get("ORCHESTRATOR_SECRET_BACKEND", "local")
        if backend == "keyring":
            self._keyring().set_password(
                "adaptive-ai-orchestrator", self._keyring_name(name), value)
            return
        if backend == "aws":
            import boto3
            client = boto3.client("secretsmanager")
            secret_id = self._aws_secret_id()
            try:
                values = json.loads(client.get_secret_value(SecretId=secret_id)["SecretString"])
                values[name] = value
                client.put_secret_value(SecretId=secret_id, SecretString=json.dumps(values))
            except client.exceptions.ResourceNotFoundException:
                client.create_secret(Name=secret_id, SecretString=json.dumps({name: value}))
            return
        values = self._read()
        values[name] = value
        self.path.write_bytes(self._cipher().encrypt(json.dumps(values).encode()))

    def get(self, name: str) -> Optional[str]:
        backend = os.environ.get("ORCHESTRATOR_SECRET_BACKEND", "local")
        if backend == "keyring":
            return self._keyring().get_password(
                "adaptive-ai-orchestrator", self._keyring_name(name))
        if backend == "aws":
            import boto3
            payload = boto3.client("secretsmanager").get_secret_value(SecretId=self._aws_secret_id())
            return json.loads(payload["SecretString"]).get(name)
        return self._read().get(name)

    @staticmethod
    def _aws_secret_id() -> str:
        from .tenancy import current_tenant
        base = os.environ.get("ORCHESTRATOR_AWS_SECRET_ID", "adaptive-orchestrator")
        return f"{base}/{current_tenant()}"

    def activate(self) -> int:
        """Load this organization’s vault values into the current worker."""
        backend = os.environ.get("ORCHESTRATOR_SECRET_BACKEND", "local")
        if backend == "aws":
            import boto3
            try:
                payload = boto3.client("secretsmanager").get_secret_value(
                    SecretId=self._aws_secret_id())
                values = json.loads(payload["SecretString"])
            except Exception:  # absence leaves tools honestly unconfigured
                values = {}
        elif backend == "keyring":
            # Keyring APIs cannot enumerate secrets portably. Activate the
            # credential names understood by the setup catalogue.
            from .setup import CREDENTIAL_GUIDES
            try:
                values = {name: value for name in CREDENTIAL_GUIDES
                          if (value := self.get(name)) is not None}
            except Exception:
                # Linux without Secret Service should still open the app; a
                # save attempt returns the actionable keyring error.
                values = {}
        else:
            from .tenancy import current_tenant
            prefix = f"{current_tenant()}:"
            values = {k[len(prefix):]: v for k, v in self._read().items() if k.startswith(prefix)}
        os.environ.update({str(k): str(v) for k, v in values.items()})
        from .config import reload_settings
        reload_settings()
        return len(values)

    def rotate(self, new_key: str) -> None:
        values = self._read()
        old = os.environ.get("ORCHESTRATOR_VAULT_KEY")
        try:
            os.environ["ORCHESTRATOR_VAULT_KEY"] = new_key
            self.path.write_bytes(self._cipher().encrypt(json.dumps(values).encode()))
        finally:
            if old is None:
                os.environ.pop("ORCHESTRATOR_VAULT_KEY", None)
            else:
                os.environ["ORCHESTRATOR_VAULT_KEY"] = old

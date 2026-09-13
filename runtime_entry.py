"""Entry point used by PyInstaller for npm release binaries."""
from orchestrator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
